"""Reinforcement-learning traders on the same market model as every other entry.

Environment (training rows only): observation = standardized SEQ_COLUMNS features at bar t + current position;
action = SHORT / FLAT / LONG (continuous agents: a in [-1, 1], |a| < 1/3 -> FLAT). Reward per bar (bps / 100):

    new_pos * r_next[t]  -  side cost(t) * |new_pos - pos|  -  new_pos * funding(t)
    -  lambda_dd * (increase of the episode drawdown)  -  lambda_turn * |new_pos - pos|

using alpha.tournament.backtesting.engine.Market (same fees, spread, impact, latency, funding as the backtest).
It never sees directional accuracy. Episodes: random 2,048-bar windows of the training period.
Inference rolls the deterministic policy forward through the requested rows (flat at the start of each block);
the model's 'score' is the position itself, so the shared policy with hold = 1 reproduces it exactly.

    DQN / PPO / A2C / SAC / TD3 / DDPG    stable-baselines3
    DoubleDQN / DuelingDQN                 own small PyTorch DQN (double-Q target / dueling head)
"""

import numpy as np

from alpha.tournament.features.groups import SEQ_COLUMNS
from alpha.tournament.models.base import BaseTradingModel, Preprocessor, train_rows

EPISODE = 2048
TIMESTEPS = 100_000


def _env_cls():
    import gymnasium as gym

    class TradingEnv(gym.Env):
        def __init__(self, obs, ret, cost, fund, rows, continuous, lam_dd, lam_turn, seed):
            super().__init__()
            self.obs_m, self.ret, self.cost, self.fund = obs, ret, cost, fund
            self.rows = rows
            self.continuous = continuous
            self.lam_dd, self.lam_turn = lam_dd, lam_turn
            self.rng = np.random.default_rng(seed)
            n_obs = obs.shape[1] + 1
            self.observation_space = gym.spaces.Box(-10, 10, (n_obs,), np.float32)
            self.action_space = (gym.spaces.Box(-1, 1, (1,), np.float32) if continuous else gym.spaces.Discrete(3))

        def _o(self):
            return np.append(self.obs_m[self.rows[self.i]], self.pos).astype(np.float32)

        def reset(self, seed=None, options=None):
            self.start = int(self.rng.integers(0, max(1, len(self.rows) - EPISODE)))
            self.i, self.pos, self.eq, self.peak = self.start, 0.0, 0.0, 0.0
            return self._o(), {}

        def step(self, a):
            new = action_to_pos(a, self.continuous)
            t = self.rows[self.i]
            pnl = new * self.ret[t] - self.cost[t] * abs(new - self.pos) - new * self.fund[t]
            self.eq += pnl
            dd_before = self.peak - (self.eq - pnl)
            self.peak = max(self.peak, self.eq)
            dd_inc = max(0.0, (self.peak - self.eq) - dd_before)
            r = (pnl - self.lam_dd * dd_inc - self.lam_turn * abs(new - self.pos)) / 100.0
            self.pos = new
            self.i += 1
            done = self.i >= min(len(self.rows) - 1, self.start + EPISODE)
            return self._o(), float(r), done, False, {}

    return TradingEnv


def action_to_pos(a, continuous: bool) -> float:
    if continuous:
        v = float(np.asarray(a).ravel()[0])
        return 0.0 if abs(v) < 1 / 3 else float(np.sign(v))
    return float(int(np.asarray(a).ravel()[0]) - 1)


class _RL(BaseTradingModel):
    family = "rl"
    complexity = 4
    input_kind = "sequence"
    target_kinds = ("reg",)
    output_units = "bps"
    continuous = False
    experimental = True

    def _obs(self, X):
        return self.pre.transform(X[SEQ_COLUMNS])

    def _env(self, X, train):
        m = self.context.market
        n = len(X)
        ret = np.nan_to_num(m.ret_next[:n])
        cost = m.costs.total[:n]
        fund = m.funding_bar[:n]
        rows = np.flatnonzero(train & np.isfinite(m.ret_next[:n]))
        return _env_cls()(self._obs(X), ret, cost, fund, rows, self.continuous,
                          float(self.params.get("lambda_dd", 0.1)), float(self.params.get("lambda_turn", 1.0)),
                          self.seed)

    def fit(self, X, y, train, val):
        ti = np.flatnonzero(train)
        self.pre = Preprocessor().fit(X[SEQ_COLUMNS].iloc[ti])
        env = self._env(X, train)
        self.agent = self.make(env)
        self.agent.learn(total_timesteps=int(self.params.get("timesteps", TIMESTEPS)))
        return self

    def act(self, o):
        a, _ = self.agent.predict(o, deterministic=True)
        return a

    def _raw(self, X, rows):
        O = self._obs(X)
        out = []
        idx = np.flatnonzero(rows)
        pos, prev = 0.0, None
        for i in idx:
            if prev is not None and i != prev + 1:  # a new contiguous block starts flat
                pos = 0.0
            pos = action_to_pos(self.act(np.append(O[i], pos).astype(np.float32)), self.continuous)
            out.append(pos)
            prev = i
        return np.array(out)

    def make(self, env):
        raise NotImplementedError


def _sb3(algo: str, env, seed: int, **kw):
    import stable_baselines3 as sb3

    cls = getattr(sb3, algo)
    return cls("MlpPolicy", env, seed=seed, verbose=0, device="cpu", policy_kwargs={"net_arch": [64, 64]}, **kw)


class DQN(_RL):
    name = "dqn"

    def make(self, env):
        return _sb3("DQN", env, self.seed, learning_starts=2000, buffer_size=50_000, exploration_fraction=0.3,
                    target_update_interval=1000, train_freq=4)


class PPO(_RL):
    name = "ppo"

    def make(self, env):
        return _sb3("PPO", env, self.seed, n_steps=2048, batch_size=256, ent_coef=0.01)


class A2C(_RL):
    name = "a2c"

    def make(self, env):
        return _sb3("A2C", env, self.seed, n_steps=32, ent_coef=0.01)


class SAC(_RL):
    name, continuous = "sac", True

    def make(self, env):
        return _sb3("SAC", env, self.seed, learning_starts=2000, buffer_size=50_000)


class TD3(_RL):
    name, continuous = "td3", True

    def make(self, env):
        return _sb3("TD3", env, self.seed, learning_starts=2000, buffer_size=50_000)


class DDPG(_RL):
    name, continuous = "ddpg", True

    def make(self, env):
        return _sb3("DDPG", env, self.seed, learning_starts=2000, buffer_size=50_000)


class _OwnDQN(_RL):
    """Own DQN with double-Q targets and / or a dueling head."""
    double, dueling = True, False

    def make(self, env):
        return None

    def fit(self, X, y, train, val):
        import torch

        torch.manual_seed(self.seed)
        ti = np.flatnonzero(train)
        self.pre = Preprocessor().fit(X[SEQ_COLUMNS].iloc[ti])
        env = self._env(X, train)
        n_obs = env.observation_space.shape[0]
        self.q = self._net(n_obs)
        tgt = self._net(n_obs)
        tgt.load_state_dict(self.q.state_dict())
        opt = torch.optim.Adam(self.q.parameters(), lr=5e-4)
        rng = np.random.default_rng(self.seed)
        cap, buf, k = 50_000, [], 0
        o, _ = env.reset()
        steps = int(self.params.get("timesteps", TIMESTEPS))
        for step in range(steps):
            eps = max(0.05, 1 - step / (0.3 * steps))
            if rng.random() < eps:
                a = int(rng.integers(3))
            else:
                with torch.no_grad():
                    a = int(self.q(torch.tensor(o)[None]).argmax())
            o2, r, done, _, _ = env.step(a)
            if len(buf) < cap:
                buf.append((o, a, r, o2, done))
            else:
                buf[k % cap] = (o, a, r, o2, done)
            k += 1
            o = env.reset()[0] if done else o2
            if step > 2000 and step % 4 == 0:
                b = [buf[i] for i in rng.integers(0, len(buf), 256)]
                O, A, R, O2, Dn = (torch.tensor(np.array(x)) for x in zip(*b))
                with torch.no_grad():
                    if self.double:
                        a2 = self.q(O2).argmax(1, keepdim=True)
                        nq = tgt(O2).gather(1, a2).squeeze(1)
                    else:
                        nq = tgt(O2).max(1).values
                    y_ = R.float() + 0.99 * nq * (1 - Dn.float())
                qv = self.q(O).gather(1, A.long()[:, None]).squeeze(1)
                loss = torch.nn.functional.smooth_l1_loss(qv, y_)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if step % 1000 == 0:
                tgt.load_state_dict(self.q.state_dict())
        return self

    def _net(self, n_obs):
        import torch.nn as nn

        dueling = self.dueling

        class Q(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = nn.Sequential(nn.Linear(n_obs, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
                self.v = nn.Linear(64, 1)
                self.a = nn.Linear(64, 3)

            def forward(self, x):
                h = self.body(x.float())
                a = self.a(h)
                return self.v(h) + a - a.mean(1, keepdim=True) if dueling else a

        return Q()

    def act(self, o):
        import torch

        with torch.no_grad():
            return int(self.q(torch.tensor(o)[None]).argmax())


class DoubleDQN(_OwnDQN):
    name, double, dueling = "double_dqn", True, False


class DuelingDQN(_OwnDQN):
    name, double, dueling = "dueling_dqn", True, True
