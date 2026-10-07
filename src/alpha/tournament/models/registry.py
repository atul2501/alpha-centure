"""Model registry: name -> class path. Adding a model = one line here + a class implementing BaseTradingModel."""

import importlib

M = "alpha.tournament.models."
REGISTRY: dict[str, str] = {
    # rules / naive
    "flat": M + "rules.rules.Flat", "buy_hold": M + "rules.rules.BuyHold", "random": M + "rules.rules.RandomEntries",
    "momentum": M + "rules.rules.Momentum", "reversal": M + "rules.rules.Reversal",
    "breakout": M + "rules.rules.Breakout", "flow_follow": M + "rules.rules.FlowFollow",
    # linear
    "linear_regression": M + "linear.linear.LinearReg", "ridge": M + "linear.linear.Ridge",
    "lasso": M + "linear.linear.Lasso", "elasticnet": M + "linear.linear.ElasticNet",
    "logistic": M + "linear.linear.Logistic", "linear_svm": M + "linear.linear.LinearSVM", "svr": M + "linear.linear.SVR",
    # tree
    "decision_tree": M + "tree.tree.DecisionTree", "random_forest": M + "tree.tree.RandomForest",
    "extra_trees": M + "tree.tree.ExtraTrees", "gradient_boosting": M + "tree.tree.GradientBoosting",
    "hist_gb": M + "tree.tree.HistGB", "xgboost": M + "tree.tree.XGBoost", "lightgbm": M + "tree.tree.LightGBM",
    "catboost": M + "tree.tree.CatBoost",
    # classical
    "ar": M + "classical.classical.AR", "ma": M + "classical.classical.MA", "arma": M + "classical.classical.ARMA",
    "arima": M + "classical.classical.ARIMA", "var": M + "classical.classical.VAR",
    "garch": M + "classical.classical.GARCH", "egarch": M + "classical.classical.EGARCH",
    "gjr_garch": M + "classical.classical.GJRGARCH", "kalman": M + "classical.classical.KalmanTrend",
    "state_space": M + "classical.classical.StateSpaceTrend",
    # regime (mode 1: direct trading)
    "hmm": M + "regime.regime.HMMDirect", "gaussian_hmm": M + "regime.regime.GaussianHMMDirect",
    "multivariate_hmm": M + "regime.regime.MultiHMMDirect", "markov_switching": M + "regime.regime.MSDirect",
    "hsmm": M + "regime.regime.HSMMDirect", "explicit_duration_hmm": M + "regime.regime.EDHMMDirect",
    "slds": M + "regime.regime.SLDSDirect", "switching_kalman": M + "regime.regime.SwitchingKalmanDirect",
    "switching_state_space": M + "regime.regime.SwitchingSSMDirect",
    # hybrids (regime mode 2 -> predictor; sequence model -> tree)
    "hybrid": M + "ensemble.hybrid.RegimeHybrid",
    # deep
    "mlp": M + "deep.nets.MLP", "dnn": M + "deep.nets.DNN", "cnn1d": M + "deep.nets.CNN1D", "tcn": M + "deep.nets.TCN",
    "rnn": M + "deep.nets.RNN", "lstm": M + "deep.nets.LSTM", "gru": M + "deep.nets.GRU",
    "bilstm": M + "deep.nets.BiLSTM", "bigru": M + "deep.nets.BiGRU",
    # transformer family
    "transformer": M + "transformer.nets.VanillaTransformer", "temporal_transformer": M + "transformer.nets.TemporalTransformer",
    "patchtst": M + "transformer.nets.PatchTST", "dlinear": M + "transformer.nets.DLinear",
    "nlinear": M + "transformer.nets.NLinear", "itransformer": M + "transformer.nets.ITransformer",
    "tft": M + "transformer.nets.TFTLite", "informer": M + "transformer.nets.InformerLite",
    "autoformer": M + "transformer.nets.AutoformerLite", "fedformer": M + "transformer.nets.FEDformerLite",
    "timesnet": M + "transformer.nets.TimesNetLite", "crossformer": M + "transformer.nets.CrossformerLite",
    # state space
    "s4": M + "ssm.nets.S4D", "s5": M + "ssm.nets.S5", "mamba": M + "ssm.nets.Mamba", "mamba2": M + "ssm.nets.Mamba2",
    "mamba_attention": M + "ssm.nets.MambaAttention", "mamba_cnn": M + "ssm.nets.MambaCNN",
    "mamba_transformer": M + "ssm.nets.MambaTransformer",
    # foundation (zero-shot, experimental)
    "chronos": M + "foundation.fm.Chronos", "chronos_bolt": M + "foundation.fm.ChronosBolt",
    "timesfm": M + "foundation.fm.TimesFM", "moirai": M + "foundation.fm.Moirai",
    "lag_llama": M + "foundation.fm.LagLlama", "moment": M + "foundation.fm.Moment",
    # reinforcement learning
    "dqn": M + "rl.agents.DQN", "double_dqn": M + "rl.agents.DoubleDQN", "dueling_dqn": M + "rl.agents.DuelingDQN",
    "ppo": M + "rl.agents.PPO", "a2c": M + "rl.agents.A2C", "sac": M + "rl.agents.SAC", "td3": M + "rl.agents.TD3",
    "ddpg": M + "rl.agents.DDPG",
    # meta-labeling and ensembles (built from other experiments' out-of-fold output)
    "meta": M + "meta.meta.MetaLabel",
    "voting": M + "ensemble.ensembles.Voting", "prob_average": M + "ensemble.ensembles.ProbAverage",
    "stacking": M + "ensemble.ensembles.Stacking",
}


def get(name: str):
    path = REGISTRY[name]
    mod, cls = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)
