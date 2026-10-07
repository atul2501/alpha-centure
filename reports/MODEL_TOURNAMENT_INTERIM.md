# SOL 15m Model Tournament — INTERIM report

_Generated 2026-10-06 17:07 UTC. 32 of ~150 planned experiments finished; the rest are still running. Final report: `reports/MODEL_TOURNAMENT_REPORT.md` when the pipeline completes._

## Bottom line so far

- **No model passes the pre-registered gates.** Every finished model fails at least one gate (passes all: 0 of 32).
- Overfitting / data-mining checks across all finished experiments: PBO = **0.62** (gate ≤ 0.2), Hansen SPA p = **0.32** (gate < 0.05). The best results are consistent with luck among 32 tries.
- Benchmark: buy-and-hold SOL made $69,775 on $30k over the same window (2021-07 → 2024-07). Most profitable models make their money on **long** trades, i.e. they ride the bull market rather than time it.
- Costs: ~14 bps per round trip (5 bps taker fee each side + spread + impact + latency). That is what kills most 15m signals.

How to read: DEV-OOS = 12 quarterly walk-forward test folds, never used for fitting or tuning. Net is after fees, slippage and funding, on $30k at 1× notional. t = net per-trade t-stat (≥ 2 needed). DSR = deflated Sharpe, adjusted for the number of models tried (≥ 0.95 needed).

## All finished models (ranked by net P&L)

| # | Model | Family | Net | PF | Win | Exp bps | Sharpe | Max DD | Trades | Fees | Slippage | Funding | Long | Short | Net @2× cost | t | DSR | Gates failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | random_forest | tree | $76,948 | 1.19 | 55% | 25.8 | 1.03 | 110% | 994 | $29,820 | $11,811 | $-7,407 | $86,977 | $-10,029 | $35,317 | 1.84 | 0.34 | 5 |
| 2 | buy_hold | rules | $69,775 | 3.45 | 50% | 5814.6 | 0.60 | 95% | 4 | $120 | $47 | $-4,346 | $69,775 | – | $69,608 | 0.92 | 0.29 | 9 |
| 3 | decision_tree | tree | $59,003 | 1.11 | 59% | 10.0 | 0.76 | 103% | 1969 | $59,070 | $23,776 | $-2,907 | $71,052 | $-12,049 | $-23,843 | 1.30 | 0.22 | 6 |
| 4 | linear_svm | linear | $48,477 | 1.11 | 55% | 13.2 | 0.64 | 83% | 1220 | $36,600 | $14,364 | $-5,793 | $90,166 | $-41,690 | $-2,487 | 1.11 | 0.14 | 6 |
| 5 | xgboost | tree | $43,458 | 1.10 | 47% | 10.3 | 0.61 | 170% | 1407 | $42,210 | $16,611 | $-7,021 | $65,059 | $-21,601 | $-15,364 | 1.08 | 0.13 | 8 |
| 6 | lightgbm | tree | $38,926 | 1.07 | 54% | 7.1 | 0.53 | 193% | 1826 | $54,780 | $21,620 | $-5,600 | $52,654 | $-13,728 | $-37,474 | 0.93 | 0.12 | 9 |
| 7 | logistic | linear | $37,280 | 1.09 | 55% | 9.1 | 0.52 | 77% | 1373 | $41,190 | $16,468 | $-4,951 | $20,803 | $16,478 | $-20,377 | 0.87 | 0.10 | 6 |
| 8 | catboost | tree | $29,640 | 1.10 | 48% | 10.2 | 0.53 | 112% | 969 | $29,070 | $11,515 | $-8,803 | $36,235 | $-6,595 | $-10,945 | 0.89 | 0.08 | 8 |
| 9 | elasticnet | linear | $29,464 | 1.06 | 49% | 6.3 | 0.36 | 234% | 1567 | $47,010 | $19,494 | $-2,723 | $11,354 | $18,111 | $-37,040 | 0.62 | 0.08 | 10 |
| 10 | lasso | linear | $23,874 | 1.04 | 45% | 4.3 | 0.29 | 165% | 1830 | $54,900 | $21,688 | $-3,575 | $34,120 | $-10,246 | $-52,714 | 0.52 | 0.05 | 10 |
| 11 | extra_trees | tree | $23,237 | 1.05 | 46% | 5.5 | 0.30 | 126% | 1420 | $42,600 | $16,725 | $-6,890 | $57,743 | $-34,506 | $-36,088 | 0.48 | 0.05 | 10 |
| 12 | hist_gb | tree | $13,772 | 1.02 | 54% | 3.3 | 0.17 | 174% | 1408 | $42,240 | $16,714 | $-3,741 | $14,388 | $-616 | $-45,182 | 0.30 | 0.04 | 10 |
| 13 | state_space | classical | $7,453 | 1.83 | 15% | 92.0 | 0.66 | 18% | 27 | $810 | $324 | $2,592 | $12,329 | $-4,876 | $6,320 | 0.62 | 0.05 | 8 |
| 14 | svr | linear | $6,666 | 1.01 | 54% | 1.1 | 0.09 | 141% | 2020 | $60,600 | $24,366 | $-9,928 | $4,606 | $2,061 | $-78,300 | 0.16 | 0.03 | 12 |
| 15 | random | rules | $5,185 | 1.06 | 52% | 5.1 | 0.21 | 56% | 337 | $10,110 | $4,406 | $23 | $7,184 | $-2,000 | $-9,331 | 0.35 | 0.04 | 11 |
| 16 | ma | classical | $2,396 | 1.01 | 49% | 0.6 | 0.03 | 181% | 1241 | $37,230 | $15,036 | $16,024 | $14,290 | $-11,894 | $-49,870 | 0.06 | 0.02 | 12 |
| 17 | momentum | rules | $411 | 1.00 | 31% | 0.2 | 0.01 | 93% | 677 | $20,310 | $8,224 | $2,679 | $28,199 | $-27,788 | $-28,123 | 0.01 | 0.02 | 12 |
| 18 | flat | rules | $0 | – | –% | – | – | 0% | 0 | $0 | $0 | $0 | – | – | $0 | – | – | 16 |
| 19 | egarch | classical | $-292 | 1.00 | 55% | -0.1 | -0.00 | 185% | 939 | $28,170 | $11,767 | $908 | $5,676 | $-5,968 | $-40,229 | -0.01 | 0.02 | 16 |
| 20 | gradient_boosting | tree | $-1,842 | 1.00 | 45% | -0.2 | -0.02 | 208% | 3045 | $91,350 | $36,633 | $186 | $18,542 | $-20,384 | $-129,825 | -0.04 | 0.02 | 14 |
| 21 | ridge | linear | $-5,366 | 0.99 | 45% | -1.2 | -0.06 | 200% | 1540 | $46,200 | $18,643 | $-1,626 | $-2,821 | $-2,545 | $-70,209 | -0.11 | 0.01 | 15 |
| 22 | garch | classical | $-11,419 | 0.97 | 54% | -3.6 | -0.18 | 249% | 1056 | $31,680 | $13,146 | $593 | $8,671 | $-20,090 | $-56,245 | -0.33 | 0.01 | 16 |
| 23 | breakout | rules | $-11,486 | 0.97 | 30% | -3.8 | -0.17 | 101% | 1007 | $30,210 | $12,014 | $1,234 | $41,700 | $-53,185 | $-53,710 | -0.29 | 0.01 | 16 |
| 24 | linear_regression | linear | $-12,138 | 0.98 | 46% | -2.8 | -0.14 | 186% | 1466 | $43,980 | $17,865 | $-2,360 | $-9,983 | $-2,155 | $-73,983 | -0.27 | 0.01 | 16 |
| 25 | reversal | rules | $-12,977 | 0.97 | 55% | -2.9 | -0.23 | 113% | 1471 | $44,130 | $18,366 | $-2,755 | $18,090 | $-31,068 | $-75,474 | -0.37 | 0.01 | 15 |
| 26 | flow_follow | rules | $-14,006 | 0.97 | 42% | -3.2 | -0.24 | 181% | 1454 | $43,620 | $17,729 | $-7,076 | $710 | $-14,716 | $-75,355 | -0.39 | 0.01 | 16 |
| 27 | kalman | classical | $-25,729 | 0.95 | 31% | -6.7 | -0.32 | 129% | 1289 | $38,670 | $15,245 | $-2,197 | $40,173 | $-65,902 | $-79,644 | -0.50 | 0.00 | 15 |
| 28 | ar | classical | $-46,954 | 0.89 | 53% | -14.8 | -0.59 | 294% | 1054 | $31,620 | $12,815 | $16,804 | $-23,675 | $-23,279 | $-91,389 | -1.00 | 0.00 | 15 |
| 29 | arma | classical | $-54,781 | 0.89 | 52% | -12.8 | -0.76 | 343% | 1430 | $42,900 | $17,309 | $14,181 | $-3,577 | $-51,205 | $-114,990 | -1.28 | 0.00 | 15 |
| 30 | gjr_garch | classical | $-56,083 | 0.81 | 53% | -23.2 | -1.09 | 263% | 807 | $24,210 | $10,060 | $665 | $16,161 | $-72,245 | $-90,353 | -1.87 | 0.00 | 15 |
| 31 | arima | classical | $-58,549 | 0.88 | 55% | -15.2 | -0.77 | 347% | 1284 | $38,520 | $15,832 | $14,070 | $-14,323 | $-44,226 | $-112,901 | -1.34 | 0.00 | 15 |
| 32 | var | classical | $-79,743 | 0.86 | 50% | -13.5 | -0.97 | 304% | 1972 | $59,160 | $24,328 | $11,915 | $-15,229 | $-64,515 | $-163,231 | -1.83 | 0.00 | 16 |

## Why models fail (how many models fail each gate)

| Gate | Models failing |
|---|---|
| t >= 2 | 32 of 32 |
| DSR >= 0.95 (N=32) | 32 of 32 |
| PBO <= 0.2 | 32 of 32 |
| SPA p < 0.05 | 32 of 32 |
| max DD <= 30% | 30 of 32 |
| net > 0 at cost x2 | 29 of 32 |
| Calmar >= 0.5 | 27 of 32 |
| gross >= 1.5x cost | 24 of 32 |
| >= 60% folds positive | 23 of 32 |
| net > 0 at cost x1.5 | 23 of 32 |
| >= 60% regime buckets positive | 22 of 32 |
| net > 0 at slippage x2 | 19 of 32 |
| net > 0 | 15 of 32 |
| expectancy > 0 | 15 of 32 |
| PF > 1 | 15 of 32 |
| no month > 25% of profit | 10 of 32 |
| trades >= 300 | 3 of 32 |

Two gates (shuffled-label control, parameter stability) are only computed for the top candidates in the final analysis and are not counted here.

## By family

| Family | Models | Best | Median | Best t |
|---|---|---|---|---|
| classical | 10 | $7,453 (state_space) | $-36,341 | 0.62 |
| linear | 7 | $48,477 (linear_svm) | $23,874 | 1.11 |
| rules | 7 | $69,775 (buy_hold) | $0 | 0.92 |
| tree | 8 | $76,948 (random_forest) | $34,283 | 1.84 |

## Still to run (132 entries)

| Stage | Entry | Model | Plan |
|---|---|---|---|
| stage_3 | gaussian_hmm | gaussian_hmm | queued |
| stage_3 | hmm | hmm | queued |
| stage_3 | multivariate_hmm | multivariate_hmm | queued |
| stage_3 | markov_switching | markov_switching | queued |
| stage_3 | hsmm | hsmm | queued |
| stage_3 | explicit_duration_hmm | explicit_duration_hmm | queued |
| stage_3 | slds | slds | queued |
| stage_3 | switching_kalman | switching_kalman | queued |
| stage_3 | switching_state_space | switching_state_space | queued |
| stage_3 | hmm_lightgbm | hybrid | queued |
| stage_3 | hmm_xgboost | hybrid | queued |
| stage_3 | hmm_catboost | hybrid | queued |
| stage_3 | hmm_random_forest | hybrid | queued |
| stage_3 | hsmm_lightgbm | hybrid | queued |
| stage_3 | hsmm_xgboost | hybrid | queued |
| stage_3 | hsmm_catboost | hybrid | queued |
| stage_3 | hsmm_random_forest | hybrid | queued |
| stage_3 | kalman_lightgbm | hybrid | queued |
| stage_3 | kalman_xgboost | hybrid | queued |
| stage_3 | slds_lightgbm | hybrid | queued |
| stage_3 | ms_lightgbm | hybrid | queued |
| stage_3 | abl_lgb_price | lightgbm | queued |
| stage_3 | abl_lgb_price_volume | lightgbm | queued |
| stage_3 | abl_lgb_price_volume_flow | lightgbm | queued |
| stage_3 | abl_lgb_technical | lightgbm | queued |
| stage_3 | abl_lgb_core | lightgbm | queued |
| stage_3 | abl_lgb_core_deriv | lightgbm | queued |
| stage_3 | abl_ridge_price | ridge | queued |
| stage_3 | abl_ridge_price_volume | ridge | queued |
| stage_3 | abl_ridge_price_volume_flow | ridge | queued |
| stage_3 | abl_ridge_technical | ridge | queued |
| stage_3 | abl_ridge_core | ridge | queued |
| stage_3 | abl_ridge_core_deriv | ridge | queued |
| stage_3 | loo_lgb_no_price | lightgbm | queued |
| stage_3 | loo_lgb_no_flow | lightgbm | queued |
| stage_3 | loo_lgb_no_context | lightgbm | queued |
| stage_3 | loo_lgb_no_funding | lightgbm | queued |
| stage_3 | loo_lgb_no_trend_mom | lightgbm | queued |
| stage_3 | ob_without_lgb | lightgbm | queued |
| stage_3 | ob_with_lgb | lightgbm | queued |
| stage_3 | ob_without_ridge | ridge | queued |
| stage_3 | ob_with_ridge | ridge | queued |
| stage_3 | ob_only_lgb | lightgbm | queued |
| stage_3 | ob_only_ridge | ridge | queued |
| stage_4 | mlp | mlp | spot check |
| stage_4 | lstm | lstm | spot check |
| stage_4 | tcn | tcn | spot check |
| stage_4 | patchtst | patchtst | spot check |
| stage_4 | dlinear | dlinear | spot check |
| stage_4 | mamba | mamba | spot check |
| stage_4 | dnn | dnn | full sweep only if gate G1 opens |
| stage_4 | cnn1d | cnn1d | full sweep only if gate G1 opens |
| stage_4 | rnn | rnn | full sweep only if gate G1 opens |
| stage_4 | gru | gru | full sweep only if gate G1 opens |
| stage_4 | bilstm | bilstm | full sweep only if gate G1 opens |
| stage_4 | bigru | bigru | full sweep only if gate G1 opens |
| stage_4 | lstm_sweep | lstm | full sweep only if gate G1 opens |
| stage_4 | tcn_sweep | tcn | full sweep only if gate G1 opens |
| stage_4 | transformer | transformer | full sweep only if gate G1 opens |
| stage_4 | temporal_transformer | temporal_transformer | full sweep only if gate G1 opens |
| stage_4 | nlinear | nlinear | full sweep only if gate G1 opens |
| stage_4 | itransformer | itransformer | full sweep only if gate G1 opens |
| stage_4 | tft | tft | full sweep only if gate G1 opens |
| stage_4 | informer | informer | full sweep only if gate G1 opens |
| stage_4 | autoformer | autoformer | full sweep only if gate G1 opens |
| stage_4 | fedformer | fedformer | full sweep only if gate G1 opens |
| stage_4 | timesnet | timesnet | full sweep only if gate G1 opens |
| stage_4 | crossformer | crossformer | full sweep only if gate G1 opens |
| stage_4 | patchtst_sweep | patchtst | full sweep only if gate G1 opens |
| stage_4 | s4 | s4 | full sweep only if gate G1 opens |
| stage_4 | s5 | s5 | full sweep only if gate G1 opens |
| stage_4 | mamba2 | mamba2 | full sweep only if gate G1 opens |
| stage_4 | mamba_attention | mamba_attention | full sweep only if gate G1 opens |
| stage_4 | mamba_cnn | mamba_cnn | full sweep only if gate G1 opens |
| stage_4 | mamba_transformer | mamba_transformer | full sweep only if gate G1 opens |
| stage_4 | mamba_sweep | mamba | full sweep only if gate G1 opens |
| stage_4 | hsmm_lstm | hybrid | full sweep only if gate G1 opens |
| stage_4 | hsmm_gru | hybrid | full sweep only if gate G1 opens |
| stage_4 | hsmm_transformer | hybrid | full sweep only if gate G1 opens |
| stage_4 | hsmm_mamba | hybrid | full sweep only if gate G1 opens |
| stage_5 | chronos_bolt | chronos_bolt | spot check |
| stage_5 | ppo | ppo | spot check |
| stage_5 | dqn | dqn | spot check |
| stage_5 | moirai | moirai | queued |
| stage_5 | lag_llama | lag_llama | queued |
| stage_5 | moment | moment | queued |
| stage_5 | chronos | chronos | full sweep only if gate G1 opens |
| stage_5 | timesfm | timesfm | full sweep only if gate G1 opens |
| stage_5 | double_dqn | double_dqn | full sweep only if gate G1 opens |
| stage_5 | dueling_dqn | dueling_dqn | full sweep only if gate G1 opens |
| stage_5 | a2c | a2c | full sweep only if gate G1 opens |
| stage_5 | sac | sac | full sweep only if gate G1 opens |
| stage_5 | td3 | td3 | full sweep only if gate G1 opens |
| stage_5 | ddpg | ddpg | full sweep only if gate G1 opens |
| stage_6 | voting_families | voting | queued |
| stage_6 | voting_simple | voting | queued |
| stage_6 | prob_average_families | prob_average | queued |
| stage_6 | prob_average_simple | prob_average | queued |
| stage_6 | stacking_families | stacking | queued |
| stage_6 | stacking_simple | stacking | queued |
| stage_6 | meta_momentum_logistic | meta | queued |
| stage_6 | meta_momentum_lightgbm | meta | queued |
| stage_6 | meta_momentum_xgboost | meta | queued |
| stage_6 | meta_momentum_catboost | meta | queued |
| stage_6 | meta_momentum_random_forest | meta | queued |
| stage_6 | meta_momentum_mlp | meta | queued |
| stage_6 | meta_breakout_logistic | meta | queued |
| stage_6 | meta_breakout_lightgbm | meta | queued |
| stage_6 | meta_breakout_xgboost | meta | queued |
| stage_6 | meta_breakout_catboost | meta | queued |
| stage_6 | meta_breakout_random_forest | meta | queued |
| stage_6 | meta_breakout_mlp | meta | queued |
| stage_6 | meta_lightgbm_logistic | meta | queued |
| stage_6 | meta_lightgbm_lightgbm | meta | queued |
| stage_6 | meta_lightgbm_xgboost | meta | queued |
| stage_6 | meta_lightgbm_catboost | meta | queued |
| stage_6 | meta_lightgbm_random_forest | meta | queued |
| stage_6 | meta_lightgbm_mlp | meta | queued |
| stage_6 | meta_hsmm_lightgbm_logistic | meta | queued |
| stage_6 | meta_hsmm_lightgbm_lightgbm | meta | queued |
| stage_6 | meta_hsmm_lightgbm_xgboost | meta | queued |
| stage_6 | meta_hsmm_lightgbm_catboost | meta | queued |
| stage_6 | meta_hsmm_lightgbm_random_forest | meta | queued |
| stage_6 | meta_hsmm_lightgbm_mlp | meta | queued |
| stage_6 | mamba_lightgbm | hybrid | full sweep only if gate G1 opens |
| stage_6 | mamba_xgboost | hybrid | full sweep only if gate G1 opens |
| stage_6 | mamba_catboost | hybrid | full sweep only if gate G1 opens |
| stage_6 | transformer_lightgbm | hybrid | full sweep only if gate G1 opens |
| stage_6 | transformer_xgboost | hybrid | full sweep only if gate G1 opens |
| stage_6 | tcn_lightgbm | hybrid | full sweep only if gate G1 opens |
| stage_6 | lstm_lightgbm | hybrid | full sweep only if gate G1 opens |
| stage_6 | gru_lightgbm | hybrid | full sweep only if gate G1 opens |

## Compared with your P6 book

P6 (23-coin ridge momentum, rebalanced every few days) is not part of this tournament, which is SOL only at 15 minutes. Its record: DEV-OOS Sharpe 1.64, VALID-A +44.5% (pass), VALID-B −3.4% (fail), profitable on unseen coins. Nothing finished here has comparable evidence: the best SOL 15m models have t < 2 and DSR well below 0.95.
