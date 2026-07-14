# HPO Round rtrrl-hop-stream-v1 / 002

- Imported Aim runs: 12
- Best value: 111.9764175415039
- Selected candidates: 16 / requested 16
- Rejected near history: 0
- Rejected near batch: 0
- Rejected infeasible (Brax batch*mb%env): 0

## Candidates

1. `control/hpo/runs/rtrrl-hop-stream-v1/round_002/configs/config_001.yml` profile=`c7a-medium` d_hist=0.495 acq=2.539
   params: `{"entropy_rate": 0.001, "eta_f": 0.0, "eta_pi": 0.3, "lambda_pi": 0.95, "lambda_rnn": 0.99, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.001, "optimizer_params_td.learning_rate": 0.001, "pass_obs": true, "trace_mode": "accumulate"}`
2. `control/hpo/runs/rtrrl-hop-stream-v1/round_002/configs/config_002.yml` profile=`c7a-medium` d_hist=0.498 acq=2.530
   params: `{"entropy_rate": 1e-05, "eta_f": 0.0, "eta_pi": 0.3, "lambda_pi": 0.99, "lambda_rnn": 0.9, "lambda_v": 0.95, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.001, "optimizer_params_td.learning_rate": 0.001, "pass_obs": false, "trace_mode": "dutch"}`
3. `control/hpo/runs/rtrrl-hop-stream-v1/round_002/configs/config_003.yml` profile=`c7a-medium` d_hist=0.484 acq=2.523
   params: `{"entropy_rate": 0.001, "eta_f": 1.0, "eta_pi": 0.3, "lambda_pi": 0.99, "lambda_rnn": 0.9, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.003, "optimizer_params_td.learning_rate": 0.001, "pass_obs": false, "trace_mode": "dutch"}`
4. `control/hpo/runs/rtrrl-hop-stream-v1/round_002/configs/config_004.yml` profile=`c7a-medium` d_hist=0.330 acq=2.523
   params: `{"entropy_rate": 0.001, "eta_f": 1.0, "eta_pi": 0.3, "lambda_pi": 0.95, "lambda_rnn": 0.95, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.003, "optimizer_params_td.learning_rate": 0.001, "pass_obs": false, "trace_mode": "accumulate"}`
5. `control/hpo/runs/rtrrl-hop-stream-v1/round_002/configs/config_005.yml` profile=`c7a-medium` d_hist=0.441 acq=2.523
   params: `{"entropy_rate": 1e-05, "eta_f": 0.0, "eta_pi": 1.0, "lambda_pi": 0.95, "lambda_rnn": 0.99, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.0003, "optimizer_params_td.learning_rate": 0.001, "pass_obs": true, "trace_mode": "accumulate"}`
6. `control/hpo/runs/rtrrl-hop-stream-v1/round_002/configs/config_006.yml` profile=`c7a-medium` d_hist=0.466 acq=2.515
   params: `{"entropy_rate": 1e-05, "eta_f": 1.0, "eta_pi": 0.3, "lambda_pi": 0.99, "lambda_rnn": 0.95, "lambda_v": 0.9, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.0003, "optimizer_params_td.learning_rate": 0.001, "pass_obs": true, "trace_mode": "accumulate"}`
7. `control/hpo/runs/rtrrl-hop-stream-v1/round_002/configs/config_007.yml` profile=`c7a-medium` d_hist=0.379 acq=2.507
   params: `{"entropy_rate": 1e-06, "eta_f": 1.0, "eta_pi": 0.3, "lambda_pi": 0.9, "lambda_rnn": 0.95, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.001, "optimizer_params_td.learning_rate": 0.003, "pass_obs": false, "trace_mode": "accumulate"}`
8. `control/hpo/runs/rtrrl-hop-stream-v1/round_002/configs/config_008.yml` profile=`c7a-medium` d_hist=0.379 acq=2.443
   params: `{"entropy_rate": 0.001, "eta_f": 0.0, "eta_pi": 1.0, "lambda_pi": 0.99, "lambda_rnn": 0.99, "lambda_v": 0.95, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.001, "optimizer_params_td.learning_rate": 0.001, "pass_obs": true, "trace_mode": "accumulate"}`
9. `control/hpo/runs/rtrrl-hop-stream-v1/round_002/configs/config_009.yml` profile=`c7a-medium` d_hist=0.370 acq=2.428
   params: `{"entropy_rate": 1e-05, "eta_f": 0.0, "eta_pi": 3.0, "lambda_pi": 0.9, "lambda_rnn": 0.95, "lambda_v": 0.95, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.001, "optimizer_params_td.learning_rate": 0.001, "pass_obs": false, "trace_mode": "accumulate"}`
10. `control/hpo/runs/rtrrl-hop-stream-v1/round_002/configs/config_010.yml` profile=`c7a-medium` d_hist=0.400 acq=2.356
   params: `{"entropy_rate": 0.001, "eta_f": 0.0, "eta_pi": 0.3, "lambda_pi": 0.99, "lambda_rnn": 0.99, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.0003, "optimizer_params_td.learning_rate": 0.0003, "pass_obs": false, "trace_mode": "accumulate"}`
11. `control/hpo/runs/rtrrl-hop-stream-v1/round_002/configs/config_011.yml` profile=`c7a-medium` d_hist=0.449 acq=2.356
   params: `{"entropy_rate": 1e-05, "eta_f": 1.0, "eta_pi": 3.0, "lambda_pi": 0.99, "lambda_rnn": 0.99, "lambda_v": 0.95, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.0001, "optimizer_params_td.learning_rate": 0.001, "pass_obs": true, "trace_mode": "accumulate"}`
12. `control/hpo/runs/rtrrl-hop-stream-v1/round_002/configs/config_012.yml` profile=`c7a-medium` d_hist=0.471 acq=2.349
   params: `{"entropy_rate": 0.001, "eta_f": 1.0, "eta_pi": 1.0, "lambda_pi": 0.9, "lambda_rnn": 0.99, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.0003, "optimizer_params_td.learning_rate": 0.0003, "pass_obs": true, "trace_mode": "accumulate"}`
13. `control/hpo/runs/rtrrl-hop-stream-v1/round_002/configs/config_013.yml` profile=`c7a-medium` d_hist=0.394 acq=2.336
   params: `{"entropy_rate": 0.001, "eta_f": 1.0, "eta_pi": 1.0, "lambda_pi": 0.9, "lambda_rnn": 0.99, "lambda_v": 0.9, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.001, "optimizer_params_td.learning_rate": 0.003, "pass_obs": false, "trace_mode": "accumulate"}`
14. `control/hpo/runs/rtrrl-hop-stream-v1/round_002/configs/config_014.yml` profile=`c7a-medium` d_hist=0.414 acq=2.270
   params: `{"entropy_rate": 0.0001, "eta_f": 1.0, "eta_pi": 3.0, "lambda_pi": 0.95, "lambda_rnn": 0.9, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.0001, "optimizer_params_td.learning_rate": 0.001, "pass_obs": false, "trace_mode": "accumulate"}`
15. `control/hpo/runs/rtrrl-hop-stream-v1/round_002/configs/config_015.yml` profile=`c7a-medium` d_hist=0.300 acq=2.042
   params: `{"entropy_rate": 1e-05, "eta_f": 0.3, "eta_pi": 0.3, "lambda_pi": 0.99, "lambda_rnn": 0.99, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.003, "optimizer_params_td.learning_rate": 0.001, "pass_obs": false, "trace_mode": "accumulate"}`
16. `control/hpo/runs/rtrrl-hop-stream-v1/round_002/configs/config_016.yml` profile=`c7a-medium` d_hist=0.433 acq=2.039
   params: `{"entropy_rate": 1e-06, "eta_f": 0.3, "eta_pi": 1.0, "lambda_pi": 0.9, "lambda_rnn": 0.9, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.001, "optimizer_params_td.learning_rate": 0.001, "pass_obs": false, "trace_mode": "accumulate"}`
