# HPO Round rtrrl-hop-stream-v1 / 001

- Imported Aim runs: 11
- Best value: 111.9764175415039
- Selected candidates: 16 / requested 16
- Rejected near history: 0
- Rejected near batch: 0
- Rejected infeasible (Brax batch*mb%env): 0

## Candidates

1. `control/hpo/runs/rtrrl-hop-stream-v1/round_001/configs/config_001.yml` profile=`c7a-medium` d_hist=0.467 acq=2.502
   params: `{"entropy_rate": 0.001, "eta_f": 1.0, "eta_pi": 1.0, "hidden_size": 64, "lambda_pi": 0.95, "lambda_rnn": 0.99, "lambda_v": 0.95, "mlp_actor": true, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.001, "optimizer_params_td.learning_rate": 0.001, "pass_obs": true, "trace_mode": "accumulate"}`
2. `control/hpo/runs/rtrrl-hop-stream-v1/round_001/configs/config_002.yml` profile=`c7a-medium` d_hist=0.508 acq=2.502
   params: `{"entropy_rate": 0.0001, "eta_f": 0.0, "eta_pi": 0.3, "hidden_size": 32, "lambda_pi": 0.95, "lambda_rnn": 0.99, "lambda_v": 0.99, "mlp_actor": false, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.001, "optimizer_params_td.learning_rate": 0.001, "pass_obs": false, "trace_mode": "dutch"}`
3. `control/hpo/runs/rtrrl-hop-stream-v1/round_001/configs/config_003.yml` profile=`c7a-medium` d_hist=0.456 acq=2.489
   params: `{"entropy_rate": 1e-06, "eta_f": 0.0, "eta_pi": 0.3, "hidden_size": 128, "lambda_pi": 0.99, "lambda_rnn": 0.99, "lambda_v": 0.99, "mlp_actor": false, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.0003, "optimizer_params_td.learning_rate": 0.001, "pass_obs": false, "trace_mode": "dutch"}`
4. `control/hpo/runs/rtrrl-hop-stream-v1/round_001/configs/config_004.yml` profile=`c7a-medium` d_hist=0.354 acq=2.489
   params: `{"entropy_rate": 0.001, "eta_f": 0.3, "eta_pi": 0.3, "hidden_size": 128, "lambda_pi": 0.99, "lambda_rnn": 0.99, "lambda_v": 0.95, "mlp_actor": false, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.0003, "optimizer_params_td.learning_rate": 0.001, "pass_obs": false, "trace_mode": "accumulate"}`
5. `control/hpo/runs/rtrrl-hop-stream-v1/round_001/configs/config_005.yml` profile=`c7a-medium` d_hist=0.512 acq=2.488
   params: `{"entropy_rate": 0.001, "eta_f": 1.0, "eta_pi": 1.0, "hidden_size": 16, "lambda_pi": 0.99, "lambda_rnn": 0.99, "lambda_v": 0.9, "mlp_actor": true, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.003, "optimizer_params_td.learning_rate": 0.001, "pass_obs": false, "trace_mode": "accumulate"}`
6. `control/hpo/runs/rtrrl-hop-stream-v1/round_001/configs/config_006.yml` profile=`c7a-medium` d_hist=0.375 acq=2.488
   params: `{"entropy_rate": 0.0001, "eta_f": 0.3, "eta_pi": 1.0, "hidden_size": 32, "lambda_pi": 0.99, "lambda_rnn": 0.95, "lambda_v": 0.95, "mlp_actor": false, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.001, "optimizer_params_td.learning_rate": 0.003, "pass_obs": false, "trace_mode": "accumulate"}`
7. `control/hpo/runs/rtrrl-hop-stream-v1/round_001/configs/config_007.yml` profile=`c7a-medium` d_hist=0.512 acq=2.406
   params: `{"entropy_rate": 0.0001, "eta_f": 1.0, "eta_pi": 1.0, "hidden_size": 32, "lambda_pi": 0.99, "lambda_rnn": 0.9, "lambda_v": 0.9, "mlp_actor": false, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.0003, "optimizer_params_td.learning_rate": 0.001, "pass_obs": true, "trace_mode": "dutch"}`
8. `control/hpo/runs/rtrrl-hop-stream-v1/round_001/configs/config_008.yml` profile=`c7a-medium` d_hist=0.348 acq=2.354
   params: `{"entropy_rate": 0.0001, "eta_f": 1.0, "eta_pi": 3.0, "hidden_size": 64, "lambda_pi": 0.99, "lambda_rnn": 0.99, "lambda_v": 0.99, "mlp_actor": true, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.0003, "optimizer_params_td.learning_rate": 0.0003, "pass_obs": false, "trace_mode": "accumulate"}`
9. `control/hpo/runs/rtrrl-hop-stream-v1/round_001/configs/config_009.yml` profile=`c7a-medium` d_hist=0.348 acq=2.354
   params: `{"entropy_rate": 0.0001, "eta_f": 1.0, "eta_pi": 1.0, "hidden_size": 64, "lambda_pi": 0.99, "lambda_rnn": 0.99, "lambda_v": 0.95, "mlp_actor": false, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.0003, "optimizer_params_td.learning_rate": 0.003, "pass_obs": false, "trace_mode": "accumulate"}`
10. `control/hpo/runs/rtrrl-hop-stream-v1/round_001/configs/config_010.yml` profile=`c7a-medium` d_hist=0.362 acq=2.354
   params: `{"entropy_rate": 1e-05, "eta_f": 1.0, "eta_pi": 0.3, "hidden_size": 16, "lambda_pi": 0.95, "lambda_rnn": 0.99, "lambda_v": 0.99, "mlp_actor": false, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.0003, "optimizer_params_td.learning_rate": 0.003, "pass_obs": false, "trace_mode": "dutch"}`
11. `control/hpo/runs/rtrrl-hop-stream-v1/round_001/configs/config_011.yml` profile=`c7a-medium` d_hist=0.423 acq=2.353
   params: `{"entropy_rate": 1e-06, "eta_f": 0.3, "eta_pi": 1.0, "hidden_size": 128, "lambda_pi": 0.99, "lambda_rnn": 0.95, "lambda_v": 0.99, "mlp_actor": true, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.0001, "optimizer_params_td.learning_rate": 0.001, "pass_obs": false, "trace_mode": "accumulate"}`
12. `control/hpo/runs/rtrrl-hop-stream-v1/round_001/configs/config_012.yml` profile=`c7a-medium` d_hist=0.450 acq=2.281
   params: `{"entropy_rate": 0.001, "eta_f": 0.0, "eta_pi": 1.0, "hidden_size": 128, "lambda_pi": 0.95, "lambda_rnn": 0.99, "lambda_v": 0.95, "mlp_actor": false, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.001, "optimizer_params_td.learning_rate": 0.0001, "pass_obs": false, "trace_mode": "accumulate"}`
13. `control/hpo/runs/rtrrl-hop-stream-v1/round_001/configs/config_013.yml` profile=`c7a-medium` d_hist=0.490 acq=2.281
   params: `{"entropy_rate": 1e-06, "eta_f": 0.0, "eta_pi": 1.0, "hidden_size": 32, "lambda_pi": 0.99, "lambda_rnn": 0.99, "lambda_v": 0.95, "mlp_actor": true, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.001, "optimizer_params_td.learning_rate": 0.001, "pass_obs": false, "trace_mode": "accumulate"}`
14. `control/hpo/runs/rtrrl-hop-stream-v1/round_001/configs/config_014.yml` profile=`c7a-medium` d_hist=0.436 acq=2.248
   params: `{"entropy_rate": 0.0001, "eta_f": 0.0, "eta_pi": 1.0, "hidden_size": 128, "lambda_pi": 0.99, "lambda_rnn": 0.99, "lambda_v": 0.99, "mlp_actor": false, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.001, "optimizer_params_td.learning_rate": 0.003, "pass_obs": false, "trace_mode": "accumulate"}`
15. `control/hpo/runs/rtrrl-hop-stream-v1/round_001/configs/config_015.yml` profile=`c7a-medium` d_hist=0.448 acq=2.248
   params: `{"entropy_rate": 0.001, "eta_f": 1.0, "eta_pi": 1.0, "hidden_size": 32, "lambda_pi": 0.99, "lambda_rnn": 0.95, "lambda_v": 0.99, "mlp_actor": true, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.003, "optimizer_params_td.learning_rate": 0.001, "pass_obs": false, "trace_mode": "accumulate"}`
16. `control/hpo/runs/rtrrl-hop-stream-v1/round_001/configs/config_016.yml` profile=`c7a-medium` d_hist=0.450 acq=1.908
   params: `{"entropy_rate": 1e-06, "eta_f": 1.0, "eta_pi": 0.3, "hidden_size": 16, "lambda_pi": 0.9, "lambda_rnn": 0.99, "lambda_v": 0.95, "mlp_actor": false, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.001, "optimizer_params_td.learning_rate": 0.0003, "pass_obs": true, "trace_mode": "accumulate"}`
