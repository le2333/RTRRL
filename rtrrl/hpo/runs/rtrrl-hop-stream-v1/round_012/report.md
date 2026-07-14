# HPO Round rtrrl-hop-stream-v1 / 012

- Imported Aim runs: 101
- Best value: 244.43310546875
- Selected candidates: 32 / requested 32
- Rejected near history: 0
- Rejected near batch: 4
- Rejected infeasible (Brax batch*mb%env): 0

## Candidates

1. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_001.yml` profile=`c7a-medium` d_hist=0.223 acq=9.872
   params: `{"entropy_rate": 0.001, "eta_f": 0.0, "eta_pi": 3.0, "lambda_pi": 0.97, "lambda_rnn": 0.99, "lambda_v": 0.93, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 3e-05, "optimizer_params_td.learning_rate": 0.0003, "pass_obs": true, "trace_mode": "accumulate"}`
2. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_002.yml` profile=`c7a-medium` d_hist=0.317 acq=9.851
   params: `{"entropy_rate": 1e-06, "eta_f": 0.0, "eta_pi": 3.0, "lambda_pi": 0.97, "lambda_rnn": 0.99, "lambda_v": 0.93, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.003, "optimizer_params_td.learning_rate": 0.0003, "pass_obs": true, "trace_mode": "accumulate"}`
3. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_003.yml` profile=`c7a-medium` d_hist=0.146 acq=9.848
   params: `{"entropy_rate": 0.0003, "eta_f": 0.0, "eta_pi": 3.0, "lambda_pi": 0.97, "lambda_rnn": 0.99, "lambda_v": 0.93, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.003, "optimizer_params_td.learning_rate": 0.0003, "pass_obs": true, "trace_mode": "accumulate"}`
4. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_004.yml` profile=`c7a-medium` d_hist=0.304 acq=9.836
   params: `{"entropy_rate": 1e-05, "eta_f": 0.1, "eta_pi": 0.3, "lambda_pi": 0.97, "lambda_rnn": 0.93, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 3e-05, "optimizer_params_td.learning_rate": 0.003, "pass_obs": false, "trace_mode": "dutch"}`
5. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_005.yml` profile=`c7a-medium` d_hist=0.298 acq=9.833
   params: `{"entropy_rate": 3e-05, "eta_f": 0.0, "eta_pi": 3.0, "lambda_pi": 0.97, "lambda_rnn": 0.97, "lambda_v": 0.93, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 3e-05, "optimizer_params_td.learning_rate": 0.0003, "pass_obs": true, "trace_mode": "dutch"}`
6. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_006.yml` profile=`c7a-medium` d_hist=0.155 acq=9.821
   params: `{"entropy_rate": 1e-06, "eta_f": 0.1, "eta_pi": 0.5, "lambda_pi": 0.97, "lambda_rnn": 0.93, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 3e-05, "optimizer_params_td.learning_rate": 0.0003, "pass_obs": false, "trace_mode": "accumulate"}`
7. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_007.yml` profile=`c7a-medium` d_hist=0.092 acq=9.820
   params: `{"entropy_rate": 1e-06, "eta_f": 0.1, "eta_pi": 0.1, "lambda_pi": 0.97, "lambda_rnn": 0.93, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 3e-05, "optimizer_params_td.learning_rate": 0.001, "pass_obs": false, "trace_mode": "accumulate"}`
8. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_008.yml` profile=`c7a-medium` d_hist=0.323 acq=9.818
   params: `{"entropy_rate": 1e-06, "eta_f": 0.0, "eta_pi": 3.0, "lambda_pi": 0.97, "lambda_rnn": 0.99, "lambda_v": 0.93, "normalize_obs": true, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 3e-05, "optimizer_params_td.learning_rate": 0.0003, "pass_obs": true, "trace_mode": "accumulate"}`
9. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_009.yml` profile=`c7a-medium` d_hist=0.351 acq=9.812
   params: `{"entropy_rate": 3e-05, "eta_f": 0.0, "eta_pi": 3.0, "lambda_pi": 0.97, "lambda_rnn": 0.99, "lambda_v": 0.93, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.003, "optimizer_params_td.learning_rate": 0.0003, "pass_obs": true, "trace_mode": "dutch"}`
10. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_010.yml` profile=`c7a-medium` d_hist=0.211 acq=9.811
   params: `{"entropy_rate": 1e-06, "eta_f": 0.0, "eta_pi": 1.0, "lambda_pi": 0.93, "lambda_rnn": 0.97, "lambda_v": 0.95, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 3e-05, "optimizer_params_td.learning_rate": 3e-05, "pass_obs": false, "trace_mode": "dutch"}`
11. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_011.yml` profile=`c7a-medium` d_hist=0.102 acq=9.806
   params: `{"entropy_rate": 3e-05, "eta_f": 0.0, "eta_pi": 3.0, "lambda_pi": 0.97, "lambda_rnn": 0.97, "lambda_v": 0.93, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.0001, "optimizer_params_td.learning_rate": 0.0003, "pass_obs": true, "trace_mode": "accumulate"}`
12. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_012.yml` profile=`c7a-medium` d_hist=0.368 acq=9.792
   params: `{"entropy_rate": 0.001, "eta_f": 0.1, "eta_pi": 0.3, "lambda_pi": 0.97, "lambda_rnn": 0.93, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 3e-05, "optimizer_params_td.learning_rate": 0.003, "pass_obs": false, "trace_mode": "dutch"}`
13. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_013.yml` profile=`c7a-medium` d_hist=0.236 acq=9.785
   params: `{"entropy_rate": 3e-06, "eta_f": 0.1, "eta_pi": 3.0, "lambda_pi": 0.97, "lambda_rnn": 0.93, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 3e-05, "optimizer_params_td.learning_rate": 0.003, "pass_obs": false, "trace_mode": "accumulate"}`
14. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_014.yml` profile=`c7a-medium` d_hist=0.155 acq=9.776
   params: `{"entropy_rate": 3e-05, "eta_f": 0.0, "eta_pi": 2.0, "lambda_pi": 0.97, "lambda_rnn": 0.99, "lambda_v": 0.93, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.0003, "optimizer_params_td.learning_rate": 0.0003, "pass_obs": true, "trace_mode": "accumulate"}`
15. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_015.yml` profile=`c7a-medium` d_hist=0.048 acq=9.532
   params: `{"entropy_rate": 0.0001, "eta_f": 0.0, "eta_pi": 3.0, "lambda_pi": 0.97, "lambda_rnn": 0.99, "lambda_v": 0.93, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 3e-05, "optimizer_params_td.learning_rate": 0.0003, "pass_obs": true, "trace_mode": "accumulate"}`
16. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_016.yml` profile=`c7a-medium` d_hist=0.144 acq=9.522
   params: `{"entropy_rate": 0.001, "eta_f": 0.0, "eta_pi": 3.0, "lambda_pi": 0.97, "lambda_rnn": 0.99, "lambda_v": 0.93, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 3e-05, "optimizer_params_td.learning_rate": 0.0003, "pass_obs": true, "trace_mode": "accumulate"}`
17. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_017.yml` profile=`c7a-medium` d_hist=0.144 acq=9.451
   params: `{"entropy_rate": 1e-06, "eta_f": 0.0, "eta_pi": 1.0, "lambda_pi": 0.97, "lambda_rnn": 0.93, "lambda_v": 0.95, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.0003, "optimizer_params_td.learning_rate": 3e-05, "pass_obs": false, "trace_mode": "accumulate"}`
18. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_018.yml` profile=`c7a-medium` d_hist=0.199 acq=9.392
   params: `{"entropy_rate": 3e-05, "eta_f": 0.0, "eta_pi": 3.0, "lambda_pi": 0.97, "lambda_rnn": 0.99, "lambda_v": 0.93, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.003, "optimizer_params_td.learning_rate": 0.0003, "pass_obs": true, "trace_mode": "accumulate"}`
19. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_019.yml` profile=`c7a-medium` d_hist=0.211 acq=9.370
   params: `{"entropy_rate": 1e-06, "eta_f": 0.0, "eta_pi": 1.0, "lambda_pi": 0.97, "lambda_rnn": 0.93, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.0003, "optimizer_params_td.learning_rate": 3e-05, "pass_obs": false, "trace_mode": "dutch"}`
20. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_020.yml` profile=`c7a-medium` d_hist=0.096 acq=9.336
   params: `{"entropy_rate": 1e-05, "eta_f": 0.1, "eta_pi": 0.3, "lambda_pi": 0.97, "lambda_rnn": 0.93, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 3e-05, "optimizer_params_td.learning_rate": 0.003, "pass_obs": false, "trace_mode": "accumulate"}`
21. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_021.yml` profile=`c7a-medium` d_hist=0.072 acq=9.254
   params: `{"entropy_rate": 1e-06, "eta_f": 0.1, "eta_pi": 0.3, "lambda_pi": 0.97, "lambda_rnn": 0.9, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 3e-05, "optimizer_params_td.learning_rate": 0.003, "pass_obs": false, "trace_mode": "accumulate"}`
22. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_022.yml` profile=`c7a-medium` d_hist=0.154 acq=9.199
   params: `{"entropy_rate": 1e-06, "eta_f": 0.0, "eta_pi": 1.0, "lambda_pi": 0.97, "lambda_rnn": 0.93, "lambda_v": 0.9, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.0003, "optimizer_params_td.learning_rate": 3e-05, "pass_obs": false, "trace_mode": "accumulate"}`
23. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_023.yml` profile=`c7a-medium` d_hist=0.144 acq=9.071
   params: `{"entropy_rate": 1e-06, "eta_f": 0.0, "eta_pi": 1.0, "lambda_pi": 0.93, "lambda_rnn": 0.93, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.0003, "optimizer_params_td.learning_rate": 3e-05, "pass_obs": false, "trace_mode": "accumulate"}`
24. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_024.yml` profile=`c7a-medium` d_hist=0.211 acq=9.035
   params: `{"entropy_rate": 1e-06, "eta_f": 0.0, "eta_pi": 1.0, "lambda_pi": 0.97, "lambda_rnn": 0.97, "lambda_v": 0.95, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 3e-05, "optimizer_params_td.learning_rate": 0.0003, "pass_obs": true, "trace_mode": "accumulate"}`
25. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_025.yml` profile=`c7a-medium` d_hist=0.170 acq=8.843
   params: `{"entropy_rate": 1e-06, "eta_f": 0.0, "eta_pi": 0.3, "lambda_pi": 0.97, "lambda_rnn": 0.93, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.0003, "optimizer_params_td.learning_rate": 0.001, "pass_obs": false, "trace_mode": "accumulate"}`
26. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_026.yml` profile=`c7a-medium` d_hist=0.112 acq=8.698
   params: `{"entropy_rate": 1e-05, "eta_f": 0.0, "eta_pi": 2.0, "lambda_pi": 0.97, "lambda_rnn": 0.93, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.0003, "optimizer_params_td.learning_rate": 3e-05, "pass_obs": false, "trace_mode": "accumulate"}`
27. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_027.yml` profile=`c7a-medium` d_hist=0.072 acq=8.487
   params: `{"entropy_rate": 1e-06, "eta_f": 0.0, "eta_pi": 0.3, "lambda_pi": 0.99, "lambda_rnn": 0.93, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 3e-05, "optimizer_params_td.learning_rate": 3e-05, "pass_obs": false, "trace_mode": "accumulate"}`
28. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_028.yml` profile=`c7a-medium` d_hist=0.298 acq=8.355
   params: `{"entropy_rate": 1e-06, "eta_f": 0.3, "eta_pi": 0.3, "lambda_pi": 0.97, "lambda_rnn": 0.93, "lambda_v": 0.93, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.001, "optimizer_params_td.learning_rate": 3e-05, "pass_obs": false, "trace_mode": "dutch"}`
29. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_029.yml` profile=`c7a-medium` d_hist=0.285 acq=8.159
   params: `{"entropy_rate": 1e-06, "eta_f": 0.5, "eta_pi": 0.3, "lambda_pi": 0.97, "lambda_rnn": 0.93, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.003, "optimizer_params_td.learning_rate": 3e-05, "pass_obs": false, "trace_mode": "accumulate"}`
30. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_030.yml` profile=`c7a-medium` d_hist=0.298 acq=7.783
   params: `{"entropy_rate": 3e-05, "eta_f": 0.0, "eta_pi": 3.0, "lambda_pi": 0.99, "lambda_rnn": 0.99, "lambda_v": 0.93, "normalize_obs": true, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 3e-05, "optimizer_params_td.learning_rate": 0.0003, "pass_obs": true, "trace_mode": "accumulate"}`
31. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_031.yml` profile=`c7a-medium` d_hist=0.144 acq=7.560
   params: `{"entropy_rate": 1e-06, "eta_f": 0.0, "eta_pi": 1.0, "lambda_pi": 0.97, "lambda_rnn": 0.97, "lambda_v": 0.95, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 3e-05, "optimizer_params_td.learning_rate": 0.0003, "pass_obs": false, "trace_mode": "accumulate"}`
32. `control/hpo/runs/rtrrl-hop-stream-v1/round_012/configs/config_032.yml` profile=`c7a-medium` d_hist=0.228 acq=7.487
   params: `{"entropy_rate": 3e-05, "eta_f": 0.3, "eta_pi": 0.3, "lambda_pi": 0.97, "lambda_rnn": 0.97, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 3e-05, "optimizer_params_td.learning_rate": 0.001, "pass_obs": false, "trace_mode": "accumulate"}`
