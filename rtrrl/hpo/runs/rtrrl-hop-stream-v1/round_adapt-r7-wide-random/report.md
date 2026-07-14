# Manual Wide R7

- Strategy: stratified random categorical wide
- Seed: 77
- Items: 32

1. `config/rtrrl_hop_374.yml` `RTRRL-HOP-374`
   params: `{"entropy_rate": 3e-07, "env_params.batch_size": 8, "eta_f": 2.0, "eta_pi": 0.03, "gamma": 0.995, "lambda_pi": 0.99, "lambda_rnn": 0.955, "lambda_v": 0.88, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 3e-07, "optimizer_params_td.learning_rate": 0.0003, "pass_obs": true, "trace_mode": "dutch", "update_period": 0.1}`
2. `config/rtrrl_hop_375.yml` `RTRRL-HOP-375`
   params: `{"entropy_rate": 0.0015, "env_params.batch_size": 2, "eta_f": 0.2, "eta_pi": 10.0, "gamma": 0.97, "lambda_pi": 0.97, "lambda_rnn": 0.7, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 1e-07, "optimizer_params_td.learning_rate": 1e-05, "pass_obs": true, "trace_mode": "accumulate", "update_period": 0.01}`
3. `config/rtrrl_hop_376.yml` `RTRRL-HOP-376`
   params: `{"entropy_rate": 0.001, "env_params.batch_size": 4, "eta_f": 0.08, "eta_pi": 2.5, "gamma": 0.8, "lambda_pi": 0.5, "lambda_rnn": 0.5, "lambda_v": 0.93, "normalize_obs": true, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.003, "optimizer_params_td.learning_rate": 0.01, "pass_obs": false, "trace_mode": "accumulate", "update_period": 0.5}`
4. `config/rtrrl_hop_377.yml` `RTRRL-HOP-377`
   params: `{"entropy_rate": 0.0, "env_params.batch_size": 8, "eta_f": 0.45, "eta_pi": 5.0, "gamma": 0.95, "lambda_pi": 0.7, "lambda_rnn": 0.85, "lambda_v": 0.99, "normalize_obs": true, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.01, "optimizer_params_td.learning_rate": 5e-05, "pass_obs": true, "trace_mode": "accumulate", "update_period": 0.11}`
5. `config/rtrrl_hop_378.yml` `RTRRL-HOP-378`
   params: `{"entropy_rate": 3e-07, "env_params.batch_size": 18, "eta_f": 0.65, "eta_pi": 2.0, "gamma": 0.93, "lambda_pi": 0.99, "lambda_rnn": 0.5, "lambda_v": 0.5, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.0001, "optimizer_params_td.learning_rate": 1e-06, "pass_obs": false, "trace_mode": "dutch", "update_period": 0.7}`
6. `config/rtrrl_hop_379.yml` `RTRRL-HOP-379`
   params: `{"entropy_rate": 3e-05, "env_params.batch_size": 12, "eta_f": 0.5, "eta_pi": 10.0, "gamma": 0.9, "lambda_pi": 0.85, "lambda_rnn": 0.85, "lambda_v": 0.7, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.003, "optimizer_params_td.learning_rate": 3e-06, "pass_obs": true, "trace_mode": "accumulate", "update_period": 1.0}`
7. `config/rtrrl_hop_380.yml` `RTRRL-HOP-380`
   params: `{"entropy_rate": 1e-05, "env_params.batch_size": 32, "eta_f": 2.0, "eta_pi": 0.03, "gamma": 0.95, "lambda_pi": 0.96, "lambda_rnn": 0.93, "lambda_v": 0.97, "normalize_obs": true, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 1e-05, "optimizer_params_td.learning_rate": 0.0001, "pass_obs": true, "trace_mode": "dutch", "update_period": 0.3}`
8. `config/rtrrl_hop_381.yml` `RTRRL-HOP-381`
   params: `{"entropy_rate": 1e-07, "env_params.batch_size": 1, "eta_f": 0.12, "eta_pi": 5.0, "gamma": 0.99, "lambda_pi": 0.7, "lambda_rnn": 0.97, "lambda_v": 0.5, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 1e-06, "optimizer_params_td.learning_rate": 0.003, "pass_obs": true, "trace_mode": "accumulate", "update_period": 0.11}`
9. `config/rtrrl_hop_382.yml` `RTRRL-HOP-382`
   params: `{"entropy_rate": 0.0003, "env_params.batch_size": 2, "eta_f": 1.0, "eta_pi": 0.2, "gamma": 0.99, "lambda_pi": 0.9, "lambda_rnn": 0.97, "lambda_v": 0.93, "normalize_obs": true, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.001, "optimizer_params_td.learning_rate": 0.0001, "pass_obs": false, "trace_mode": "dutch", "update_period": 0.03}`
10. `config/rtrrl_hop_383.yml` `RTRRL-HOP-383`
   params: `{"entropy_rate": 0.0001, "env_params.batch_size": 14, "eta_f": 1.0, "eta_pi": 2.0, "gamma": 0.85, "lambda_pi": 0.965, "lambda_rnn": 0.99, "lambda_v": 0.99, "normalize_obs": true, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.001, "optimizer_params_td.learning_rate": 0.01, "pass_obs": true, "trace_mode": "dutch", "update_period": 0.11}`
11. `config/rtrrl_hop_384.yml` `RTRRL-HOP-384`
   params: `{"entropy_rate": 0.0, "env_params.batch_size": 64, "eta_f": 0.1, "eta_pi": 0.25, "gamma": 0.8, "lambda_pi": 0.95, "lambda_rnn": 0.99, "lambda_v": 0.85, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 1e-06, "optimizer_params_td.learning_rate": 3e-07, "pass_obs": true, "trace_mode": "accumulate", "update_period": 0.01}`
12. `config/rtrrl_hop_385.yml` `RTRRL-HOP-385`
   params: `{"entropy_rate": 1e-07, "env_params.batch_size": 4, "eta_f": 0.6, "eta_pi": 1.0, "gamma": 0.85, "lambda_pi": 0.975, "lambda_rnn": 0.945, "lambda_v": 0.97, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 3e-05, "optimizer_params_td.learning_rate": 0.001, "pass_obs": true, "trace_mode": "dutch", "update_period": 0.01}`
13. `config/rtrrl_hop_386.yml` `RTRRL-HOP-386`
   params: `{"entropy_rate": 0.0015, "env_params.batch_size": 4, "eta_f": 0.03, "eta_pi": 0.3, "gamma": 0.8, "lambda_pi": 0.7, "lambda_rnn": 0.945, "lambda_v": 0.95, "normalize_obs": true, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 7e-06, "optimizer_params_td.learning_rate": 3e-07, "pass_obs": true, "trace_mode": "dutch", "update_period": 0.7}`
14. `config/rtrrl_hop_387.yml` `RTRRL-HOP-387`
   params: `{"entropy_rate": 3e-06, "env_params.batch_size": 24, "eta_f": 0.15, "eta_pi": 5.0, "gamma": 0.97, "lambda_pi": 0.5, "lambda_rnn": 0.955, "lambda_v": 0.7, "normalize_obs": true, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 3e-07, "optimizer_params_td.learning_rate": 0.001, "pass_obs": true, "trace_mode": "accumulate", "update_period": 0.11}`
15. `config/rtrrl_hop_388.yml` `RTRRL-HOP-388`
   params: `{"entropy_rate": 0.0015, "env_params.batch_size": 8, "eta_f": 0.15, "eta_pi": 10.0, "gamma": 0.99, "lambda_pi": 0.96, "lambda_rnn": 0.96, "lambda_v": 0.97, "normalize_obs": true, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 1e-05, "optimizer_params_td.learning_rate": 0.001, "pass_obs": true, "trace_mode": "accumulate", "update_period": 0.08}`
16. `config/rtrrl_hop_389.yml` `RTRRL-HOP-389`
   params: `{"entropy_rate": 3e-05, "env_params.batch_size": 24, "eta_f": 0.45, "eta_pi": 0.32, "gamma": 0.93, "lambda_pi": 0.95, "lambda_rnn": 0.97, "lambda_v": 0.95, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 3e-05, "optimizer_params_td.learning_rate": 3e-06, "pass_obs": true, "trace_mode": "dutch", "update_period": 0.3}`
17. `config/rtrrl_hop_390.yml` `RTRRL-HOP-390`
   params: `{"entropy_rate": 0.0001, "env_params.batch_size": 16, "eta_f": 1.0, "eta_pi": 0.5, "gamma": 0.995, "lambda_pi": 0.99, "lambda_rnn": 0.7, "lambda_v": 0.88, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 1e-05, "optimizer_params_td.learning_rate": 1e-05, "pass_obs": true, "trace_mode": "dutch", "update_period": 0.1}`
18. `config/rtrrl_hop_391.yml` `RTRRL-HOP-391`
   params: `{"entropy_rate": 0.001, "env_params.batch_size": 24, "eta_f": 0.65, "eta_pi": 5.0, "gamma": 0.85, "lambda_pi": 0.99, "lambda_rnn": 0.945, "lambda_v": 0.93, "normalize_obs": true, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.0001, "optimizer_params_td.learning_rate": 0.01, "pass_obs": true, "trace_mode": "dutch", "update_period": 0.09}`
19. `config/rtrrl_hop_392.yml` `RTRRL-HOP-392`
   params: `{"entropy_rate": 0.003, "env_params.batch_size": 16, "eta_f": 2.0, "eta_pi": 0.03, "gamma": 0.95, "lambda_pi": 0.93, "lambda_rnn": 0.5, "lambda_v": 0.97, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 7e-06, "optimizer_params_td.learning_rate": 1e-07, "pass_obs": true, "trace_mode": "dutch", "update_period": 0.3}`
20. `config/rtrrl_hop_393.yml` `RTRRL-HOP-393`
   params: `{"entropy_rate": 0.003, "env_params.batch_size": 64, "eta_f": 0.12, "eta_pi": 0.3, "gamma": 0.97, "lambda_pi": 0.7, "lambda_rnn": 0.94, "lambda_v": 0.97, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 3e-06, "optimizer_params_td.learning_rate": 0.001, "pass_obs": true, "trace_mode": "accumulate", "update_period": 0.5}`
21. `config/rtrrl_hop_394.yml` `RTRRL-HOP-394`
   params: `{"entropy_rate": 0.0, "env_params.batch_size": 1, "eta_f": 2.0, "eta_pi": 5.0, "gamma": 0.9, "lambda_pi": 0.93, "lambda_rnn": 0.85, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 3e-07, "optimizer_params_td.learning_rate": 2e-05, "pass_obs": false, "trace_mode": "dutch", "update_period": 0.2}`
22. `config/rtrrl_hop_395.yml` `RTRRL-HOP-395`
   params: `{"entropy_rate": 0.0, "env_params.batch_size": 8, "eta_f": 0.0, "eta_pi": 5.0, "gamma": 0.85, "lambda_pi": 0.5, "lambda_rnn": 0.95, "lambda_v": 0.97, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.01, "optimizer_params_td.learning_rate": 0.003, "pass_obs": true, "trace_mode": "dutch", "update_period": 0.1}`
23. `config/rtrrl_hop_396.yml` `RTRRL-HOP-396`
   params: `{"entropy_rate": 1e-08, "env_params.batch_size": 12, "eta_f": 0.12, "eta_pi": 0.32, "gamma": 0.97, "lambda_pi": 0.9, "lambda_rnn": 0.955, "lambda_v": 0.7, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 3e-05, "optimizer_params_td.learning_rate": 3e-05, "pass_obs": true, "trace_mode": "accumulate", "update_period": 0.03}`
24. `config/rtrrl_hop_397.yml` `RTRRL-HOP-397`
   params: `{"entropy_rate": 0.01, "env_params.batch_size": 12, "eta_f": 0.08, "eta_pi": 0.38, "gamma": 0.93, "lambda_pi": 0.97, "lambda_rnn": 0.7, "lambda_v": 0.95, "normalize_obs": true, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 0.003, "optimizer_params_td.learning_rate": 1e-07, "pass_obs": true, "trace_mode": "dutch", "update_period": 0.2}`
25. `config/rtrrl_hop_398.yml` `RTRRL-HOP-398`
   params: `{"entropy_rate": 3e-05, "env_params.batch_size": 16, "eta_f": 1.0, "eta_pi": 0.5, "gamma": 0.85, "lambda_pi": 0.95, "lambda_rnn": 0.85, "lambda_v": 0.7, "normalize_obs": true, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 1e-05, "optimizer_params_td.learning_rate": 0.003, "pass_obs": true, "trace_mode": "dutch", "update_period": 0.09}`
26. `config/rtrrl_hop_399.yml` `RTRRL-HOP-399`
   params: `{"entropy_rate": 0.0, "env_params.batch_size": 12, "eta_f": 0.05, "eta_pi": 2.5, "gamma": 0.8, "lambda_pi": 0.5, "lambda_rnn": 0.5, "lambda_v": 0.88, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 5e-06, "optimizer_params_td.learning_rate": 0.003, "pass_obs": false, "trace_mode": "accumulate", "update_period": 0.11}`
27. `config/rtrrl_hop_400.yml` `RTRRL-HOP-400`
   params: `{"entropy_rate": 0.0007, "env_params.batch_size": 32, "eta_f": 2.0, "eta_pi": 5.0, "gamma": 0.93, "lambda_pi": 0.93, "lambda_rnn": 0.94, "lambda_v": 0.88, "normalize_obs": true, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 3e-07, "optimizer_params_td.learning_rate": 5e-05, "pass_obs": true, "trace_mode": "accumulate", "update_period": 0.01}`
28. `config/rtrrl_hop_401.yml` `RTRRL-HOP-401`
   params: `{"entropy_rate": 0.001, "env_params.batch_size": 18, "eta_f": 0.25, "eta_pi": 5.0, "gamma": 0.8, "lambda_pi": 0.5, "lambda_rnn": 0.5, "lambda_v": 0.85, "normalize_obs": true, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 1e-06, "optimizer_params_td.learning_rate": 3e-05, "pass_obs": true, "trace_mode": "accumulate", "update_period": 0.3}`
29. `config/rtrrl_hop_402.yml` `RTRRL-HOP-402`
   params: `{"entropy_rate": 0.003, "env_params.batch_size": 16, "eta_f": 0.08, "eta_pi": 0.03, "gamma": 0.9, "lambda_pi": 0.965, "lambda_rnn": 0.5, "lambda_v": 0.97, "normalize_obs": false, "normalize_reward": true, "optimizer_params_rnn.learning_rate": 3e-07, "optimizer_params_td.learning_rate": 0.0001, "pass_obs": true, "trace_mode": "dutch", "update_period": 0.7}`
30. `config/rtrrl_hop_403.yml` `RTRRL-HOP-403`
   params: `{"entropy_rate": 0.002, "env_params.batch_size": 14, "eta_f": 0.4, "eta_pi": 0.4, "gamma": 0.95, "lambda_pi": 0.85, "lambda_rnn": 0.93, "lambda_v": 0.5, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.0003, "optimizer_params_td.learning_rate": 0.0001, "pass_obs": true, "trace_mode": "accumulate", "update_period": 0.12}`
31. `config/rtrrl_hop_404.yml` `RTRRL-HOP-404`
   params: `{"entropy_rate": 0.0015, "env_params.batch_size": 64, "eta_f": 0.5, "eta_pi": 0.35, "gamma": 0.8, "lambda_pi": 0.5, "lambda_rnn": 0.99, "lambda_v": 0.99, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 0.001, "optimizer_params_td.learning_rate": 1e-06, "pass_obs": true, "trace_mode": "dutch", "update_period": 0.07}`
32. `config/rtrrl_hop_405.yml` `RTRRL-HOP-405`
   params: `{"entropy_rate": 0.0001, "env_params.batch_size": 32, "eta_f": 0.08, "eta_pi": 2.0, "gamma": 0.995, "lambda_pi": 0.98, "lambda_rnn": 0.96, "lambda_v": 0.88, "normalize_obs": false, "normalize_reward": false, "optimizer_params_rnn.learning_rate": 3e-06, "optimizer_params_td.learning_rate": 2e-05, "pass_obs": false, "trace_mode": "accumulate", "update_period": 0.01}`
