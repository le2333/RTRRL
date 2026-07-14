# PPO HalfCheetah Experiment Index

Naming convention: `PPO-HC-XXX <parent>+ <key change>`; runs with more than two key changes are treated as new roots.

| ID | Hash | Name | Parent | Key change | Result |
|---|---|---|---|---|---|
| PPO-HC-001 | `ec57b326` | PPO-HC-001 baseline masked official | root | baseline masked official | max=2122.4, 5M=1143.9, 10M=2004.0, 20M=2051.9, last=239.9 |
| PPO-HC-002 | `3a9ba608` | PPO-HC-002 001+ learning_rate 0.0003->0.001 | PPO-HC-001 | learning_rate 0.0003->0.001 | max=1946.5, 5M=1072.0, 10M=1873.9, 20M=1923.9, last=224.7 |
| PPO-HC-003 | `5f2c18d3` | PPO-HC-003 001+ unroll_length 20->5 | PPO-HC-001 | unroll_length 20->5 | max=1357.7, 5M=1341.3, 10M=1343.3, 20M=1343.3, last=1357.7 |
| PPO-HC-004 | `71d17a7b` | PPO-HC-004 001+ unroll_length 20->10 | PPO-HC-001 | unroll_length 20->10 | max=1828.1, 5M=1792.5, 10M=1641.4, 20M=1189.9, last=1796.5 |
| PPO-HC-005 | `de9944c6` | PPO-HC-005 004+ learning_rate 0.0003->0.001 | PPO-HC-004 | learning_rate 0.0003->0.001 | max=1889.5, 5M=1797.4, 10M=1594.9, 20M=1072.1, last=1807.2 |
| PPO-HC-006 | `70b2a038` | PPO-HC-006 002 rerun | PPO-HC-002 | rerun | max=1946.5, 5M=1072.0, 10M=1873.9, 20M=1923.9, last=224.7 |
| PPO-HC-007 | `558498bc` | PPO-HC-007 005+ num_updates_per_batch 8->16 | PPO-HC-005 | num_updates_per_batch 8->16 | max=1733.4, 5M=1659.4, 10M=1550.9, 20M=1093.2, last=1686.3 |
| PPO-HC-008 | `701703ba` | PPO-HC-008 004+ learning_rate 0.0003->0.003 | PPO-HC-004 | learning_rate 0.0003->0.003 | max=1712.2, 5M=1628.5, 10M=1428.2, 20M=1048.9, last=1655.4 |
| PPO-HC-009 | `ae1df624` | PPO-HC-009 001+ batch_size 512->64 | PPO-HC-001 | batch_size 512->64 | max=1844.0, 5M=1785.8, 10M=1574.8, 20M=1832.1, last=1809.3 |
| PPO-HC-010 | `8bf31efc` | PPO-HC-010 001+ batch_size 512->128 | PPO-HC-001 | batch_size 512->128 | max=1850.2, 5M=1769.3, 10M=1840.9, 20M=1833.0, last=1848.8 |
| PPO-HC-011 | `2bbfbed9` | PPO-HC-011 001+ batch_size 512->256 | PPO-HC-001 | batch_size 512->256 | max=2147.5, 5M=2074.3, 10M=1843.1, 20M=1242.8, last=2123.2 |
| PPO-HC-012 | `12ae4d76` | PPO-HC-012 011+ learning_rate 0.0003->0.0005 | PPO-HC-011 | learning_rate 0.0003->0.0005 | max=2111.1, 5M=2010.6, 10M=1771.3, 20M=1196.5, last=2064.4 |
| PPO-HC-013 | `36abbd2b` | PPO-HC-013 010+ learning_rate 0.0003->0.0005 | PPO-HC-010 | learning_rate 0.0003->0.0005 | max=1762.5, 5M=1644.3, 10M=1704.6, 20M=1691.3, last=1762.5 |
| PPO-HC-014 | `24f978a2` | PPO-HC-014 root batch64 stable-explore | root | new root; batch_size 512->64; learning_rate 0.0003->0.0001; entropy_cost 0.001->0.003; clipping_epsilon 0.3->0.2 | max=2141.9, 5M=1910.2, 10M=1499.2, 20M=2141.9, last=1959.1 |
| PPO-HC-015 | `a6276eaf` | PPO-HC-015 014+ batch_size 64->128 | PPO-HC-014 | batch_size 64->128 | max=2101.4, 5M=1782.2, 10M=1959.3, 20M=1936.6, last=2101.4 |
| PPO-HC-016 | `2323dc06` | PPO-HC-016 011+ entropy_cost 0.001->0.003, clipping_epsilon 0.3->0.2 | PPO-HC-011 | entropy_cost 0.001->0.003; clipping_epsilon 0.3->0.2 | max=2195.6, 5M=2041.8, 10M=1707.2, 20M=1043.7, last=2113.9 |
| PPO-HC-017 | `598da7bc` | PPO-HC-017 001+ entropy_cost 0.001->0.003 | PPO-HC-001 | entropy_cost 0.001->0.003 | max=2123.1, 5M=1125.1, 10M=1996.6, 20M=2049.7, last=235.7 |
| PPO-HC-018 | `1e9be56d` | PPO-HC-018 001+ entropy_cost 0.001->0.005 | PPO-HC-001 | entropy_cost 0.001->0.005 | max=2198.0, 5M=1123.8, 10M=2005.6, 20M=2112.7, last=219.8 |
| PPO-HC-019 | `96b72c55` | PPO-HC-019 001+ entropy_cost 0.001->0.01 | PPO-HC-001 | entropy_cost 0.001->0.01 | max=2352.4, 5M=1112.4, 10M=2149.2, 20M=2281.4, last=217.7 |
| PPO-HC-020 | `aab774eb` | PPO-HC-020 001+ entropy_cost 0.001->0.0125 | PPO-HC-001 | entropy_cost 0.001->0.0125 | max=2295.2, 5M=1120.9, 10M=2087.3, 20M=2181.5, last=217.2 |
| PPO-HC-021 | `74e9c52f` | PPO-HC-021 001+ entropy_cost 0.001->0.015 | PPO-HC-001 | entropy_cost 0.001->0.015 | max=2198.9, 5M=1084.1, 10M=2011.2, 20M=2089.1, last=217.4 |
| PPO-HC-022 | `da71c8b9` | PPO-HC-022 001+ entropy_cost 0.001->0.02 | PPO-HC-001 | entropy_cost 0.001->0.02 | max=2084.6, 5M=1100.0, 10M=1866.9, 20M=1955.3, last=198.3 |
| PPO-HC-023 | `6262e938` | PPO-HC-023 001+ entropy_cost 0.001->0.008 | PPO-HC-001 | entropy_cost 0.001->0.008 | max=2262.3, 5M=1114.4, 10M=2045.7, 20M=2175.2, last=225.6 |
| PPO-HC-024 | `3009a085` | PPO-HC-024 001+ entropy_cost 0.001->0.009 | PPO-HC-001 | entropy_cost 0.001->0.009 | max=2283.9, 5M=1112.0, 10M=2055.8, 20M=2217.5, last=225.2 |
| PPO-HC-025 | `70521fe2` | PPO-HC-025 001+ entropy_cost 0.001->0.011 | PPO-HC-001 | entropy_cost 0.001->0.011 | max=2247.6, 5M=1123.2, 10M=2024.4, 20M=2149.5, last=237.8 |
| PPO-HC-026 | `be0e52aa` | PPO-HC-026 019+ clipping_epsilon 0.3->0.2 | PPO-HC-019 | clipping_epsilon 0.3->0.2 | max=2045.2, 5M=858.2, 10M=1806.2, 20M=1924.6, last=44.2 |
| PPO-HC-027 | `190e8b30` | PPO-HC-027 019+ minibatch_split 512x32->64x256 | PPO-HC-019 | minibatch_split 512x32->64x256 | max=2376.4, 5M=1081.8, 10M=2069.1, 20M=2236.1, last=226.4 |
| PPO-HC-028 | `33f4a8a5` | PPO-HC-028 027+ learning_rate 0.0003->0.001 | PPO-HC-027 | learning_rate 0.0003->0.001 | max=2009.3, 5M=1009.0, 10M=1865.3, 20M=1911.9, last=132.5 |
| PPO-HC-029 | `afbc341d` | PPO-HC-029 027+ num_updates_per_batch 8->12 | PPO-HC-027 | num_updates_per_batch 8->12 | max=2399.3, 5M=1087.7, 10M=2130.7, 20M=2281.4, last=226.4 |
| PPO-HC-030 | `e4396f23` | PPO-HC-030 027+ gae_lambda 0.95->0.99 | PPO-HC-027 | gae_lambda 0.95->0.99 | max=2393.0, 5M=1107.8, 10M=2082.9, 20M=2271.4, last=212.3 |
| PPO-HC-031 | `6fa6b996` | PPO-HC-031 027+ minibatch_total 64x256->64x128 incomplete | PPO-HC-027 | minibatch_total 64x256->64x128 | incomplete; partial max=1729.7, eval2=696.4, eval4=1470.8, last=1729.7 |
| PPO-HC-032 | `586136f4` | PPO-HC-032 027+ minibatch_total 64x256->64x512 | PPO-HC-027 | minibatch_total 64x256->64x512 | max=2133.7, 5M=1957.6, 10M=1850.8, 20M=1619.5, last=1450.1 |
| PPO-HC-033 | `3cd65122` | PPO-HC-033 029+ num_updates_per_batch 12->16 | PPO-HC-029 | num_updates_per_batch 12->16 | max=2369.1, 5M=1131.0, 10M=2128.3, 20M=2241.1, last=192.2 |
| PPO-HC-034 | `e6cd50a8` | PPO-HC-034 031 rerun | PPO-HC-031 | rerun | step-sorted: max=2546.8, 5M=1249.4, 10M=1630.0, 20M=1894.7, last=2546.8 |
| PPO-HC-035 | `3cfc74dc` | PPO-HC-035 034+ minibatch_total 64x128->64x64 | PPO-HC-034 | minibatch_total 64x128->64x64 | step-sorted: max=2625.2, 5M=1428.1, 10M=1781.1, 20M=2063.8, last=2625.2 |
| PPO-HC-036 | `f78027ec` | PPO-HC-036 034+ unroll_length 20->10 | PPO-HC-034 | unroll_length 20->10 | step-sorted: max=2494.6, 5M=1505.9, 10M=1734.6, 20M=2094.4, last=2395.5 |
| PPO-HC-037 | `8f5851b3` | PPO-HC-037 034+ num_updates_per_batch 8->16 | PPO-HC-034 | num_updates_per_batch 8->16 | step-sorted: max=2589.1, 5M=1036.7, 10M=1599.2, 20M=1990.7, last=2579.0 |
