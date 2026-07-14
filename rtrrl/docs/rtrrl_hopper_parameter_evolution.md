# RTRRL Hopper Parameter Evolution

Color rule: <span style="color:green">green</span> means the value increased
relative to the parent run; <span style="color:red">red</span> means it
decreased. Colors indicate direction only, not quality.

```mermaid
flowchart TD
    R001["RTRRL-HOP-001<br/>baseline: paral64, lr1e-3<br/>lambda0.99, unmasked<br/>patience=100"]
    R002["RTRRL-HOP-002<br/>001 + patience 100→10000"]
    R003["RTRRL-HOP-003<br/>002 + paral 64→1"]
    R004["RTRRL-HOP-004<br/>003 + lr 1e-3→1e-4"]
    R005["RTRRL-HOP-005<br/>002 + lr 1e-3→1e-4"]
    R006["RTRRL-HOP-006<br/>004 + mask none→even<br/>lambda 0.99→0.9"]
    R007["RTRRL-HOP-007<br/>006 + lambda 0.9→0.99"]
    R008["RTRRL-HOP-008<br/>006 + episodes 10k→50k<br/>patience 10000→20"]
    R009["RTRRL-HOP-009<br/>004 + fixed trace/action"]
    R010["RTRRL-HOP-010<br/>008 + eta_f 1→0"]
    R011["RTRRL-HOP-011<br/>002 + normalize_obs False→True"]
    R012["RTRRL-HOP-012<br/>011 + paral 64→1"]

    R001 --> R002
    R002 --> R003 --> R004
    R002 --> R005
    R004 --> R006 --> R007
    R006 --> R008 --> R010
    R004 --> R009
    R002 --> R011 --> R012
```

| Run | Parent | Core change | paral | mask | norm_obs | lr | lambda | episodes | patience | best eval | last eval | duration |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| RTRRL-HOP-001 | root | baseline | 64 | none | False | 1e-3 | 0.99 | 10000 | 100 | 100.30 | 28.27 | 91.4m |
| RTRRL-HOP-002 | 001 | patience 100→10000 | 64 | none | False | 1e-3 | 0.99 | 10000 | <span style="color:green">10000</span> | <span style="color:green">295.41</span> | <span style="color:red">4.36</span> | <span style="color:green">181.7m</span> |
| RTRRL-HOP-003 | 002 | paral 64→1 | <span style="color:red">1</span> | none | False | 1e-3 | 0.99 | 10000 | 10000 | <span style="color:red">93.88</span> | <span style="color:green">20.79</span> | <span style="color:red">92.9m</span> |
| RTRRL-HOP-004 | 003 | lr 1e-3→1e-4 | 1 | none | False | <span style="color:red">1e-4</span> | 0.99 | 10000 | 10000 | <span style="color:green">230.95</span> | <span style="color:green">73.41</span> | <span style="color:green">100.4m</span> |
| RTRRL-HOP-005 | 002 | lr 1e-3→1e-4 | 64 | none | False | <span style="color:red">1e-4</span> | 0.99 | 10000 | 10000 | <span style="color:red">40.30</span> | <span style="color:green">18.70</span> | <span style="color:red">92.0m</span> |
| RTRRL-HOP-006 | 004 | mask none→even + lambda 0.99→0.9 | 1 | even | False | 1e-4 | <span style="color:red">0.9</span> | 10000 | 10000 | <span style="color:red">132.58</span> | <span style="color:red">61.78</span> | <span style="color:red">100.1m</span> |
| RTRRL-HOP-007 | 006 | lambda 0.9→0.99 | 1 | even | False | 1e-4 | <span style="color:green">0.99</span> | 10000 | 10000 | <span style="color:red">30.50</span> | <span style="color:red">4.76</span> | <span style="color:green">111.5m</span> |
| RTRRL-HOP-008 | 006 | episodes 10k→50k + patience 10000→20 | 1 | even | False | 1e-4 | 0.9 | <span style="color:green">50000</span> | <span style="color:red">20</span> | <span style="color:green">514.85</span> | <span style="color:red">21.81</span> | <span style="color:red">98.4m</span> |
| RTRRL-HOP-009 | 004 | fixed trace/action implementation | 1 | none | False | 1e-4 | 0.99 | 10000 | 10000 | <span style="color:red">65.63</span> | <span style="color:red">14.01</span> | <span style="color:red">97.1m</span> |
| RTRRL-HOP-010 | 008 | eta_f 1→0 | 1 | even | False | 1e-4 | 0.9 | 50000 | 20 | <span style="color:red">187.58</span> | <span style="color:green">64.28</span> | <span style="color:red">90.5m</span> |
| RTRRL-HOP-011 | 002 | normalize_obs False→True | 64 | none | True | 1e-3 | 0.99 | 10000 | 10000 | <span style="color:red">230.25</span> | <span style="color:green">136.62</span> | <span style="color:red">181.0m</span> |
| RTRRL-HOP-012 | 011 | paral 64→1 | <span style="color:red">1</span> | none | True | 1e-3 | 0.99 | 10000 | 10000 | <span style="color:red">20.06</span> | <span style="color:red">14.54</span> | <span style="color:red">94.5m</span> |

Notes:

- `RTRRL-HOP-008` has a high best eval, but the best point is at step `1000`
  (first eval), so it should not be interpreted as a learned curve.
- `RTRRL-HOP-011` is the current local `paral64 + normalize_obs + lr=1e-3 +
  no-mask` run. It is below `RTRRL-HOP-002` in best eval, but has a higher final
  eval value.
- `RTRRL-HOP-012` shows the same normalize-observation setting with `paral=1`
  remains weak.
