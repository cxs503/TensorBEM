# Context-craft offline A/B

- tasks: 40 · corpus fact units: 5219
- arms: head_dump, bm25_raw, brief_global, brief_local

## budget 400 tokens

| arm | recall | 95%CI | tolerance recall | gate | tokens | redundancy | decision share |
|---|---|---|---|---|---|---|---|
| head_dump | 0.005 | ±0.010 | 0.000 | 0.000 | 396 | 0.002 | 0.240 |
| bm25_raw | 0.103 | ±0.050 | 0.104 | 0.225 | 398 | 0.007 | 0.279 |
| brief_global | 0.100 | ±0.033 | 0.227 | 0.550 | 399 | 0.000 | 0.717 |
| brief_local | 0.304 | ±0.063 | 0.461 | 0.750 | 398 | 0.001 | 0.586 |

## budget 800 tokens

| arm | recall | 95%CI | tolerance recall | gate | tokens | redundancy | decision share |
|---|---|---|---|---|---|---|---|
| head_dump | 0.257 | ±0.129 | 0.206 | 0.250 | 782 | 0.003 | 0.276 |
| bm25_raw | 0.215 | ±0.061 | 0.175 | 0.400 | 798 | 0.016 | 0.301 |
| brief_global | 0.137 | ±0.034 | 0.316 | 0.600 | 799 | 0.000 | 0.776 |
| brief_local | 0.513 | ±0.073 | 0.544 | 0.800 | 798 | 0.000 | 0.487 |

## budget 1200 tokens

| arm | recall | 95%CI | tolerance recall | gate | tokens | redundancy | decision share |
|---|---|---|---|---|---|---|---|
| head_dump | 0.636 | ±0.128 | 0.436 | 0.550 | 1077 | 0.003 | 0.299 |
| bm25_raw | 0.291 | ±0.062 | 0.242 | 0.575 | 1199 | 0.022 | 0.300 |
| brief_global | 0.173 | ±0.040 | 0.378 | 0.600 | 1198 | 0.000 | 0.815 |
| brief_local | 0.697 | ±0.077 | 0.647 | 0.825 | 1198 | 0.001 | 0.457 |

## budget 1600 tokens

| arm | recall | 95%CI | tolerance recall | gate | tokens | redundancy | decision share |
|---|---|---|---|---|---|---|---|
| head_dump | 0.897 | ±0.076 | 0.666 | 0.725 | 1262 | 0.003 | 0.313 |
| bm25_raw | 0.340 | ±0.067 | 0.273 | 0.600 | 1598 | 0.025 | 0.301 |
| brief_global | 0.208 | ±0.045 | 0.432 | 0.625 | 1598 | 0.000 | 0.839 |
| brief_local | 0.815 | ±0.060 | 0.725 | 0.825 | 1599 | 0.001 | 0.480 |
