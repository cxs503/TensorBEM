# m14767 numbers (程序化生成，勿手改)

## 1. 拟合阶梯

| n_poles | peak-normalised max err | pointwise floor err | d (feedthrough) | alpha range | B(w_min) data | B(w_min) fit |
|---|---|---|---|---|---|---|
| 1 | 99.816 % | 116.658 % | 0 | 9.727 … 9.727 | 1.409 | 3.445 |
| 3 | 64.896 % | 127.057 % | 2.486 | 3.25 … 3.263 | 1.409 | 1.041 |
| 5 | 8.016 % | 10.910 % | 0 | 3.125 … 3.346 | 1.409 | 1.398 |
| 8 | 0.847 % | 0.848 % | 0.1076 | 0.1604 … 5.164 | 1.409 | 1.409 |
| 10 | 0.361 % | 0.811 % | 0 | 0.1565 … 14.9 | 1.409 | 1.409 |
| 12 | 0.060 % | 0.114 % | 0.02408 | 0.1224 … 11.91 | 1.409 | 1.409 |
| 15 | 0.052 % | 0.072 % | 0 | 0.1511 … 23.71 | 1.409 | 1.409 |

## 2. 实现（realization）双路径 C4

| omega (rad/s) | |G| analytic | |G| integrated | amp rel err | phase err |
|---|---|---|---|---|
| 0.3 | 189.735 | 189.735 | 0.000 % | 1.55e-06 rad |
| 1 | 861.192 | 861.164 | 0.003 % | 6.73e-06 rad |
| 3 | 1750.22 | 1750.23 | 0.000 % | 1.96e-06 rad |
| 10 | 482.284 | 482.283 | 0.000 % | 3.84e-07 rad |
| 25 | 162.709 | 162.709 | 0.000 % | 1.74e-07 rad |

### feedthrough D 项审计（合成曲线，真值 d = 50）

拟合得到 d = 50，peak err = 0.000 %

| omega (rad/s) | amp rel err (含 D) | amp rel err (丢掉 D = m14767 之前的行为) |
|---|---|---|
| 0.5 | 0.000 % | 2.814 % |
| 2 | 0.001 % | 3.647 % |
| 4 | 0.001 % | 4.566 % |

## 3. 自由衰减 SS vs IRF

| 量 | SS 辅助 ODE | IRF 卷积 (K(t) 由列表 B 积分) | 相对差 |
|---|---|---|---|
| 固有周期 (s) | 1.94433 | 1.94539 | 0.055 % |
| 衰减率 (1/s) | 0.30636 | 0.30827 | 0.618 % |
| 30 s 后幅值 / 初值 | — | — | SS 0.009 % |
| 两条路径最大位移差 (m) | — | — | 0.0001963 (峰值 0.05) |
| RK4 内部子步数 | 1 | 1 | — |

## 4. 判据 C1–C6（10 极点，Hulme 半球）

| 判据 | 量 | 值 | pass |
|---|---|---|---|
| C1 fitting | peak-normalised max err | 0.361 % | True |
| C2 stability | min pole | -14.9 | True |
| C3 free decay | period rel err | 0.645 % | True |
| C3 free decay | decay-rate rel err | 0.044 % | True |
| C4 dual path | memory amp rel err | 0.000 % | True |
| C4 dual path | memory phase err | 2.72e-06 rad | True |
| C5 Kramers-Kronig | max rel A residual | 2.969 % | True |
| C6 negative control | order-1 peak err | 99.816 % | True |

all_pass = **True**
