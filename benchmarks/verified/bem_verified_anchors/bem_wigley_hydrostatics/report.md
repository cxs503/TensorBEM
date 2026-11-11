# bem_wigley_hydrostatics

**Verdict:** `PASS (strict all-levels: FAIL)`

| quantity | closed form | value |
|---|---|---|
| displaced volume | 4LBT/9 | 0.190000000 m^3 |
| block coefficient | 4/9 | 0.444444444 |
| waterplane area | 2LB/3 | 1.266666667 m^2 |
| wetted area | quadrature | 2.271790428 m^2 |

| triangles | V (m^3) | dV/V | Cb | dCb/Cb | Sw (m^2) | dSw/Sw |
|---|---|---|---|---|---|---|
| 200 | 0.187644 | -1.240 % | 0.438933 | -1.240 % | 2.268059 | -0.164 % |
| 800 | 0.189407 | -0.312 % | 0.443058 | -0.312 % | 2.270875 | -0.040 % |
| 3200 | 0.189852 | -0.078 % | 0.444097 | -0.078 % | 2.271563 | -0.010 % |

GCI (volume): p = 1.989, U_G = 0.098 %, converged = True

![bem_wigley_hydrostatics](figures/wigley_hydrostatics.png)
