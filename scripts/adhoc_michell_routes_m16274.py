"""Ad-hoc cross-route comparison of the Michell wave-resistance implementations."""

import numpy as np

from tensorbem import michell, thin_ship_michell, wigley

L, B, T, rho, g = 10.0, 1.0, 0.625, 1025.0, 9.80665
for Fn in (0.25, 0.30, 0.35):
    U = Fn * np.sqrt(g * L)
    a = wigley.michell_wave_resistance(L, B, T, U, rho, g)
    b = michell.michell_resistance(
        lambda kk: michell.wigley_kochin_closed_form(L, B, T, kk), U, T, rho, g
    )
    hull = thin_ship_michell.WigleyHull(L=L, B=B, T=T)
    c = thin_ship_michell.michell_wave_resistance(hull, U=U, rho=rho, gravity=g)
    d = thin_ship_michell.wave_resistance_theta_form(hull, U=U, rho=rho, gravity=g)
    e = wigley.wave_resistance_kochin(L, B, T, U, rho, g)
    print(f"Fn={Fn}: wigley.michell_wave_resistance={a:.4f}  michell.closed_form={b:.4f} "
          f" thin_ship={c:.4f}  theta_form={d:.4f}  wigley.kochin={e:.4f}")