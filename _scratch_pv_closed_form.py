import numpy as np
from scipy.special import exp1
import sys
sys.path.insert(0, "/root/tp-agent/work/m16829/src")
from tensorbem import free_surface

NU = 1.0
d = 0.48
sigma = NU * d

def pv_ref(nu, dd, x, npts=400001, eps=1e-7):
    k_hi = max(60.0, 40.0 / dd)
    k = np.linspace(0.0, k_hi, npts)
    g = np.exp(-k * dd) * np.cos(k * x)
    f = np.divide(g, k - nu, out=np.zeros_like(g), where=np.abs(k - nu) >= eps)
    return float(np.trapezoid(f, k))

for X in (0.0, 1.0, 3.0, 5.3, 9.4, 13.3, 30.0, 60.0):
    b = complex(sigma, -X)
    closed = np.real(-np.exp(-b) * exp1(-b))   # hypothesis: -e^{-b} Ei(b)
    ref = pv_ref(NU, d, X / NU)
    lib = free_surface.principal_value_integral(NU, d, X / NU)
    print(f"X={X:6.2f} ref={ref: .12f} closed={closed: .12f} "
          f"d_closed={abs(closed-ref)/max(abs(ref),1e-30):.2e} lib={lib: .12f} "
          f"d_lib={abs(lib-ref)/max(abs(ref),1e-30):.2e}")

print("--- tail split ---")
LEGX, LEGW = free_surface._LEGENDRE_X, free_surface._LEGENDRE_W
def head(nu, dd, x):
    ref = np.exp(-nu * dd) * np.cos(nu * x)
    k = nu * (1.0 + LEGX)
    diff = np.exp(-k * dd) * np.cos(k * x) - ref
    return float(np.sum(LEGW * diff / LEGX))
for X in (0.0, 1.0, 3.0, 5.3, 9.4, 13.3, 30.0, 60.0, 120.0):
    b = complex(sigma, -X)
    tail = np.real(np.exp(-b) * exp1(b))
    x = X / NU
    h = head(NU, d, x)
    ref = pv_ref(NU, d, x)
    print(f"X={X:6.2f} ref={ref: .12f} head+tail={h+tail: .12f} "
          f"head_alone_dev={abs(h-ref)/abs(ref):.2e} hybrid_dev={abs(h+tail-ref)/abs(ref):.2e}")