import numpy as np
from tensorbem.geometry import PanelMesh, regular_polygon, ellipse_mesh
from tensorbem.solver import compute_added_mass_matrix_2d

rho = 1025.0

def rect_mesh(A, Hb, n_side):
    # CCW rectangle, half-width A in x, half-height Hb in y, uniform panels per side
    xs = np.linspace(-A, A, n_side + 1)
    ys = np.linspace(-Hb, Hb, n_side + 1)
    bottom = np.column_stack((xs[:-1], -Hb * np.ones(n_side)))
    right = np.column_stack((A * np.ones(n_side), ys[:-1]))
    top = np.column_stack((xs[::-1][:-1], Hb * np.ones(n_side)))
    left = np.column_stack((-A * np.ones(n_side), ys[::-1][:-1]))
    return PanelMesh(np.vstack([bottom, right, top, left]))

# 1. circle exact pi rho a^2
for n in (64, 128, 256):
    m = regular_polygon(n, radius=1.0)
    r = compute_added_mass_matrix_2d(m, density=rho)
    print("circle n=%4d  A11=%.9f A22=%.9f  exact=%.9f  relerr=%.3e" % (
        n, r.added_mass_matrix[0,0], r.added_mass_matrix[1,1],
        np.pi*rho, abs(r.added_mass_matrix[1,1]-np.pi*rho)/(np.pi*rho)))

# 2. ellipse exact pi rho a^2 (vertical motion)
for n in (64, 128, 256):
    m = ellipse_mesh(n, semi_major=2.0, semi_minor=1.0)
    r = compute_added_mass_matrix_2d(m, density=rho)
    ex = np.pi*rho*4.0
    print("ellipse n=%4d A22=%.6f exact=%.6f relerr=%.3e" % (n, r.added_mass_matrix[1,1], ex, abs(r.added_mass_matrix[1,1]-ex)/ex))

# 3. rectangle doubled body
A, Hb = 10.0, 6.0
for n in (8, 16, 32, 64, 128):
    m = rect_mesh(A, Hb, n)
    r = compute_added_mass_matrix_2d(m, density=rho)
    a22 = r.added_mass_matrix[1,1]
    print("rect half=%.1fx%.1f n=%4d  A22=%.3f  half=%.3f  t/m=%.3f" % (A, Hb, n, a22, a22/2, a22/2/1000))

# 4. plate limits: A/Hb large -> pi rho A^2 ; A/Hb small -> pi rho Hb^2
for (a, h) in ((40.0, 1.0), (1.0, 40.0)):
    m = rect_mesh(a, h, 96)
    r = compute_added_mass_matrix_2d(m, density=rho)
    a22 = r.added_mass_matrix[1,1]
    ex = np.pi*rho*(a**2 if a > h else h**2)
    print("limit A=%.0f Hb=%.0f A22=%.3f exact_plate=%.3f rel=%.3e" % (a, h, a22, ex, abs(a22-ex)/ex))