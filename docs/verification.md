# Numerical formulation and verification plan

## Formulation

For the 2-D Laplace equation \(\nabla^2 u=0\), use the fundamental solution

\[
G(x,y)=-\frac{1}{2\pi}\log|x-y|, \qquad -\nabla_y^2 G=\delta(x-y).
\]

With outward normal derivative \(q=\partial u/\partial n\), the boundary integral equation at a smooth boundary point is

\[
c(x)u(x)=\int_\Gamma G(x,y)q(y)\,ds_y-
\int_\Gamma \frac{\partial G(x,y)}{\partial n_y}u(y)\,ds_y,
\]

where \(c=1/2\) at a smooth boundary point. Constant panels and midpoint collocation give

\[
\mathbf G\mathbf q=(\tfrac12\mathbf I+\mathbf H)\mathbf u.
\]

The self-panel single-layer logarithmic integral is evaluated analytically. Off-diagonal integrals use midpoint quadrature, and the straight-panel self double-layer principal value is zero.

## What the tests establish

- geometry validation, panel normals, and kernel sign convention;
- finite output and small algebraic linear-system residual;
- basic constant-boundary-value behavior.

These tests are necessary but not sufficient to claim a production-quality BEM solver. The unit-disk benchmark compares against the circular harmonic reference on a polygonal geometry; convergence trends must be checked over increasing panel counts.

## Verification roadmap

1. Add a circle-specific quadrature/reference study and publish convergence tables.
2. Add independent manufactured solutions with known boundary flux.
3. Compare with an independent trusted implementation or published benchmark.
4. Add robust near-singular and corner quadrature, mesh quality diagnostics, and conditioning safeguards.
5. Only then extend to 3-D surface BEM, ship potential flow, wave radiation/diffraction, or acoustics.
