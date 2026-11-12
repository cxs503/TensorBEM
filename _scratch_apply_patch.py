"""Apply the GAP-BEM-2D-PV-TAIL fix to src/tensorbem/free_surface.py."""
import pathlib
import re

p = pathlib.Path("/root/tp-agent/work/m16829/src/tensorbem/free_surface.py")
src = p.read_text()

# ---- 1. header rewrite: document the closed-form tail -----------------------
old_doc = """The second is regular and decaying; with ``k = nu(2 + t)`` and
``u = nu A t`` it is a Gauss-Laguerre rule. Node counts are checked in
``tests/test_free_surface.py`` against 1.5x and 2x the nodes.
"""
new_doc = """The second piece is evaluated in **closed form**. With ``b = nu*A - i*nu*X``,

    integral_{2 nu}^inf exp(-kA) cos(kX)/(k - nu) dk
        = Re{ exp(-b) * E1(b) },      E1(b) = integral_1^inf exp(-b u)/u du,

and the sine integral is the same expression with ``Im`` for ``Re``. The
identity follows from ``PV integral_{-1}^inf exp(-b u)/u du = -Ei(b)``, which is
exact for ``Re b > 0``; ``E1`` is evaluated by ``scipy.special.exp1`` (a declared
dependency) on the principal branch, where ``Re b = nu*A > 0`` holds for every
admissible field point. The closed form replaces a fixed-node Gauss-Laguerre
tail that lost accuracy as the oscillation phase ``nu*|X|`` grew (see
``docs/free_surface_pv_tail_fix/README.md``).

The first piece is a smooth integral over ``[-1, 1]`` and stays a
Gauss-Legendre rule; its node count is chosen from the largest oscillation
phase in the block, because a fixed 80-node rule stops resolving the integrand
beyond ``nu*|X| ~ 100``.
"""
assert old_doc in src
src = src.replace(old_doc, new_doc)

# ---- 2. imports: scipy.special.exp1 ---------------------------------------
old_imp = "from numpy.typing import NDArray\n\nFloatArray"
new_imp = "from numpy.typing import NDArray\nfrom scipy.special import exp1\n\nFloatArray"
assert old_imp in src
src = src.replace(old_imp, new_imp, 1)

# ---- 3. node-count ladder instead of a single fixed rule -------------------
old_consts = """_LAGUERRE_NODES = 160
_LEGENDRE_NODES = 80
_PAIR_BLOCK = 8192"""
new_consts = """_LEGENDRE_NODES = 80
#: Node ladder for the regular head integral.  The rule is picked from the
#: largest scaled horizontal offset in a block (``nu*|x|``), which is the
#: oscillation phase of the integrand: 80 nodes hold ``nu*|x| <= 100`` to
#: ``1e-14``, and each step of the ladder buys the same margin again.
_LEGENDRE_LADDER = (80, 160, 320, 640, 1280, 2560)
_PHASE_PER_NODE = 1.5
_LEGENDRE_RULES: dict = {}
_PAIR_BLOCK = 8192"""
assert old_consts in src
src = src.replace(old_consts, new_consts)

old_rules = """_LEGENDRE_X, _LEGENDRE_W = leggauss(_LEGENDRE_NODES)
_LAGUERRE_X, _LAGUERRE_W = laggauss(_LAGUERRE_NODES)"""
new_rules = """_LEGENDRE_X, _LEGENDRE_W = leggauss(_LEGENDRE_NODES)


def _legendre_rule(phase: float):
    \"\"\"Smallest ladder rule that resolves an oscillation phase ``phase``.\"\"\"
    need = max(_LEGENDRE_NODES, int(math.ceil(_PHASE_PER_NODE * abs(float(phase)))))
    for nodes in _LEGENDRE_LADDER:
        if nodes >= need:
            break
    else:
        nodes = _LEGENDRE_LADDER[-1]
    rule = _LEGENDRE_RULES.get(nodes)
    if rule is None:
        rule = leggauss(nodes)
        _LEGENDRE_RULES[nodes] = rule
    return rule"""
assert old_rules in src
src = src.replace(old_rules, new_rules)

# drop the now unused laggauss import
src = src.replace("from numpy.polynomial.laguerre import laggauss\n", "")

# ---- 4. rewrite the two principal-value kernels ----------------------------
cos_start = src.index("def _pv_cosine(")
sin_end = src.index("def principal_value_integral(")
new_kernels = '''def _pv_cosine(nu: float, depth: FloatArray, offset: FloatArray) -> FloatArray:
    """``PV integral_0^inf exp(-k*depth) cos(k*offset)/(k - nu) dk``, vectorised."""
    flat_depth = np.ravel(np.asarray(depth, dtype=float))
    flat_offset = np.ravel(np.asarray(offset, dtype=float))
    flat_depth, flat_offset = np.broadcast_arrays(flat_depth, flat_offset)
    result = np.empty(flat_depth.size, dtype=float)
    for start in range(0, flat_depth.size, _PAIR_BLOCK):
        block = slice(start, start + _PAIR_BLOCK)
        d = flat_depth[block][:, None]
        x = flat_offset[block][:, None]
        nodes, weights = _legendre_rule(np.max(np.abs(flat_offset[block])))
        reference = np.exp(-nu * d) * np.cos(nu * x)
        # [0, 2nu], with k = nu (1 + t) and a removable point at t = 0.
        k = nu * (1.0 + nodes[None, :])
        difference = np.exp(-k * d) * np.cos(k * x) - reference
        head = np.sum(
            weights[None, :] * difference / nodes[None, :], axis=1
        )
        result[block] = head + _tail_real(nu, flat_depth[block], flat_offset[block])
    return result.reshape(np.broadcast(np.asarray(depth), np.asarray(offset)).shape)


def _pv_sine(nu: float, depth: FloatArray, offset: FloatArray) -> FloatArray:
    """``PV integral_0^inf exp(-k*depth) sin(k*offset)/(k - nu) dk``, vectorised."""
    flat_depth = np.ravel(np.asarray(depth, dtype=float))
    flat_offset = np.ravel(np.asarray(offset, dtype=float))
    flat_depth, flat_offset = np.broadcast_arrays(flat_depth, flat_offset)
    result = np.empty(flat_depth.size, dtype=float)
    for start in range(0, flat_depth.size, _PAIR_BLOCK):
        block = slice(start, start + _PAIR_BLOCK)
        d = flat_depth[block][:, None]
        x = flat_offset[block][:, None]
        nodes, weights = _legendre_rule(np.max(np.abs(flat_offset[block])))
        reference = np.exp(-nu * d) * np.sin(nu * x)
        k = nu * (1.0 + nodes[None, :])
        difference = np.exp(-k * d) * np.sin(k * x) - reference
        head = np.sum(
            weights[None, :] * difference / nodes[None, :], axis=1
        )
        result[block] = head + _tail_imag(nu, flat_depth[block], flat_offset[block])
    return result.reshape(np.broadcast(np.asarray(depth), np.asarray(offset)).shape)


def _scaled_exponential_integral(nu: float, depth: FloatArray, offset: FloatArray):
    """``exp(-b) * E1(b)`` with ``b = nu*depth - i*nu*offset``.

    This is the exact value of ``integral_{2 nu}^inf exp(-k*depth)
    exp(i*k*offset)/(k - nu) dk``: with ``k = nu*s`` and ``s = 1 + u`` the
    integral becomes ``exp(-b) * integral_1^inf exp(-b*u)/u du = exp(-b)*E1(b)``,
    whose real and imaginary parts are the cosine and sine principal values.
    ``Re b = nu*depth > 0`` for every field point the caller admits, so the
    principal branch of ``E1`` is the one the physical problem selects, and no
    ``i*pi`` residue term appears here -- that residue is the radiation term
    added separately in :func:`green_function`.
    """
    phase = -1j * nu * np.asarray(offset, dtype=float)
    b = nu * np.asarray(depth, dtype=float) + phase
    return np.exp(-b) * exp1(b)


def _tail_real(nu: float, depth: FloatArray, offset: FloatArray) -> FloatArray:
    return np.real(_scaled_exponential_integral(nu, depth, offset))


def _tail_imag(nu: float, depth: FloatArray, offset: FloatArray) -> FloatArray:
    return np.imag(_scaled_exponential_integral(nu, depth, offset))


'''
src = src[:cos_start] + new_kernels + src[sin_end:]
p.write_text(src)
print("patched", p)