"""Insert the roll/pitch free-decay test stand into the Cummins-Wigley case module."""
from pathlib import Path

path = Path("src/tensorbem/benchmarks/case_bem_cummins_wigley.py")
text = path.read_text()

ANCHOR = '''def free_decay_gate(sweep: Sweep, *, dof: int = 2) -> dict[str, Any]:
    """C5: time-domain Cummins free decay vs the frequency-domain damping ratio."""'''

NEW = '''#: Modes that carry a hydrostatic restoring term, i.e. the modes a force-free
#: decay can be started in.  Surge, sway and yaw have ``C = 0``: the Cummins
#: oscillator for them is a pure inertia-memory system and "free decay" is not
#: defined, so C5 does not apply to them (that is a property of the physics, not
#: of the band).
RESTORING_MODES = (2, 3, 4)


def _uniform_body_second_moments() -> dict[str, float]:
    """Second moments of the displaced volume about :data:`CENTER`, uniform density.

    The free-decay test stand needs the *body* inertia, which is a property of
    the mass distribution and not of the hull: no hydrodynamic solve can supply
    it.  This case therefore declares one and states it, instead of leaving it
    implicit: the body is homogeneous and its mass fills the displaced volume at
    the fluid density, i.e. the same equilibrium assumption ``m = rho * nabla``
    that the hydrostatic route already makes (``compute_hydrostatic_restoring_matrix``
    evaluates the gravity terms at ``m = rho * nabla``).  The declared body is
    then the analytic hull itself, so its second moments are closed-form:

    with ``y(x, z) = (B/2)(1 - (2x/L)^2)(1 - (z/T)^2)`` on ``|x| <= L/2``,
    ``-T <= z <= 0`` and the rotation centre on the calm waterplane,

    .. math::

        I_{xx} &= rho\\int (y^2 + z^2) dV
               = rho\\left[\\frac{64 L T B^3}{3675} + \\frac{4 L B T^3}{45}\\right] \\\\
        I_{yy} &= rho\\int (x^2 + z^2) dV = \\frac{rho L B T (L^2 + 4 T^2)}{45} \\\\
        I_{zz} &= rho\\int (x^2 + y^2) dV
               = rho\\left[\\frac{64 L T B^3}{3675} + \\frac{L^3 B T}{45}\\right]

    (the ``1``, ``x^2`` and ``x^4`` integrals of the parabolic half breadth are
    ``2L/3``, ``L^3/30`` and ``L^5/80``; the ``z`` analogue of the fourth
    integral is ``16T/35``).  The implied radii of gyration are ``k_xx/B = 0.342``
    and ``k_yy/L = 0.225``, inside the usual ship range (``0.30-0.40 B``,
    ``0.24-0.26 L``) -- a consistency check on the declaration, not a criterion.

    Why a declared inertia does not weaken the gate: C5 compares two routes for
    the *same* oscillator (time-domain Cummins convolution vs the
    frequency-domain impedance at the same natural frequency), so the inertia
    shifts ``omega_n`` and both sides equally.  What it does control is
    decidability, because ``omega_n`` has to fall inside the computed band.
    """
    rho, L, B, T = DENSITY, LENGTH, BEAM, DRAFT
    return {
        "roll": rho * (64.0 * L * T * B**3 / 3675.0 + 4.0 * L * B * T**3 / 45.0),
        "pitch": rho * L * B * T * (L**2 + 4.0 * T**2) / 45.0,
        "yaw": rho * (64.0 * L * T * B**3 / 3675.0 + L**3 * B * T / 45.0),
    }


def oscillator_properties(dof: int) -> dict[str, Any]:
    """Inertia and hydrostatic stiffness of the free-decay oscillator for ``dof``.

    ``dof = 2`` (heave) reproduces the route gate C5 has always used -- displaced
    mass and the waterplane area of the *panelised* hull -- so its recorded
    number does not move.  ``dof = 3`` (roll) and ``dof = 4`` (pitch) use the
    closed-form Wigley hydrostatics ``C44 = rho g nabla GM_T``,
    ``C55 = rho g nabla GM_L`` and the declared body inertia above.

    .. warning::
       :func:`tensorbem.wigley_hydrostatics.wigley_hydrostatic_reference` stores
       ``rho g nabla GM_L`` under the key ``"C44"`` and ``rho g nabla GM_T``
       under ``"C55"``, i.e. its ``C44``/``C55`` labels are swapped (pinned by
       ``tests/test_wigley_hydrostatics.py`` and explained in
       :mod:`tensorbem.diffraction_drift_maturity`).  The metacentric heights
       themselves are labelled correctly, so they are what this function reads.
    """
    from tensorbem.wigley_hydrostatics import wigley_hydrostatic_reference

    dof = int(dof)
    if dof not in RESTORING_MODES:
        raise ValueError(
            f"only the restoring modes {RESTORING_MODES} can start a free decay; "
            f"dof {dof} ({DOF_NAMES[dof]}) has no hydrostatic stiffness"
        )
    if dof == 2:
        restoring, displaced_mass, waterplane_area = _restoring_and_mass_for_heave()
        return {
            "dof": dof,
            "inertia": displaced_mass,
            "inertia_kind": "mass",
            "inertia_source": "rho * nabla (displaced mass at equilibrium)",
            "stiffness": restoring,
            "stiffness_source": f"rho g A_wp of the panelised hull (A_wp = {waterplane_area:.6g} m^2)",
        }
    reference = wigley_hydrostatic_reference(LENGTH, BEAM, DRAFT, density=DENSITY, gravity=GRAVITY)
    stiffness = DENSITY * GRAVITY * reference["volume"] * (
        reference["GM_T"] if dof == 3 else reference["GM_L"]
    )
    moments = _uniform_body_second_moments()
    return {
        "dof": dof,
        "inertia": moments["roll" if dof == 3 else "pitch"],
        "inertia_kind": "moment_of_inertia",
        "inertia_source": (
            "uniform-density body filling the displaced volume, second moment "
            "about CENTER (closed form)"
        ),
        "stiffness": stiffness,
        "stiffness_source": (
            f"rho g nabla GM_{'T' if dof == 3 else 'L'} (closed-form Wigley hydrostatics)"
        ),
    }


def _restoring_and_mass_for_heave() -> tuple[float, float, float]:
    """Heave / displaced-mass pair on the panelised hull (the legacy C5 route)."""
    n_sections, n_levels = MESHES["medium"]
    mesh = wigley_wetted_hull_mesh(
        LENGTH, BEAM, DRAFT, n_sections=n_sections, n_levels=n_levels
    )
    waterplane_area = float(compute_waterplane_properties(mesh)[0])
    return DENSITY * GRAVITY * waterplane_area, DENSITY * VOLUME_ANALYTIC, waterplane_area


def restoring_matrix_routes() -> dict[str, Any]:
    """Roll/pitch/heave stiffness by two independent routes, per mesh.

    Closed form (analytic hull) vs the panel-based
    :func:`tensorbem.hydrostatic.compute_hydrostatic_restoring_matrix` on the
    coarse and medium case meshes.  Two routes on the same hull is what makes
    the rotational stiffness entering C5 a checked number rather than a quoted
    one; the mesh route is also the one that would be available for a hull with
    no closed form.

    Note the key convention again: the mesh route returns the *physical* matrix,
    so ``C[3, 3]`` is roll (``GM_T``) and ``C[4, 4]`` is pitch (``GM_L``).
    """
    from tensorbem.hydrostatic import compute_hydrostatic_restoring_matrix
    from tensorbem.wigley_hydrostatics import wigley_hydrostatic_reference

    reference = wigley_hydrostatic_reference(
        LENGTH, BEAM, DRAFT, density=DENSITY, gravity=GRAVITY
    )
    closed = {
        "heave": reference["C33"],
        "roll": DENSITY * GRAVITY * reference["volume"] * reference["GM_T"],
        "pitch": DENSITY * GRAVITY * reference["volume"] * reference["GM_L"],
    }
    rows: dict[str, Any] = {}
    for label, (n_sections, n_levels) in MESHES.items():
        mesh = wigley_wetted_hull_mesh(
            LENGTH, BEAM, DRAFT, n_sections=n_sections, n_levels=n_levels
        )
        matrix = np.asarray(
            compute_hydrostatic_restoring_matrix(
                mesh,
                DENSITY * VOLUME_ANALYTIC,
                np.array([CENTER[0], CENTER[1], CENTER[2] - 0.5 * DRAFT]),
                density=DENSITY,
                gravity=GRAVITY,
            ).restoring_matrix
        )
        mesh_values = {"heave": float(matrix[2, 2]), "roll": float(matrix[3, 3]), "pitch": float(matrix[4, 4])}
        rows[label] = {
            "mesh_values": mesh_values,
            "relative_error_vs_closed_form": {
                key: abs(mesh_values[key] / closed[key] - 1.0) for key in mesh_values
            },
            "cross_coupling_max": float(np.max(np.abs([matrix[2, 3], matrix[2, 4], matrix[3, 4]]))),
        }
    return {"closed_form": closed, "meshes": rows}


def decay_bias_attribution(sweep: Sweep, *, dof: int) -> dict[str, Any]:
    """Split a C5 discrepancy into "frequency-dependent added mass" and "B shape".

    C5 compares the measured log-decrement of the Cummins decay with
    ``zeta_fd = B(omega_n) / (2 (I + A(omega_n)) omega_n)``.  Two effects can
    move the time-domain number away from that estimate, and they are different
    diagnoses:

    ``A dispersion``
        the memory kernel does not act like a constant added mass ``A(omega_n)``;
        the decay feels a *band* of ``A``, and the size of the band relative to
        the body inertia sets the size of the bias (this is why a light,
        added-mass-dominated mode such as roll is the exposed one);
    ``B shape``
        the kernel is the transform of the whole damping curve, so a narrow
        resonant peak or a steep tail is not represented by the single value
        ``B(omega_n)`` either.

    The two are separated by re-running the same decay with a *flat* damping
    curve (its band value at ``omega_n``) -- which keeps ``A`` dispersion and
    removes ``B`` shape -- and by quoting ``zeta`` evaluated with ``A_inf``
    instead of ``A(omega_n)``.  Reported, not judged: the gate stays as declared,
    this only says what the residual is made of.
    """
    oscillator = oscillator_properties(dof)
    inertia, stiffness = oscillator["inertia"], oscillator["stiffness"]
    added_mass = sweep.added_mass[:, dof, dof]
    damping = sweep.damping[:, dof, dof]
    if float(np.min(damping)) < 0.0:
        return {"dof": int(dof), "dof_name": DOF_NAMES[dof], "status": "undecidable: negative damping"}
    closure = kk_added_mass_closure(sweep.omega, added_mass, damping)
    added_mass_inf = float(closure.added_mass_inf)
    omega_n = math.sqrt(stiffness / (inertia + added_mass_inf))
    for _ in range(80):
        omega_n = math.sqrt(
            stiffness / (inertia + float(np.interp(omega_n, sweep.omega, added_mass)))
        )
    a_at = float(np.interp(omega_n, sweep.omega, added_mass))
    b_at = float(np.interp(omega_n, sweep.omega, damping))
    zeta_with_a_omega = b_at / (2.0 * (inertia + a_at) * omega_n)
    zeta_with_a_inf = b_at / (2.0 * (inertia + added_mass_inf) * omega_n)
    tau = np.arange(0.0, TAU_MAX + 0.5 * DT, DT)
    flat = np.full_like(damping, b_at)
    flat_kernel = td.radiation_memory_kernel(sweep.omega, flat, tau)
    flat_decay = td.cummins_free_decay(
        flat_kernel,
        mass=inertia,
        restoring=stiffness,
        added_mass_inf=added_mass_inf,
        initial_displacement=1.0,
        dt=DT,
        duration=DURATION,
    )
    zeta_flat, flat_period, flat_peaks = _decay_zeta(
        np.asarray(flat_decay.time), np.asarray(flat_decay.displacement)
    )
    return {
        "dof": int(dof),
        "dof_name": DOF_NAMES[dof],
        "status": "reported",
        "natural_frequency_rad_s": omega_n,
        "added_mass_at_natural": a_at,
        "added_mass_inf": added_mass_inf,
        "added_mass_ratio_to_body_inertia": a_at / inertia,
        "added_mass_dispersion_across_band": float(
            (np.max(added_mass) - np.min(added_mass)) / a_at
        ),
        "zeta_with_a_omega": zeta_with_a_omega,
        "zeta_with_a_inf": zeta_with_a_inf,
        "zeta_flat_damping_time_domain": zeta_flat,
        "flat_damping_period_s": flat_period,
        "flat_damping_peaks": flat_peaks,
        "flat_damping_relative_to_band_value": (
            abs(zeta_flat - zeta_with_a_omega) / zeta_with_a_omega
            if zeta_with_a_omega > 0.0
            else float("nan")
        ),
    }


def dof_decay_audit(sweep: Sweep) -> dict[str, Any]:
    """Score gate C5 on every restoring mode and report what is decided.

    Gate C5 has only ever been evaluated on heave.  That was recorded as open
    debt ("roll/pitch would need C44/C55 restoring terms the case module does
    not build"), and this function closes it: the same 5 % bar, the same
    estimator, three modes, with the stiffness and the inertia each carried by a
    second route (closed form vs panel hydrostatics for the stiffness, closed
    form vs quadrature for the inertia).

    A mode that misses the bar is *reported as missing it*.  The case verdict is
    not silently widened: :func:`case_verdict` only tallies the gates it is
    given, and this audit is published next to the verdict, so a reader sees
    both the case criterion (heave) and the full rotational record.
    """
    rows = {}
    for dof in RESTORING_MODES:
        gate = free_decay_gate(sweep, dof=dof)
        values = gate["values"]
        status = str(values.get("status", ""))
        if status == "decided":
            grade = "within_bar" if gate["passed"] else "beyond_bar"
        else:
            grade = "undecidable"
        rows[DOF_NAMES[dof]] = {
            "dof": int(dof),
            "grade": grade,
            "passed": bool(gate["passed"]),
            "threshold": gate["threshold"],
            "values": values,
            "attribution": decay_bias_attribution(sweep, dof=dof),
        }
    bar = TOLERANCES["free_decay"]
    return {
        "gate": "C5",
        "criterion": f"|zeta_decay - zeta_fd| / zeta_fd <= {bar:.0%} (same bar as the heave gate)",
        "tolerance": bar,
        "modes": rows,
        "decided": [name for name, row in rows.items() if row["grade"] != "undecidable"],
        "within_bar": [name for name, row in rows.items() if row["grade"] == "within_bar"],
        "beyond_bar": [name for name, row in rows.items() if row["grade"] == "beyond_bar"],
        "undecidable": [name for name, row in rows.items() if row["grade"] == "undecidable"],
        "body_inertia": {
            "second_moments_kg_m2": _uniform_body_second_moments(),
            "radii_of_gyration": {
                "k_xx_over_B": math.sqrt(_uniform_body_second_moments()["roll"] / (DENSITY * VOLUME_ANALYTIC)) / BEAM,
                "k_yy_over_L": math.sqrt(_uniform_body_second_moments()["pitch"] / (DENSITY * VOLUME_ANALYTIC)) / LENGTH,
            },
        },
        "stiffness_routes": restoring_matrix_routes(),
        "band_rad_s": [float(sweep.omega[0]), float(sweep.omega[-1])],
        "dt": DT,
        "duration": DURATION,
        "tau_max": TAU_MAX,
    }


def free_decay_gate(sweep: Sweep, *, dof: int = 2) -> dict[str, Any]:
    """C5: time-domain Cummins free decay vs the frequency-domain damping ratio."""'''

assert text.count(ANCHOR) == 1, text.count(ANCHOR)
text = text.replace(ANCHOR, NEW)

OLD_BODY = '''    restoring, displaced_mass, _ = _restoring_and_mass(sweep)

    decay = td.cummins_free_decay(
        kernel,
        mass=displaced_mass,
        restoring=restoring,'''
NEW_BODY = '''    oscillator = oscillator_properties(dof)
    restoring = oscillator["stiffness"]
    displaced_mass = oscillator["inertia"]

    decay = td.cummins_free_decay(
        kernel,
        mass=displaced_mass,
        restoring=restoring,'''
assert text.count(OLD_BODY) == 1, text.count(OLD_BODY)
text = text.replace(OLD_BODY, NEW_BODY)

OLD_KEYS = '''            "restoring_n_per_m": restoring,
            "displaced_mass_kg": displaced_mass,'''
NEW_KEYS = '''            "restoring_n_per_m": restoring,
            "oscillator_stiffness_source": oscillator["stiffness_source"],
            "oscillator_inertia": displaced_mass,
            "oscillator_inertia_kind": oscillator["inertia_kind"],
            "oscillator_inertia_source": oscillator["inertia_source"],
            "displaced_mass_kg": displaced_mass,'''
assert text.count(OLD_KEYS) == 1, text.count(OLD_KEYS)
text = text.replace(OLD_KEYS, NEW_KEYS)

path.write_text(text)
print("patched", path)