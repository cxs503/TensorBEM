from pathlib import Path

path = Path("src/tensorbem/benchmarks/case_cylinder_stream.py")
text = path.read_text()

# pull the azimuth attachment out of near_wall_study so the staged runner can
# re-attach it without duplicating the cross-check
start = text.index('    azimuth = near_wall_azimuth_study(radius=radius, speed=speed, ladder=ladder)')
end = text.index('    return {\n        "azimuth": azimuth,', start)
block = text[start:end]
text = text[:start] + '''    azimuth = attach_azimuth_study(window, radius, speed, ladder)
''' + text[end:]

helper = '''

def attach_azimuth_study(
    window: Sequence[dict[str, Any]],
    radius: float = RADIUS,
    speed: float = SPEED,
    ladder: Sequence[int] = NEAR_WALL_LADDER,
) -> dict[str, Any]:
    """Measure the azimuthal sweep and cross-check it against the radial ladder.

    ``window`` is the ``near_wall["window"]["levels"]`` list: the same midpoint
    direction measured from the circle instead of from the chord, which is what
    makes the two studies comparable.
    """
    azimuth = near_wall_azimuth_study(radius=radius, speed=speed, ladder=ladder)
    frame = []
    for index, level in enumerate(azimuth["levels"]):
        row = next(
            row
            for row in level["sweep"]
            if row["clearance_over_h_max"] == NEAR_WALL_WINDOW_FACTOR
        )
        midpoint = float(row["midpoint_error"])
        circle = float(window[index]["max_relative_velocity_error"])
        frame.append(
            {
                "panels": level["panels"],
                "midpoint_error_from_chord": midpoint,
                "midpoint_error_from_circle": circle,
                "relative_difference": (
                    abs(midpoint - circle) / circle if circle > 0.0 else float("inf")
                ),
            }
        )
    azimuth["midpoint_cross_check"] = {
        "levels": frame,
        "max_relative_difference": max(item["relative_difference"] for item in frame),
        "tolerance": NEAR_WALL_AZIMUTH_FRAME_TOL,
        "note": (
            "the radial ladder probes at radius a + delta (measured from the "
            "circle) and the azimuthal sweep at chord midpoint + delta (measured "
            "from the polygon), so the same midpoint direction differs by the "
            "faceting offset; agreement within NEAR_WALL_AZIMUTH_FRAME_TOL "
            "confirms both studies measure the same regime"
        ),
    }
    return azimuth

'''
anchor = "\ndef near_wall_study(\n"
assert text.count(anchor) == 1
text = text.replace(anchor, helper + "\ndef near_wall_study(\n", 1)
path.write_text(text)
print("refactor ok")
