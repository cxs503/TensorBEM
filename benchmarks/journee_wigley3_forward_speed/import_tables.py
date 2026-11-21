"""Extract the original numeric measures from the author's report 0909 ZIP."""

import argparse
import csv
import hashlib
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
SHA256 = "dc930dbd62bfda3f37853f75911be74028b8d8baddbe3963643652fa3b90da59"


def extract(archive, destination):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Expected the original report 0909 ASCII archive")
    with ZipFile(archive) as z:
        for table, filename in (("08", "motions"), ("06", "loads")):
            names = (
                ["heave", "heave_deg", "pitch", "pitch_deg"]
                if table == "08"
                else ["force3", "force3_deg", "moment5", "moment5_deg"]
            )
            width = 6 if table == "08" else 5
            with (destination / (filename + ".csv")).open("w") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(
                    ["Fn", "lambda_L", "source_line", "source_group", "wave_cm", *names]
                )
                for speed in (20, 30, 40):
                    data = z.read(f"{table}-III-{speed}.exp").decode("ascii")
                    for line, text in enumerate(data.splitlines(), 1):
                        values = text.split()
                        if not values:
                            continue
                        if len(values) != 1 + 2 * width:
                            raise ValueError(
                                f"Unexpected column count at {table}/{speed}/{line}"
                            )
                        for group, offset in enumerate((1, 1 + width), 1):
                            fields = values[offset : offset + 5]
                            if all(float(v) == -1111 for v in fields):
                                continue
                            if any(float(v) == -1111 for v in fields):
                                raise ValueError(
                                    f"Partially missing group at {table}/{speed}/{line}"
                                )
                            writer.writerow(
                                [speed / 100, values[0], line, group, *fields]
                            )

        # Table 2 gives the heave-driven column (33,53), Table 3 the
        # pitch-driven column (55,35). Each oscillation level has its own
        # measured frequency; never align the two groups by row number.
        with (destination / "radiation.csv").open("w") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(
                [
                    "Fn",
                    "driven",
                    "source_line",
                    "source_group",
                    "w_nd",
                    "A_self",
                    "B_self",
                    "A_cross",
                    "B_cross",
                ]
            )
            for table, mode in [("02", 3), ("03", 5)]:
                for speed in (20, 30, 40):
                    data = z.read(f"{table}-III-{speed}.exp").decode("ascii")
                    for line, text in enumerate(data.splitlines(), 1):
                        fields = text.split()
                        if not fields:
                            continue
                        if len(fields) != 10 or "-1111" in fields:
                            raise ValueError(
                                f"Unexpected radiation record at {table}/{speed}/{line}"
                            )
                        for group, start in enumerate((0, 5), 1):
                            writer.writerow(
                                [
                                    speed / 100,
                                    mode,
                                    line,
                                    group,
                                    *fields[start : start + 5],
                                ]
                            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    extract(p.parse_args().archive, ROOT)
