"""Write complete benchmark tables even when another run is interrupted."""

import csv
from pathlib import Path
from tempfile import NamedTemporaryFile


def write(path, fields, rows):
    with NamedTemporaryFile(
        mode="w", newline="", dir=path.parent, delete=False
    ) as stream:
        tmp = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            stream.close()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
