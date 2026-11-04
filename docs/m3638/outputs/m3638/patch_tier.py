"""Second pass: the 3-D free-surface rows belong to the ``standard`` cost tier.

``DeclaredNumber.tier`` defaults to ``fast`` (milliseconds).  A Hulme ladder is
~30 s and the CHIEF study ~45 s, comparable to the 256-panel Ursell rows that
are already ``standard``, so every new group gets an explicit tier.
"""

from pathlib import Path
import re

PATH = Path("src/tensorbem/benchmarks/declared_numbers.py")
text = PATH.read_text()

GROUPS = ("hulme", "hulme_chief", "submerged_sphere")
NEW_GROUPS = set(GROUPS)

pattern = re.compile(
    r'(DeclaredNumber\(\n(?:[^\n]*\n)*?)(\s*)(    \),)',
)


def fix(match: re.Match) -> str:
    block = match.group(0)
    if f'group="hulme"' not in block and 'group="submerged_sphere"' not in block:
        return block
    if 'tier="standard"' in block:
        return block
    return block.replace("        parameters=", "        tier=\"standard\",\n        parameters=", 1)


text, n = pattern.subn(fix, text)
PATH.write_text(text)
print("blocks scanned:", n)