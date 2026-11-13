"""context_craft — task brief packages + three-layer memory compression.

Domain-agnostic curation toolkit (see docs/context_craft_protocol.md).

    from tools.context_craft import pack, extract_corpus
    items = extract_corpus(".", ["docs/*.md"])
    pkg = pack(items, "appendix tolerance for the 3% criterion", 800, "verification")
    print(pkg.to_markdown())
"""

from .ab import evaluate, run, to_markdown  # noqa: F401
from .extract import classify, extract_corpus, extract_file, iter_markdown  # noqa: F401
from .memory import (  # noqa: F401
    DEFAULT_LAYERS,
    CompressResult,
    compress,
    load_memory,
    save_memory,
    should_compress,
    touch,
    verify_forget_log,
)
from .model import (  # noqa: F401
    FLOOR_KINDS,
    KIND_WEIGHT,
    KINDS,
    BriefPackage,
    FactItem,
    MemoryEntry,
    dedupe,
    estimate_tokens,
    sha256_text,
)
from .pack import (  # noqa: F401
    PROFILES,
    TaskProfile,
    get_profile,
    pack,
    pack_two_stream,
    rank,
    tokenize,
)

__all__ = [
    "pack", "pack_two_stream", "rank", "tokenize", "extract_corpus", "extract_file", "classify",
    "iter_markdown", "FactItem", "BriefPackage", "MemoryEntry", "dedupe",
    "estimate_tokens", "sha256_text", "KINDS", "KIND_WEIGHT", "FLOOR_KINDS",
    "PROFILES", "TaskProfile", "get_profile", "compress", "should_compress",
    "load_memory", "save_memory", "verify_forget_log", "touch", "DEFAULT_LAYERS",
    "CompressResult", "run", "evaluate", "to_markdown",
]