"""Download GPT-2 checkpoints into ./weights (or $NANOSERVE_CACHE).

Usage: python scripts/download_weights.py gpt2 [gpt2-medium ...]
"""

from __future__ import annotations

import sys

from nanoserve.weights import download


def main() -> None:
    names = sys.argv[1:] or ["gpt2"]
    for name in names:
        print(download(name))


if __name__ == "__main__":
    main()
