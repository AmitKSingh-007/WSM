"""Start the WebMiningSearch desktop application."""

import sys
from pathlib import Path


# The learning modules live in src/ and can also be run individually.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from app import main  # noqa: E402


if __name__ == "__main__":
    main()
