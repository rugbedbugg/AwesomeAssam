"""Exit codes shared by the ``scripts/`` entry points.

The codes are part of the CLI contract and are documented in each script's
``--help`` epilog. ``EXIT_USAGE`` matches argparse's own usage-error exit code.
"""

from typing import Final

EXIT_OK: Final = 0
EXIT_FAILURE: Final = 1
EXIT_USAGE: Final = 2
EXIT_MISSING_DEPENDENCY: Final = 3
EXIT_MODEL_UNAVAILABLE: Final = 4
