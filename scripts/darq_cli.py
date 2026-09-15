"""Main command dispatcher for DARQ."""

import sys
from pathlib import Path


def _add_project_root_to_path() -> None:
    """Ensure local project files are importable when using the installed darq command."""
    project_root = Path(__file__).resolve().parents[1]

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def main() -> None:
    """Dispatch DARQ commands.

    Available modes:
    - darq app          -> launch the Gradio app
    - darq --dat ...    -> run the DaTSCAN pipeline
    """

    _add_project_root_to_path()

    if len(sys.argv) > 1 and sys.argv[1] == "app":
        from app_darq import main as app_main

        app_main()
        return

    from scripts.dat2mri import main as pipeline_main

    pipeline_main()