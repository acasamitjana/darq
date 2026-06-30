"""Main command dispatcher for DARQ."""

import sys


def main() -> None:
    """Dispatch DARQ commands.

    Available modes:
    - darq app          -> launch the Gradio app
    - darq --dat ...    -> run the DaTSCAN pipeline
    """

    if len(sys.argv) > 1 and sys.argv[1] == "app":
        from app_darq import main as app_main

        app_main()
        return

    from scripts.dat2mri import main as pipeline_main

    pipeline_main()