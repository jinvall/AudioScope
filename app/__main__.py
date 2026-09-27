"""``python -m app`` - entry point dispatch.

With no sub-command this is the single entry point: it brings up capture
alongside the review window, so events appear as they are detected.

    python -m app                     # launch the system
    python -m app.capture             # live capture, ring buffer, recording
    python -m app.test input.wav      # same pipeline from a file
    python -m app.gui                 # review window only
    python -m app.events              # list and inspect stored events
    python -m app.separate ...        # source separation (Phase 4)
"""

import sys

USAGE = """Audio Microscope

  python -m app                        launch the system: capture + review window

  python -m app.capture [options]      live capture -> analysis -> detection
  python -m app.test input.wav         run the pipeline over a WAV file
  python -m app.analyse input.wav      analysis features and noise floor
  python -m app.events [options]       list and inspect stored events
  python -m app.gui [options]          event review workstation only
  python -m app.separate in.wav ...    query-based source separation

Try:
  python -m app --source network       capture from the phone, then review
  python -m app --no-capture           just open the review window
  python -m app --capture-only         run capture without the review window
  python -m app.capture --list-devices
  python -m app.capture --source network --port 8190
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv:
        # No sub-command: bring the whole system up.  This is the single entry
        # point the desktop launcher uses.
        from .launch import main as launch_main

        return launch_main([])

    command = argv[0]

    if command in ("-h", "--help", "help"):
        print(USAGE)
        return 0

    if command.startswith("-"):
        # Options with no sub-command belong to the launcher, which owns them.
        # Previously they were rejected as an unknown command, which made
        # `launch.sh --source network` silently do nothing.
        from .launch import main as launch_main

        return launch_main(argv)

    if command == "capture":
        from .capture import main as capture_main

        return capture_main(argv[1:])
    if command == "test":
        from .test import main as test_main

        return test_main(argv[1:])
    if command in ("analyse", "analyze"):
        from .analyse import main as analyse_main

        return analyse_main(argv[1:])
    if command == "events":
        # `app.events` is a package, so it has its own __main__.
        from .events.cli import main as events_main

        return events_main(argv[1:])
    if command in ("gui", "review"):
        # The GUI imports PyQt5, which is optional; import it only when asked
        # so the headless commands work without it.
        from .gui.app import main as gui_main

        return gui_main(argv[1:])
    if command == "separate":
        from .separate import main as separate_main

        return separate_main(argv[1:])

    print(f"unknown command: {command}\n", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
