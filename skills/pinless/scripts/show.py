"""Open a file (the poster's preview, the sample of styles, the output folder) in the computer's own viewer.

    python show.py <file or folder>

Prints OPENED=<path> when the system took it, or SHOW=<path> when nothing could show it (no desktop, as over SSH or
on a server): then tell the user where the file is.
"""
import os
import platform as _platform
import shutil
import subprocess
import sys
from pathlib import Path


def opener(platform=None, env=None, which=shutil.which, release=None):
    """The command that opens a file with its default app: ["startfile"] on Windows (os.startfile), ["open"] on
    macOS, ["wslview"] or ["explorer.exe"] on WSL, ["xdg-open"] on a Linux desktop, and None where there is no
    screen to show it on (no DISPLAY or WAYLAND_DISPLAY), or no opener."""
    platform = sys.platform if platform is None else platform
    env = os.environ if env is None else env
    release = _platform.release() if release is None else release
    if platform.startswith("win"):
        return ["startfile"]
    if platform == "darwin":
        return ["open"]
    if "microsoft" in release.lower():  # WSL: the Windows desktop shows it
        for prog in ("wslview", "explorer.exe"):
            if which(prog):
                return [prog]
        return None
    if not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
        return None
    return ["xdg-open"] if which("xdg-open") else None


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) != 1:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    path = Path(args[0]).expanduser().resolve()
    if not path.exists():
        print("No such file: {}".format(path), file=sys.stderr)
        return 1
    cmd = opener()
    shown = False
    try:
        if cmd == ["startfile"]:
            os.startfile(str(path))  # noqa: S606 - the user's own file, in their own viewer
            shown = True
        elif cmd:
            target = str(path)
            if cmd == ["explorer.exe"]:
                target = subprocess.run(["wslpath", "-w", target], capture_output=True, text=True).stdout.strip()
            proc = subprocess.Popen(cmd + [target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    start_new_session=True)
            try:  # an opener that cannot reach a screen fails within a moment; one still running is the viewer
                shown = proc.wait(timeout=5) == 0 or cmd == ["explorer.exe"]  # explorer.exe answers 1 when it works
            except subprocess.TimeoutExpired:
                shown = True
    except OSError:
        shown = False
    print("{}={}".format("OPENED" if shown else "SHOW", path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
