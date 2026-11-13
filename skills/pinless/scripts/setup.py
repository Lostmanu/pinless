"""Get Pinless ready for the skill: its own Python environment, installed from the copy of the repo this skill came
with, and say where the command is.

    python setup.py [--home DIR] [--upgrade]

The environment lives in $PINLESS_HOME, else in --home (the plugin's data folder, which Claude Code keeps across
updates), else in ~/.pinless; nothing is installed into the user's own Python. It is installed again only when the
code it came from changes, and checked every time. When ready it prints, one per line:

    PINLESS=<the pinless command>   PYTHON=<the environment's python>   VERSION=pinless x.y.z
    STYLES=<docs/styles.jpg, the twelve styles side by side>
    DEMO_YEAR, DEMO_ATLAS, DEMO_TWO, DEMO_TRAIL=<demo exports of a synthetic runner, all of 2025>

Exit codes: 0 ready. 2 this Python cannot do it (too old, no venv module or ensurepip, or the map libraries do not
install on it): try another Python, last of all `uv run --python-preference only-managed --python 3.12 --no-project`.
3 the download failed (no internet, a proxy, GitHub refusing it) or another install is still running: the message
says which, and trying again later may work.

Written for any Python 3: an old one must reach the version check and exit 2, not fail to compile.
"""
import argparse
import contextlib
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MIN = (3, 10)
GITHUB = "https://github.com/Lostmanu/pinless/archive/refs/heads/main.zip"
STAMP = ".pinless-source"
HERE = Path(__file__).resolve()
WAIT_S = 900  # how long a second session waits for an install in progress before giving up
NETWORK = ("ConnectionError", "NewConnectionError", "Max retries", "Temporary failure in name resolution",
           "Name or service not known", "getaddrinfo failed", "ProxyError", "SSLError", "timed out", "Read timed out",
           "Could not fetch URL", "Network is unreachable", "No route to host")
REFUSED = ("404", "403", "Not Found", "Forbidden")
DEMOS = {"DEMO_YEAR": "strava_export_madrid_demo.zip", "DEMO_ATLAS": "strava_export_atlas_demo.zip",
         "DEMO_TWO": "strava_export_madrid_partner_demo.zip", "DEMO_TRAIL": "hike_demo.gpx"}


def source_root(here=HERE):
    """The repo this skill came with (skills/pinless/scripts/setup.py sits three folders below it), or None when the
    skill was copied on its own: then Pinless comes from GitHub."""
    root = here.parents[3] if len(here.parents) > 3 else None
    py = root / "pyproject.toml" if root else None
    if py and py.is_file() and 'name = "pinless"' in py.read_text(encoding="utf-8", errors="replace"):
        return root
    return None


def home(arg=None, env=None):
    """$PINLESS_HOME, else --home, else ~/.pinless. A --home still written as ${...} (the skill read by an agent that
    does not fill in Claude Code's paths) counts as none."""
    env = os.environ if env is None else env
    if env.get("PINLESS_HOME"):
        return Path(env["PINLESS_HOME"]).expanduser()
    if arg and "${" not in arg:
        return Path(arg).expanduser()
    return Path.home() / ".pinless"


def fingerprint(root):
    """A hash of what gets installed (pyproject.toml and the pinless package): a new version of the plugin with the
    same code installs nothing again."""
    h = hashlib.sha256()
    files = [root / "pyproject.toml"] + sorted((root / "pinless").rglob("*"))
    for f in files:
        if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
            h.update(f.relative_to(root).as_posix().encode())
            h.update(b"\0")
            h.update(f.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def wanted(root, today=None):
    """What the environment must hold: this copy's fingerprint, or, for a skill copied alone, GitHub's main branch of
    this week (so it is fetched again once a week, not on every poster)."""
    return fingerprint(root) if root else "github " + time.strftime("%G-W%V", today or time.localtime())


def command(env_dir, name, nt=None):
    """A program inside the environment: Scripts\\name.exe on Windows, bin/name elsewhere."""
    nt = os.name == "nt" if nt is None else nt
    return Path(env_dir) / ("Scripts" if nt else "bin") / (name + (".exe" if nt else ""))


def healthy(py, pinless=False, run=subprocess.run):
    """The environment's python runs, is 3.10 or newer and has pip (an environment cut short, by a missing ensurepip
    or a timeout, leaves a python without pip); with pinless, it also imports Pinless (a system upgrade can swap the
    Python under an environment and leave its packages behind)."""
    code = "import sys, pip; assert sys.version_info[:2] >= (3, 10)" + ("; import pinless.cli" if pinless else "")
    try:
        return run([str(py), "-c", code], capture_output=True, text=True, timeout=180).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def make(env_dir):
    """A fresh environment: None when it worked, else why not. A failed one is removed whole, never left half made."""
    shutil.rmtree(str(env_dir), ignore_errors=True)
    try:
        import venv
        venv.EnvBuilder(with_pip=True).create(str(env_dir))
    except BaseException as e:  # Debian and Ubuntu without python3-venv raise or exit; Ctrl-C too
        shutil.rmtree(str(env_dir), ignore_errors=True)
        if isinstance(e, KeyboardInterrupt):
            raise
        return "{}: {}".format(type(e).__name__, e)
    return None


def build_copy(root, tmp):
    """What pip builds (pyproject.toml, README.md, the pinless package), copied to a temporary folder: pip builds in
    the folder it is given, and the plugin's own folder is shared by every session."""
    dst = Path(tmp) / "pinless-src"
    dst.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        if (root / name).is_file():
            shutil.copy2(str(root / name), str(dst / name))
    shutil.copytree(str(root / "pinless"), str(dst / "pinless"), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return dst


def why(r):
    """("network" | "refused" | "build", the last lines of pip's output)."""
    text = (r.stderr or "") + "\n" + (r.stdout or "")
    tail = text.strip().splitlines()[-12:]
    if "github" in text.lower() and any(k in text for k in REFUSED):
        return "refused", tail
    if any(k in text for k in NETWORK):
        return "network", tail
    return "build", tail


@contextlib.contextmanager
def locked(path, wait_s=WAIT_S):
    """One install at a time in a home: an OS lock, which dies with its process (a run killed by a timeout does not
    leave the next one waiting). Yields False if another install held it for wait_s."""
    f = open(str(path), "a+")
    got, said, end = False, False, time.monotonic() + wait_s
    try:
        while not got:
            try:
                if os.name == "nt":
                    import msvcrt
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                got = True
            except OSError:
                if time.monotonic() > end:
                    break
                if not said:
                    print("Another Pinless install is running; waiting for it...", file=sys.stderr, flush=True)
                    said = True
                time.sleep(2)
        yield got
    finally:
        f.close()  # closing the file releases the lock


def _pip(py, *args):
    return subprocess.run([str(py), "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "-q"]
                          + list(args), capture_output=True, text=True)


def _say(kind, tail):
    head = {"network": "Could not download Pinless or its libraries: no internet connection, or a proxy in the way.",
            "refused": "GitHub refused the download: the repository may be private, or the link has moved.",
            "build": "The map libraries Pinless needs do not install on this Python (no ready-made packages for it)."}
    print(head[kind], *tail, sep="\n  ", file=sys.stderr)
    return 2 if kind == "build" else 3


def _report(py, pinless, root, version):
    print("PINLESS={}".format(pinless))
    print("PYTHON={}".format(py))
    print("VERSION={}".format(version))
    if root and (root / "docs" / "styles.jpg").is_file():
        print("STYLES={}".format(root / "docs" / "styles.jpg"))
    for key, name in DEMOS.items():
        if root and (root / "demo" / name).is_file():
            print("{}={}".format(key, root / "demo" / name))
    return 0


def _version(pinless):
    try:
        r = subprocess.run([str(pinless), "--version"], capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def ready(env_dir, root, want, upgrade=False, again=True):
    """Make the environment hold `want` and work; the exit code."""
    py, pinless, stamp = command(env_dir, "python"), command(env_dir, "pinless"), env_dir / STAMP
    have = stamp.read_text(encoding="utf-8").strip() if stamp.is_file() else None
    if not healthy(py):
        trouble = make(env_dir)
        if trouble:
            print("This Python cannot make an environment ({}). On Debian or Ubuntu: sudo apt install python3-venv. "
                  "Or use uv, which brings its own Python.".format(trouble), file=sys.stderr)
            return 2
        have = None
    if have != want or upgrade or not pinless.is_file():
        print("Updating Pinless..." if have else
              "Installing Pinless and the map libraries it uses (a few hundred MB, only the first time)...",
              file=sys.stderr, flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            src = str(build_copy(root, tmp)) if root else GITHUB
            r = _pip(py, "--upgrade", src)
            if r.returncode == 0 and have is not None:  # same version number, new code: put this code in place
                r = _pip(py, "--force-reinstall", "--no-deps", src)
        if r.returncode:
            kind, tail = why(r)
            old = _version(pinless) if have is not None else None
            if old:  # the update failed but the copy before it works: keep using it, try again next time
                print("Could not update Pinless; using the version installed before.", *tail, sep="\n  ",
                      file=sys.stderr)
                return _report(py, pinless, root, old)
            return _say(kind, tail)
    version = _version(pinless)
    if version is None:
        if again:  # the Python under the environment changed, or a broken install: once more, from scratch
            shutil.rmtree(str(env_dir), ignore_errors=True)
            return ready(env_dir, root, want, upgrade, again=False)
        print("Pinless was installed but does not run on this Python.", file=sys.stderr)
        return 2
    stamp.write_text(want, encoding="utf-8")  # only once it runs: a broken install is never sealed as done
    return _report(py, pinless, root, version)


def main(argv=None):
    if sys.version_info < MIN:
        sys.stderr.write("Pinless needs Python 3.10 or newer; this one is {}.\n".format(platform.python_version()))
        return 2
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Install Pinless in its own environment for the skill.")
    p.add_argument("--home", default=None, help="folder for the environment (default ~/.pinless)")
    p.add_argument("--upgrade", action="store_true", help="install again even if nothing changed")
    a = p.parse_args(argv)
    base = home(a.home)
    base.mkdir(parents=True, exist_ok=True)
    root = source_root()
    with locked(base / "setup.lock") as got:
        if not got:
            print("Another Pinless install is still running. Try again in a few minutes.", file=sys.stderr)
            return 3
        return ready(base / "venv", root, wanted(root), a.upgrade)


if __name__ == "__main__":
    raise SystemExit(main())
