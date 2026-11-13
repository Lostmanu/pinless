"""The Claude Code plugin (.claude-plugin/) and its skill (skills/pinless/): the manifests name the skill that ships
here, every option and style the skill tells Claude to use is one the command has, and the setup and viewer scripts
pick the right paths and programs. None of these tests installs anything or opens a window."""
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
from PIL import Image

from pinless import cli, colouring, render, styles

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "pinless"
TEXTS = {p.name: p.read_text(encoding="utf-8") for p in (SKILL / "SKILL.md", SKILL / "reference.md")}
SETUP_FLAGS = {"--home", "--upgrade"}  # setup.py's own
UV_FLAGS = {"--python", "--no-project", "--python-preference"}  # the fallback: uv brings its own Python


def _load(name):
    spec = importlib.util.spec_from_file_location(f"skill_{name}", SKILL / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_plugin_and_its_marketplace_name_the_skill_that_ships_here():
    plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
    assert plugin["name"] == "pinless" and market["name"] == "pinless", "the install line is pinless@pinless"
    assert [(p["name"], p["source"]) for p in market["plugins"]] == [("pinless", "./")], "the repo is the plugin"
    assert "version" not in plugin, "no fixed version: every commit reaches `claude plugin update`"
    head = re.match(r"---\n(.*?)\n---\n", TEXTS["SKILL.md"], re.S).group(1)
    meta = dict(line.split(": ", 1) for line in head.splitlines())
    assert meta["name"] == "pinless" and 200 < len(meta["description"]) < 1536, "Claude Code cuts it at 1,536"
    assert len(TEXTS["SKILL.md"].splitlines()) < 500 and "reference.md" in TEXTS["SKILL.md"]
    icon = ROOT / ".claude-plugin" / "icon.png"
    with Image.open(icon) as im:
        assert im.size[0] == im.size[1] and 512 <= im.size[0] <= 2048 and icon.stat().st_size < 2_000_000


def test_every_option_the_skill_names_is_one_the_command_has():
    """The skill is Claude's only manual: a flag renamed in the command and not in the skill would make every
    poster fail in front of the user, and a style dropped from the table would never be offered."""
    p = cli.parser()
    options = {s for act in p._actions for s in act.option_strings}
    choices = {s: set(act.choices) for act in p._actions if act.choices for s in act.option_strings}
    for name, text in TEXTS.items():
        named = set(re.findall(r"(?<![\w-])--[a-z][a-z-]+", text))
        assert named - options - SETUP_FLAGS - UV_FLAGS == set(), f"{name} names options pinless does not have"
        for flag, values in re.findall(r"(--[a-z-]+) ([\w]+(?:\|[\w]+)+|[a-z]\w*)", text):
            if flag in choices:
                assert set(values.split("|")) <= choices[flag], f"{name}: {flag} {values}"
    setup = _load("setup")
    assert SETUP_FLAGS <= {s for s in re.findall(r'add_argument\("(--[a-z-]+)"', (SKILL / "scripts" / "setup.py")
                                                 .read_text(encoding="utf-8"))}
    assert setup.MIN == (3, 10)
    rows = re.findall(r"^\| `(\w+)` \|(.*)$", TEXTS["reference.md"], re.M)
    assert [r[0] for r in rows] == list(styles.STYLES), "the table lists the twelve styles, in order"
    for name, row in rows:
        s = styles.STYLES[name]
        own = {s["bg"], s["ink"], s["accent"], *s["heat"]}
        assert set(re.findall(r"#[0-9a-f]{6}", row)) <= own, f"the colours in {name}'s row are its own"
    for colour in ("Huella", "Carbon", "Midnight", "Ink"):
        assert colour.lower() in styles.STYLES and f'"{colour}, ' in TEXTS["SKILL.md"]
    assert set(colouring.MODES) == {"heat", "single", "month", "pace", "elevation"}
    assert {"A3", "A4", "50x70"} == set(render.SIZES)


def test_the_command_says_its_version(capsys):
    with pytest.raises(SystemExit) as ex:
        cli.main(["--version"])
    assert ex.value.code == 0 and capsys.readouterr().out.strip() == f"pinless {cli.version()}"


def test_the_setup_script_keeps_its_environment_where_it_is_told_and_installs_only_new_code(tmp_path):
    setup = _load("setup")
    assert setup.source_root() == ROOT, "the repo the skill came with"
    alone = tmp_path / "a" / "b" / "c" / "setup.py"
    alone.parent.mkdir(parents=True)
    assert setup.source_root(alone) is None, "a skill copied on its own installs from GitHub"
    home = Path.home()
    assert setup.home("/data/pinless", {"PINLESS_HOME": str(tmp_path)}) == tmp_path, "$PINLESS_HOME first"
    assert setup.home(str(tmp_path / "data"), {}) == tmp_path / "data", "then the plugin's data folder"
    assert setup.home("${CLAUDE_PLUGIN_DATA}", {}) == home / ".pinless", "a path nobody filled in counts as none"
    assert setup.home(None, {}) == home / ".pinless"
    assert setup.command("v", "pinless", nt=True) == Path("v") / "Scripts" / "pinless.exe"
    assert setup.command("v", "python", nt=False) == Path("v") / "bin" / "python"
    repo = tmp_path / "repo"
    (repo / "pinless" / "__pycache__").mkdir(parents=True)
    (repo / "pyproject.toml").write_text('name = "pinless"\n', encoding="utf-8")
    (repo / "pinless" / "cli.py").write_text("x = 1\n", encoding="utf-8")
    first = setup.fingerprint(repo)
    (repo / "pinless" / "__pycache__" / "cli.cpython-313.pyc").write_bytes(b"\0")
    assert setup.fingerprint(repo) == first, "compiled files are not new code"
    (repo / "pinless" / "cli.py").write_text("x = 2\n", encoding="utf-8")
    assert setup.fingerprint(repo) != first, "new code installs again"


    assert setup.wanted(None, time.strptime("2026-11-13", "%Y-%m-%d")) == "github 2026-W46", "GitHub: once a week"
    assert setup.wanted(repo) == setup.fingerprint(repo)


class _Env:
    """A fake environment for setup.ready: which steps ran, and what each answers."""

    def __init__(self, setup, monkeypatch, tmp_path, healthy=True, make=None, pip=(0,), version="pinless 0.1.0"):
        self.calls, self.pip_codes, self.dir = [], list(pip), tmp_path / "venv"
        self.dir.mkdir(exist_ok=True)
        monkeypatch.setattr(setup, "healthy", lambda py, pinless=False: self.calls.append("healthy") or healthy)
        monkeypatch.setattr(setup, "make", lambda d: self.calls.append("make") or make)
        monkeypatch.setattr(setup, "build_copy", lambda root, tmp: Path(tmp))
        monkeypatch.setattr(setup, "_pip", self._pip)
        versions = list(version) if isinstance(version, list) else [version]
        monkeypatch.setattr(setup, "_version", lambda p: self.calls.append("version") or
                            (versions.pop(0) if len(versions) > 1 else versions[0]))
        (setup.command(self.dir, "pinless")).parent.mkdir(parents=True, exist_ok=True)
        setup.command(self.dir, "pinless").write_bytes(b"")

    def _pip(self, py, *args):
        self.calls.append("pip " + ("force" if "--force-reinstall" in args else "upgrade"))
        code = self.pip_codes.pop(0) if self.pip_codes else 0
        return type("R", (), {"returncode": code, "stdout": "", "stderr": "" if code == 0 else
                              "ERROR: Could not fetch URL https://pypi.org: Max retries exceeded"})()

    def stamp(self, setup):
        f = self.dir / setup.STAMP
        return f.read_text(encoding="utf-8") if f.is_file() else None


def test_the_setup_script_installs_new_code_once_and_never_seals_a_broken_install(monkeypatch, tmp_path, capsys):
    """setup.ready, with pip, venv and pinless faked: nothing to do when nothing changed; new code installed and put
    in place; a failed update keeps the version before; an install that does not run is rebuilt once, then refused,
    and never stamped as done; an environment that cannot be made is not left half made."""
    setup = _load("setup")
    env = _Env(setup, monkeypatch, tmp_path)
    (env.dir / setup.STAMP).write_text("abc", encoding="utf-8")
    assert setup.ready(env.dir, None, "abc") == 0 and not [c for c in env.calls if c.startswith("pip")]
    out = capsys.readouterr().out
    assert f"PINLESS={setup.command(env.dir, 'pinless')}" in out and "VERSION=pinless 0.1.0" in out
    env.calls.clear()
    assert setup.ready(env.dir, None, "new") == 0 and env.calls.count("pip upgrade") == env.calls.count("pip force") == 1
    assert env.stamp(setup) == "new", "new code, installed and put in place, then stamped"

    env = _Env(setup, monkeypatch, tmp_path, pip=(1,))
    (env.dir / setup.STAMP).write_text("old", encoding="utf-8")
    assert setup.ready(env.dir, None, "newer") == 0, "no network for an update: the version before still works"
    assert env.stamp(setup) == "old" and "using the version installed before" in capsys.readouterr().err

    env = _Env(setup, monkeypatch, tmp_path, version=None)
    (env.dir / setup.STAMP).unlink()
    assert setup.ready(env.dir, None, "abc") == 2 and env.stamp(setup) is None, "a broken install is never sealed"
    assert env.calls.count("pip upgrade") == 2, "rebuilt from scratch once before giving up"
    capsys.readouterr()

    env = _Env(setup, monkeypatch, tmp_path, healthy=False, make="CalledProcessError: ensurepip")
    assert setup.ready(env.dir, None, "abc") == 2 and env.calls[:2] == ["healthy", "make"]
    assert "sudo apt install python3-venv" in capsys.readouterr().err

    env = _Env(setup, monkeypatch, tmp_path, healthy=False, pip=(1,))
    assert setup.ready(env.dir, None, "abc") == 3 and env.stamp(setup) is None, "no network on a first install"
    assert "no internet connection" in capsys.readouterr().err


def test_a_half_made_environment_is_removed_and_a_python_without_pip_is_not_healthy(monkeypatch, tmp_path):
    """Ubuntu without python3-venv: venv writes bin/python, then ensurepip fails. The folder must go, or every later
    run would find a python, skip the rebuild and fail on a missing pip."""
    setup = _load("setup")
    import venv

    class Half(venv.EnvBuilder):
        def create(self, env_dir):
            (Path(env_dir) / "bin").mkdir(parents=True)
            (Path(env_dir) / "bin" / "python").write_bytes(b"")
            raise subprocess.CalledProcessError(1, ["python", "-m", "ensurepip"])

    monkeypatch.setattr(venv, "EnvBuilder", Half)
    env_dir = tmp_path / "venv"
    assert "CalledProcessError" in setup.make(env_dir) and not env_dir.exists()
    seen = []

    def run(cmd, **kw):
        seen.append(cmd[-1])
        return type("R", (), {"returncode": 1})()

    assert setup.healthy("python", run=run) is False and "import sys, pip" in seen[0]
    assert setup.healthy("python", pinless=True, run=run) is False and "import pinless.cli" in seen[1]


def test_the_setup_script_tells_a_refused_download_from_no_network_and_from_a_python_it_cannot_use(tmp_path):
    setup = _load("setup")

    def r(err):
        return type("R", (), {"stderr": err, "stdout": ""})()

    assert setup.why(r("HTTP error 404 while getting https://github.com/Lostmanu/pinless/archive/main.zip"))[0] == \
        "refused"
    assert setup.why(r("Failed to establish a new connection: [Errno 11001] getaddrinfo failed"))[0] == "network"
    assert setup.why(r("error: subprocess-exited-with-error\nFailed building wheel for pyproj"))[0] == "build"
    lock = tmp_path / "setup.lock"
    with setup.locked(lock) as first:
        with setup.locked(lock, wait_s=0) as second:
            assert first and not second, "one install at a time"
    with setup.locked(lock, wait_s=0) as again:
        assert again, "the lock goes with its holder"


def test_the_viewer_script_opens_where_there_is_a_screen_and_says_where_the_file_is_where_there_is_none(
        tmp_path, capsys, monkeypatch):
    show = _load("show")
    every = (lambda prog: "/usr/bin/" + prog)
    assert show.opener("win32") == ["startfile"] and show.opener("darwin", {}, every, "") == ["open"]
    assert show.opener("linux", {"DISPLAY": ":0"}, every, "6.5.0") == ["xdg-open"]
    assert show.opener("linux", {}, every, "6.5.0") is None, "over SSH or on a server: no screen"
    assert show.opener("linux", {"WAYLAND_DISPLAY": "wayland-0"}, lambda p: None, "6.5.0") is None, "no xdg-open"
    assert show.opener("linux", {}, every, "5.15.153.1-microsoft-standard-WSL2") == ["wslview"]
    assert show.opener("linux", {}, lambda p: p if p == "explorer.exe" else None, "6.6-microsoft") == ["explorer.exe"]
    f = tmp_path / "poster.jpg"
    f.write_bytes(b"jpg")
    monkeypatch.setattr(show, "opener", lambda: None)
    assert show.main([str(f)]) == 0 and capsys.readouterr().out.strip() == f"SHOW={f.resolve()}"

    class Fails:
        def __init__(self, *a, **k):
            pass

        def wait(self, timeout):
            return 3  # xdg-open with no way to show it

    monkeypatch.setattr(show, "opener", lambda: ["xdg-open"])
    monkeypatch.setattr(show.subprocess, "Popen", Fails)
    assert show.main([str(f)]) == 0 and capsys.readouterr().out.strip() == f"SHOW={f.resolve()}"
    assert show.main([str(tmp_path / "nope.jpg")]) == 1 and show.main([]) == 2
    capsys.readouterr()


def test_output_paths_may_start_with_the_home_folder_in_quotes(monkeypatch, tmp_path):
    """`--out "~/Pinless/x.png"`: no shell expands ~ inside quotes, and the folder may not exist yet. Pinless expands
    it and makes the folder, before minutes of downloading and drawing rather than failing after them."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    seen = {}

    def stop(path):
        seen["export"] = path
        raise SystemExit(9)

    monkeypatch.setattr(cli.ingest, "read_export", stop)
    with pytest.raises(SystemExit):
        cli.main(["~/exports/x.zip", "--out", "~/Pinless/a.png", "--pdf", "~/Pinless/a.pdf"])
    assert Path(seen["export"]) == tmp_path / "exports" / "x.zip" and (tmp_path / "Pinless").is_dir()
