"""Your own style from a JSON file: loading, every check with a message that says what to change, and the CLI."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pinless import cli, ingest, render_editorial, styles
from test_cli_editorial import madrid_runs, stub_world

ROOT = Path(__file__).resolve().parents[1]


def _write(tmp_path, data, name="mine.json"):
    path = tmp_path / name
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return str(path)


def test_a_built_in_style_printed_as_a_file_loads_back_the_same(tmp_path, capsys):
    for name in ("huella", "midnight", "neon"):
        assert styles.main([name]) == 0
        loaded = styles.parse(json.loads(capsys.readouterr().out))
        built_in = render_editorial.STYLES[name]
        assert {k: v for k, v in loaded.items()} == {k: built_in[k] for k in loaded}
        assert loaded["street"] == built_in["street"] and loaded.get("who") == built_in.get("who")
    assert styles.main(["nope"]) == 2


def test_a_base_style_gives_everything_not_listed():
    before = render_editorial.STYLES["huella"]["street"]["major"]
    S = styles.parse({"base": "huella", "accent": "tomato", "street": {"major": ["#000000", 0.5]}})
    huella = render_editorial.STYLES["huella"]
    assert S["accent"] == "#ff6347" and S["bg"] == huella["bg"] and S["who"] == huella["who"]
    assert S["street"]["major"] == ("#000000", 0.5) and S["street"]["minor"] == huella["street"]["minor"]
    assert huella["street"]["major"] == before, "the built-in style is not changed"


@pytest.mark.parametrize("data,message", [
    ({"bg": "#ffffff"}, "missing ink, sub, accent"),
    ({"base": "huella", "colour": "#fff"}, "unknown key 'colour'"),
    ({"base": "papyrus"}, "base 'papyrus' is not a built-in style"),
    ({"base": "huella", "bg": "#ggg"}, "bg: '#ggg' is not a colour"),
    ({"base": "huella", "bg": 12}, "bg: 12 is not a colour"),
    ({"base": "huella", "heat": ["#000"] * 4}, "heat: needs a list of 5 colours"),
    ({"base": "huella", "heat": ["#000"] * 4 + ["blurple"]}, "heat[4]: 'blurple' is not a colour"),
    ({"base": "huella", "street": {"major": ["#000", 3]}}, "street.major: the width must be"),
    ({"base": "huella", "street": {"major": "#000"}}, 'street.major: ["#rrggbb", width in points]'),
    ({"base": "huella", "street": {"motorway": ["#000", 0.3]}}, 'street: an object with "major"'),
    ({"base": "huella", "glow": 0.9}, "glow: a number from 0 (none) to 0.5"),
    ({"base": "huella", "glow": True}, "glow: a number from 0 (none) to 0.5"),
    ({"base": "huella", "who": ["#000"] * 3}, "who: needs a list of 5 colours"),
    ([1, 2], "must hold one JSON object"),
])
def test_every_mistake_is_named(data, message):
    with pytest.raises(styles.StyleError, match=None) as ex:
        styles.parse(data, "mine")
    assert message in str(ex.value)


def test_a_full_style_without_base_needs_every_street_kind():
    full = {k: v for k, v in json.loads(json.dumps({**render_editorial.STYLES["ink"], "street": {
        k: list(v) for k, v in render_editorial.STYLES["ink"]["street"].items()}})).items() if k in styles.REQUIRED}
    assert styles.parse(full)["bg"] == render_editorial.STYLES["ink"]["bg"]
    full["street"] = {"major": ["#000000", 0.3]}
    with pytest.raises(styles.StyleError, match="needs \"major\", \"minor\" and \"path\""):
        styles.parse(full)


def test_files_that_are_not_json_say_where(tmp_path):
    with pytest.raises(styles.StyleError, match="line 2, column"):
        styles.load(_write(tmp_path, '{"base": "huella",\n "bg": }'))
    with pytest.raises(styles.StyleError, match="no file"):
        styles.load(str(tmp_path / "missing.json"))
    name, S = styles.load(_write(tmp_path, '﻿{"base": "carbon"}', "with-bom.json"))  # Notepad's BOM is fine
    assert name == "with-bom" and S["bg"] == render_editorial.STYLES["carbon"]["bg"]


def test_hard_to_read_colours_are_warned_about():
    assert styles.warnings(render_editorial.STYLES["huella"]) == []
    S = styles.parse({"base": "huella", "ink": "#eeeae0", "heat": ["#f2ede3"] * 5})
    notes = " ".join(styles.warnings(S))
    assert "ink #eeeae0 is hard to read" in notes and "heat[0] #f2ede3 almost vanishes" in notes


def test_the_example_style_in_the_repo_is_clean():
    for path in sorted((ROOT / "styles").glob("*.json")):
        name, S = styles.load(str(path))
        assert styles.warnings(S) == [], name


def test_the_cli_draws_with_a_style_file(monkeypatch, tmp_path, capsys):
    acts = madrid_runs()
    stub_world(monkeypatch, lambda path: (acts, ingest.IngestReport(files_seen=48, used=48)))
    seen, real = {}, render_editorial.render
    monkeypatch.setattr(render_editorial, "render", lambda *a, **kw: seen.update(kw) or real(*a, **kw))
    path = _write(tmp_path, {"base": "midnight", "accent": "#ff8800"}, "night.json")
    out = tmp_path / "p.png"
    assert cli.main(["x.zip", "--dpi", "20", "--style", path, "--out", str(out)]) == 0
    assert seen["style"] == "file:night" and render_editorial.STYLES["file:night"]["accent"] == "#ff8800"
    assert out.exists()
    faint = _write(tmp_path, {"base": "huella", "ink": "#eeeae0"}, "faint.json")
    assert cli.main(["x.zip", "--dpi", "20", "--style", faint, "--out", str(out)]) == 0
    assert "[style] ink #eeeae0 is hard to read" in capsys.readouterr().out


@pytest.mark.parametrize("extra,message", [
    (["--layout", "classic"], "a style file works with the editorial, atlas and race layouts"),
    (["--layout", "sport"], "a style file works with the editorial, atlas and race layouts"),
    ([], "--style: mine: unknown key 'colour'"),
])
def test_the_cli_refuses_a_style_file_it_cannot_use(extra, message, tmp_path, capsys):
    path = _write(tmp_path, {"base": "huella", "colour": "#fff"} if not extra else {"base": "huella"})
    with pytest.raises(SystemExit) as ex:
        cli.main(["x.zip", "--style", path, *extra])
    assert ex.value.code == 2 and message in capsys.readouterr().err


def test_a_poster_for_two_needs_who_inks(tmp_path, capsys):
    full = {k: render_editorial.STYLES["ink"][k] for k in styles.REQUIRED}
    full["street"] = {k: list(v) for k, v in full["street"].items()}
    path = _write(tmp_path, full, "nowho.json")
    with pytest.raises(SystemExit) as ex:
        cli.main(["a.zip", "--with", "b.zip", "--name", "Ana", "--with-name", "Bea", "--style", path])
    assert ex.value.code == 2 and 'needs "who", five inks' in capsys.readouterr().err


def test_python_dash_m_pinless_runs_the_cli():
    r = subprocess.run([sys.executable, "-m", "pinless", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "--style" in r.stdout
