"""Your own style: a small JSON file with the colours of the paper, the map, the routes and the text.

    {
      "base": "midnight",              optional: start from a built-in style and change only what you list
      "bg": "#0f1626",                 the paper
      "ink": "#eef2f7",                titles, numbers and the most-run routes on a poster for two
      "sub": "#8a96ab",                small text
      "accent": "#5fd4ff",             the year and the "km" unit
      "green": "#121c2f",              parks
      "water": "#16223a",              rivers, lakes and sea
      "water_line": "#22324d",         thin waterways
      "rule": "#2c3852",               the lines between sections
      "street": {"major": ["#2a3650", 0.32], "minor": ["#1e283d", 0.17], "path": ["#1a2336", 0.11]},
      "heat": ["#1f6f9c", "#2aa3d6", "#5fd4ff", "#b7f0ff", "#ffffff"],
      "glow": 0.16,                    optional, 0 to 0.5: a soft halo under the routes, for dark paper
      "who": ["#ffb45c", "#d8c4a7", "#eef2f7", "#adc2d9", "#4fc3ff"]
    }

street: colour and line width (points) of main roads, side streets and paths. heat: five route colours, from a street
run once to the most-run one. who (only for the poster for two, --with): first person, mostly first, both, mostly
second, second person. Colours are "#rrggbb" or a colour name ("white", "tomato").

Print a built-in style as a file to start from: python -m pinless.styles midnight > my-style.json
"""
from __future__ import annotations

import copy
import json
import os
import sys

from matplotlib.colors import is_color_like, to_hex, to_rgb

from .render_editorial import STYLES

REQUIRED = ("bg", "ink", "sub", "accent", "green", "water", "water_line", "rule", "street", "heat")
OPTIONAL = ("base", "glow", "who")
STREETS = ("major", "minor", "path")


class StyleError(ValueError):
    """A style file that cannot be used; the message says what to change."""


def is_file(value):
    """Is --style a file rather than the name of a built-in style?"""
    return bool(value) and (value.lower().endswith(".json") or os.path.isfile(value))


def _colour(value, where):
    if not isinstance(value, str) or not is_color_like(value):
        raise StyleError(f'{where}: {value!r} is not a colour (use "#rrggbb" or a name such as "white")')
    return to_hex(value)


def _colours(value, n, where):
    if not isinstance(value, list) or len(value) != n:
        raise StyleError(f"{where}: needs a list of {n} colours")
    return [_colour(v, f"{where}[{i}]") for i, v in enumerate(value)]


def parse(data, where="style"):
    """A complete style dict from the JSON object of a style file. Raises StyleError."""
    if not isinstance(data, dict):
        raise StyleError(f"{where}: the file must hold one JSON object {{...}}")
    unknown = sorted(set(data) - set(REQUIRED) - set(OPTIONAL))
    if unknown:
        raise StyleError(f"{where}: unknown key {', '.join(map(repr, unknown))}; the keys are "
                         f"{', '.join(REQUIRED + OPTIONAL)}")
    base = data.get("base")
    if base is not None:
        if base not in STYLES:
            raise StyleError(f"{where}: base {base!r} is not a built-in style; choose one of {', '.join(STYLES)}")
        out = copy.deepcopy(STYLES[base])
        out["street"] = dict(out["street"])
    else:
        missing = [k for k in REQUIRED if k not in data]
        if missing:
            raise StyleError(f"{where}: missing {', '.join(missing)} (or start from a built-in style with "
                             f'"base": "huella")')
        out = {}
    for key in ("bg", "ink", "sub", "accent", "green", "water", "water_line", "rule"):
        if key in data:
            out[key] = _colour(data[key], key)
    if "street" in data:
        street = data["street"]
        if not isinstance(street, dict) or not street or set(street) - set(STREETS):
            raise StyleError('street: an object with "major", "minor" and "path", each ["#rrggbb", width]')
        if base is None and set(street) != set(STREETS):
            raise StyleError('street: needs "major", "minor" and "path" (or a "base" style to take the rest from)')
        out.setdefault("street", {})
        for kind, value in street.items():
            if not (isinstance(value, list) and len(value) == 2):
                raise StyleError(f'street.{kind}: ["#rrggbb", width in points], e.g. ["#d5d0c4", 0.32]')
            width = value[1]
            if isinstance(width, bool) or not isinstance(width, (int, float)) or not 0 < width <= 2:
                raise StyleError(f"street.{kind}: the width must be a number of points above 0 and up to 2")
            out["street"][kind] = (_colour(value[0], f"street.{kind}"), float(width))
    if "heat" in data:
        out["heat"] = _colours(data["heat"], 5, "heat")
    if "glow" in data:
        glow = data["glow"]
        if isinstance(glow, bool) or not isinstance(glow, (int, float)) or not 0 <= glow <= 0.5:
            raise StyleError("glow: a number from 0 (none) to 0.5")
        out["glow"] = float(glow)
    if "who" in data:
        out["who"] = _colours(data["who"], 5, "who")
    return out


def load(path):
    """(name, style) from a JSON style file. Raises StyleError."""
    name = os.path.splitext(os.path.basename(path))[0] or "style"
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise StyleError(f"no file {path}") from None
    except json.JSONDecodeError as ex:
        raise StyleError(f"not valid JSON at line {ex.lineno}, column {ex.colno}: {ex.msg}") from None
    except (OSError, UnicodeDecodeError) as ex:
        raise StyleError(f"cannot read it ({type(ex).__name__})") from None
    return name, parse(data, name)


def _contrast(a, b):
    """WCAG contrast ratio of two colours."""
    def lum(c):
        lin = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in to_rgb(c)]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def warnings(S):
    """What may print badly: text too close to the paper, routes that vanish into the streets."""
    out = []
    for key, least in (("ink", 4.5), ("sub", 2.5), ("accent", 2.0)):
        if _contrast(S[key], S["bg"]) < least:
            out.append(f"{key} {S[key]} is hard to read on bg {S['bg']} (contrast {_contrast(S[key], S['bg']):.1f}, "
                       f"aim for {least} or more)")
    if _contrast(S["heat"][0], S["bg"]) < 1.5:
        out.append(f"heat[0] {S['heat'][0]} almost vanishes on the paper: streets run once will hardly show")
    major = S["street"]["major"][0]
    if _contrast(S["heat"][0], major) < 1.3:
        out.append(f"heat[0] {S['heat'][0]} is close to the main roads {major}: routes run once may read as streets")
    return out


def register(path):
    """Load a style file into the editorial styles (shared by the atlas and the race print) under a name that cannot
    clash with a built-in one, and return that name. Raises StyleError."""
    name, S = load(path)
    key = f"file:{name}"
    STYLES[key] = S
    return key, warnings(S)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0] not in STYLES:
        print(f"usage: python -m pinless.styles STYLE > my-style.json   (STYLE: {', '.join(STYLES)})", file=sys.stderr)
        return 2
    S = STYLES[argv[0]]
    data = {k: S[k] for k in REQUIRED}
    data["street"] = {k: list(v) for k, v in S["street"].items()}
    if "glow" in S:
        data["glow"] = S["glow"]
    if "who" in S:
        data["who"] = S["who"]
    sys.stdout.write(json.dumps(data, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
