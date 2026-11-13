# Pinless: flags, messages and colours

`<PINLESS>` and `<PYTHON>` are the paths the setup script printed. Every command ends with the three outputs,
`--out "~/Pinless/<name>.png" --pdf "~/Pinless/<name>.pdf" --preview "~/Pinless/<name>.jpg"` (Pinless expands `~` and
makes the folder), plus `--lang es` for Spanish labels, `--size A3|A4|50x70` and `--style <name or file>`. Quote
every path.

## By kind of poster

**My year in my city** (the default layout):
`"<PINLESS>" "<export>" --year <YEAR> --name "<NAME>" --color-by heat|pace|month|single|elevation ...`
- No `--year` for all years. No `--name` when they want none.
- `--type Ride`, `--type Walk` or `--type Hike` for a poster of rides, walks or hikes (the default is runs).
- Another city: `--place "<City, Country>"`. Every city they ran in: `--layout atlas` (one small map per city, up to
  `--max-cities 9`, from 1 to 12) instead of `--place`.
- Offer under "Something else", one city only (not the atlas): `--graphs` (charts of km per month, time of day and
  direction under the map), `--headline name` (the name as the headline instead of the city; it needs `--name`). On
  either: `--relief shade` (terrain relief where the land is hilly).

**One race**: `"<PINLESS>" "<file>" --layout race --activity <longest|latest|YYYY-MM-DD> --title "<RACE>" --distance <KM>`
- From an export: `--activity` picks the run and `--type Ride`, `Walk` or `Hike` another sport (the default is
  runs). A single file is one activity: no `--activity`, and its type is read from the file.
- `--distance`: 21.0975 for a half marathon, 42.195 for a marathon, 10 for a 10K; leave it out if they do not know.
  The activity's own name is never printed: only `--title`.
- `--color-by pace` is the default (the line shows the pace along the course); `elevation` or `single` if asked.

**A hike or trail**: `"<PINLESS>" "<file>" --layout trail --title "<ROUTE>"`
- From an export: `--type Hike` (or `Walk`, `TrailRun`, `Run`) and `--activity <longest|latest|YYYY-MM-DD>`. Without
  `--type` the trail print takes the longest activity of any sport, which may be a ride.
- Relief and the colours of the land are on by default and fetch elevation and land cover tiles of the area from
  Amazon S3 (never the route). `--relief off` and `--ground style` (plain paper) leave those requests out.
- `--distance <KM>` for an official length. `--bbox S,W,N,E` frames a wider map (it must hold the whole route), only
  if they ask for a different framing.

**For two**: `"<PINLESS>" "<first>" --with "<second>" --name "<FIRST NAME>" --with-name "<SECOND NAME>" --year <YEAR>`
- Take both paths and names from the user, never from the files. The first file takes the warm ink: swap the files to
  swap the inks. `--type` for another sport than runs; `--with-type <TYPE>` when the second export names it
  differently. `--color-by` and `--graphs` do not apply. Remind them to look at it together before sharing.

**A single GPX, FIT or TCX file that starts or ends at a home** (race or trail): add `--private-ends`. Its first and
last points become private places, hidden like the ones Pinless learns from an export, and the print shows only a
stretch well clear of them, with numbers for that stretch. Never ask for the coordinates instead.

## What the command prints, and what to do

- Any privacy refusal: a `[privacy] STOP` line, "Nothing was exported", "nothing of yours is left on this map after
  the privacy cuts", "too little left after the privacy cuts", or "--zone number N does not touch this map". Nothing
  was exported. Tell the user why, in their words. Never run it again with a smaller `--trim` or `--zone-radius`,
  without a `--zone` they gave, or with `--allow-private-ends` they did not ask for.
  - Race or trail, "starts or ends at a private place": the run began at home or work. From an export, suggest
    another one (`--activity latest`, or a date). Add `--allow-private-ends` only if they ask for that one anyway, and
    tell them the print then shows only a stretch well clear of that place, with numbers for that stretch only.
  - "too little left": the activity stays near a private place all along; suggest another one.
  - "--zone number N does not touch this map": ask them to check that zone (latitude first, then longitude).
  - With `--with`, a refusal that names the second person is about the second file (none of their activities on the
    map, or nothing left after the privacy cuts): suggest `--place "<City, Country>"` for a city both ran in.
- "this run starts and ends at the same place, and no other run of yours starts there" (or the same for a route):
  ask whether that place is private (a hotel, a friend's home, their own door); if it is, run again with
  `--private-ends`.
- "this export holds one activity (a single file)": expected for a single file; the home question covers it.
- `[osm] STOP ... Run the same command again later`: the public map servers are busy. Nothing was exported. Offer
  to try again in a few minutes. `[osm] STOP ... Nothing was sent or exported` with "180th meridian": that map cannot
  be made yet; waiting does not help.
- `[trail] STOP` about `--bbox`: the box does not hold the route; widen it or leave `--bbox` out.
- "no activities match the type/year filters (this export has: ...)": pick a `--type` from the list it gives, or
  another year (the demos are all 2025).
- `[distance] ... GPS noise ...` on a race or trail from a single file: offer `--distance` with the official length.
- An error from the option parser (exit code 2, "usage: pinless ..."): fix the flags; do not show it to the user.
- The closing lines to tell them, which are part of the fun: a year poster prints `[stats]` (runs, km, hours, best
  5K and 10K, streets, favourite street) and a poster for two its own summary; a race or trail prints its `[render]`
  line with the distance; an atlas lists the cities drawn.

## The twelve styles

| Style | Paper | Lines (faint to strong) | Text, accent |
|---|---|---|---|
| `huella` | cream `#f5f0e6` | brick red to dark brown `#cb694e` to `#46231d` | near black `#222820`, red `#b33b25` |
| `carbon` | dark green `#131e19` | copper to cream, glowing | cream `#f2eddf`, copper `#dfa575` |
| `midnight` | night blue `#0f1626` | light blue to white, glowing | white `#eef2f7`, sky blue `#5fd4ff` |
| `blueprint` | blue `#1d4a86` | pale blue to yellow `#ffd45c` | white `#f3f7fc`, yellow `#ffd45c` |
| `ink` | white `#fbfaf7` | grey to black | black `#111111` |
| `graphite` | charcoal `#222324` | grey to white, glowing | off white `#f1f1ef`, orange `#ff6b3d` |
| `neon` | almost black `#0c0812` | pink to yellow `#ffe066`, glowing | white `#f6f0ff`, pink `#ff3d9a` |
| `lavender` | lilac `#efebf5` | violet to deep purple `#2f1b66` | aubergine `#2a2140`, violet `#6741d9` |
| `sunset` | peach `#fbf1e6` | orange to purple `#5f3dc4` | plum `#2a1d2e`, orange `#e8590c` |
| `forest` | sage `#eef1e8` | green to deep green `#13301a` | dark green `#1f2d1f`, green `#2f6b3a` |
| `blush` | pink `#f8ece9` | rose to wine `#4a0629` | burgundy `#3a1f24`, magenta `#c2185b` |
| `ocean` | pale blue `#edf4f7` | blue to navy `#062a40` | navy `#0f2a3d`, sea blue `#0b6e99` |

## Their own colours

When they describe colours ("green and gold", "my club's red and white"), write a small style file such as
`~/Pinless/<name>-style.json` that starts from the closest style with `base` and changes only what they asked:

```json
{
  "base": "carbon",
  "bg": "#0e2a22",
  "accent": "#e8c15a",
  "heat": ["#8f7a3c", "#b89645", "#e0b84f", "#f3d98c", "#fff4cf"],
  "glow": 0.12
}
```

Keys: `bg` paper, `ink` titles and numbers, `sub` small text, `accent` the year and the "km" unit, `green` parks,
`water` and `water_line`, `rule` the lines between sections, `street` (main roads, side streets, paths), `heat` five
line colours from a street run once to the one run most (faint to strong against the paper), `glow` a soft halo for
dark paper (about 0.1), `who` the five inks of a poster for two. Colours are `#rrggbb`. `"<PYTHON>" -m pinless.styles
<style>` prints every key of a style. Pass the file with `--style "<file>"`. Pinless refuses a file with a mistake
and says what to change, and warns when a colour would be hard to read on the paper: fix the file and run again.
Show them the result and adjust; this is the part people enjoy.

## Getting the file

- **Strava, everything**: on strava.com, Settings > My Account > Download or Delete Your Account > Request your
  archive. Strava emails a link to a zip, which can take a while. Use the zip as it is, without unzipping it.
- **One activity from Strava**: open it on strava.com, the "..." menu, Export GPX.
- **Garmin Connect**: open the activity, the gear menu, Export to GPX (or Export Original for a FIT file).
- **Komoot, Wikiloc, Coros, Suunto, Polar**: each activity has a GPX download.
- A folder of GPX, TCX or FIT files works like an export.
