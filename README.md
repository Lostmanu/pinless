# Pinless

**Your year of running as a print-ready poster of your city. From your Strava export, on your own computer, without showing where you live.**

<p align="center">
  <img src="docs/wall-huella.jpg" width="49%" alt="The warm paper poster of a year of runs in Madrid, framed in oak and hung above a sideboard">
  <img src="docs/wall-carbon.jpg" width="49%" alt="The same poster in the dark carbon style, in a black frame">
</p>
<p align="center"><sub>A3 prints in standard 40x50 cm frames, drawn to scale (<code>tools/mockup.py</code>). Demo posters made from a synthetic runner (<code>tools/simulate_runner.py</code>), not from a real person.</sub></p>

```bash
pip install git+https://github.com/Lostmanu/pinless
pinless export_12345678.zip --name "Lucia" --year 2025 --pdf poster.pdf
```

That is the whole thing. Pinless finds the area where you run most, downloads that area's streets from OpenStreetMap once (cached), removes the places that would give away where you live, and writes `poster.png` and a vector `poster.pdf` ready for a print shop.

## Get your Strava export

On strava.com: **Settings > My Account > Download or Delete Your Account > Request your archive**. Strava emails you a link to a zip, which can take a while. Pass the zip as it is; there is no need to unzip it.

It also works with a folder of GPX, TCX or FIT files exported from Garmin, Coros, Suunto, Komoot or any app that exports GPX.

## What you get

<p align="center">
  <img src="docs/poster-huella.jpg" width="49%" alt="Poster of a year of runs in Madrid, warm paper style">
  <img src="docs/poster-carbon.jpg" width="49%" alt="Same poster in the dark carbon style">
</p>

- Every run drawn on your city's OpenStreetMap streets, parks, rivers and sea (the sea and estuaries come from the coastline). The routes you repeat get darker and thicker, and a small legend explains it.
- The numbers: distance, runs, recorded hours, best 5K, best 10K, longest run and how many different streets you ran.
- Your name as the inscription, or as the headline (`--headline name`) when it is a gift.
- PNG at 300 dpi, vector PDF and SVG, at A4, A3 or 50x70 cm. A light JPEG preview with `--preview`.
- Twelve styles (below), plus the earlier `classic` and `sport` layouts.
- Routes coloured by how often you ran them, or by month, pace or elevation (`--color-by`).
- An optional band of charts: kilometres per month, the time of day you run and the direction of your streets (`--graphs`).
- An atlas of every city you ran in (`--layout atlas`) and a finisher print of one race (`--layout race`), both below.
- One poster for two people, one ink each (`--with`), below.

## Styles

<p align="center"><img src="docs/styles.jpg" width="100%" alt="The same Madrid poster in twelve styles"></p>

`--style` picks one of `huella`, `carbon`, `midnight`, `blueprint`, `ink`, `graphite`, `neon`, `lavender`, `sunset`, `forest`, `blush` and `ocean`. They work for the poster, the atlas and the race print. The dark ones (`midnight`, `graphite`, `neon`) give the routes a soft glow, still drawn as vectors.

## Colour and charts

<p align="center"><img src="docs/colour-modes.jpg" width="100%" alt="One poster coloured by repetition, by month, by pace and by elevation"></p>

The line weight always grows with how often you ran a street; `--color-by` decides what the colour says: `heat` (how often, the default), `single` (one ink), `month` (when in the year), `pace` (how fast each run was) or `elevation` (how high, from your watch). `--graphs` adds kilometres per month, a 24-hour rose of when you start and a rose of the directions you run, all in the local time and place of each run.

<p align="center"><img src="docs/poster-graphs.jpg" width="49%" alt="Poster with the band of charts under the map"></p>

## Atlas: every city you ran in

<p align="center">
  <img src="docs/atlas-huella.jpg" width="40%" alt="Atlas of six cities run in one year: Madrid, Lisbon, Paris, Rome, Valencia and Zurich">
  <img src="docs/wall-atlas.jpg" width="58%" alt="The atlas framed in oak and hung above a sideboard">
</p>
<p align="center"><sub>Demo atlas from the synthetic runner with trips (<code>tools/simulate_trips.py</code>); try it with <code>pinless demo/strava_export_atlas_demo.zip --layout atlas</code>.</sub></p>

```bash
pinless export_12345678.zip --layout atlas --name "Lucia" --year 2025 --pdf atlas.pdf
```

One small map per city on the same print: the city where you ran most comes first, then the others in the order you first ran there, each with its name, country, kilometres and runs. Line colour means the same in every map, so a street you ran fifty times at home is darker than a one-off loop by the hotel. Up to 9 maps by default (`--max-cities`, at most 12); the smallest cities are left out of the maps, the print says how many, and the totals still count them.

## Race: one run as a finisher print

<p align="center">
  <img src="docs/race.jpg" width="40%" alt="Finisher print of a half marathon in Madrid with its elevation profile, time, pace and kilometre splits">
  <img src="docs/wall-race.jpg" width="58%" alt="The race print in a black frame hung above a sideboard">
</p>

```bash
pinless export_12345678.zip --layout race --activity 2025-04-06 --title "Spring Half Marathon" --name "Lucia"
```

The course coloured by your pace along it (or `--color-by elevation` or `single`), its elevation profile, distance, time, pace, climb and every kilometre split, with its date. The headline is the city unless you give one with `--title`: the activity's own name is not printed by default, because names often say where a run starts ("from Mum's"). `--activity` takes `longest` (the default), `latest`, a date or an activity id.

A run that starts or ends at one of your private places (a long run from home) is refused, since one course drawn from your door points at it. Pick another run, or pass `--allow-private-ends`: the print then shows one unbroken stretch of the run that stays more than 1.6 to 2.6 zone radii (800 to 1300 m by default, drawn at random each time) from those places, wherever the run comes back to them, shortened by a random 200 to 800 m at each end that one lies beyond, and its distance, time, pace, profile, climb and splits describe only that stretch.

## For two: one map, one ink each

<p align="center">
  <img src="docs/couple-huella.jpg" width="40%" alt="Poster of two runners' year in Madrid: Lucia's streets in red, Marcos's in blue, the streets they share in black">
  <img src="docs/wall-couple.jpg" width="58%" alt="The poster for two framed in oak and hung above a sideboard">
</p>
<p align="center"><sub>Demo poster from two synthetic runners (<code>tools/simulate_runner.py</code>); try it with <code>pinless demo/strava_export_madrid_demo.zip --with demo/strava_export_madrid_partner_demo.zip --name Lucia --with-name Marcos</code>.</sub></p>

```bash
pinless lucia.zip --with marcos.zip --name "Lucia" --with-name "Marcos" --year 2025 --pdf pareja.pdf --preview pareja.jpg
pinless lucia.zip --with marcos.zip --name "Lucía" --with-name "Marcos" --headline name --style midnight --lang es
```

Two exports, one editorial poster. The map is the area where you both run: a place only one of you runs never wins it (`--place` or `--bbox` choose another). Each export is read with its own type (`--with-type` when the second one names the sport differently), and if only some of one person's activities fall inside the map, Pinless says how many, with that person's name.

What the colours mean. The colour of a street says whose it is, in five steps: the first person's ink, mostly the first person's, both (the style's own ink), mostly the second person's, the second person's ink. It is the street's share, measured on paper: two lines less than about half a millimetre apart on an A3 print are one street (the same distance on the ground at every paper size), and the colour belongs to the street, not to the line, so one run of yours on a street the other ran twenty times takes their ink. The line weight is how often either of you ran the street, counted the same way: a street one of you ran six times is as heavy as one you each ran three times. The first export always takes the warm ink; swap the exports to swap the inks. `--color-by` and `--graphs` do not apply to a poster for two.

The numbers, and how each is computed. Every number comes from the coordinates and the clocks in the two files (and the street count from the map's OpenStreetMap street names); nothing else is looked up.

- Kilometres: both of you together, and a bar that splits them between you. A run together counts for each of you, since each of you ran it.
- Runs together and kilometres side by side: two activities, one of each of you, that overlap in time for at least 10 minutes (and half of the shorter one) and stay close (a median gap of 75 m or less, and within 150 m for 80 % of the shared time), each activity matched once. The kilometres are the mean of what the two of you covered while together. Without a clock in one of the exports, or with no run together, these two cells show your runs and recorded hours instead.
- Best 5K, best 10K and longest run: one for each of you, side by side, in your own ink.
- Streets you both ran: counted on the drawn lines only, after the privacy cuts.

Privacy for two. A map of two people's runs can show both homes, and the visits between them, so:

- Every private place of either of you (a home, a workplace, a gym) hides both people's lines. Each person's places are found with that person's own threshold, so the one who runs less keeps their places too, and with fewer than 75 runs a single loop already marks where it started.
- One zone per place: a home you share, or a door both of you run from, is one zone, drawn once and cutting both lines, so both inks break at the same edge and no hole in one ink is crossed by the other.
- Places where your ends meet are hidden even when neither of you alone starts or ends there often enough to make it a place (a cafe where you meet).
- If one of you has nothing left on the map after the cuts, nothing is exported, rather than a poster of only one of you. A `--zone` hides both people's lines.
- The colour around a hole says whose neighbourhood it is. That is the same thing `--name` already ties to a person on a solo poster.
- Not covered: several posters of the same place (one of each of you and one together, each with its own random zones) laid over each other narrow a hole down; a solo poster of one of you made without `--with` still shows the runs from the other's door; and, as on a solo poster, a place you started from only a few times among many runs, or only in another year or with another activity type, is not learned. Hide such places with `--zone`.

Use the other person's export only with their consent; make the poster together and look at it before printing or sharing it.

## Privacy

A heatmap of your runs shows where you live. Pinless removes that before drawing anything:

1. It trims the first and last 300 m of every activity.
2. It finds every place where your runs often start or end (home, work, the gym) and deletes all points inside an irregular zone around each one, including runs that only pass by. In the atlas this is done city by city, the private places of every city also protect the others (a long run grouped elsewhere can still start at home), and where you ran only a few times a single loop run is enough to hide where it started (a hotel, a family home).
3. It checks the result. If any line still touches a zone, nothing is exported.
   The race print works on a single course and always draws one unbroken stretch of it. A course that only passes one of your private places is drawn whole, since a gap in it would point at the very place it hides. A run that starts or ends at a private place (within 1.5 zone radii, in its first or last 300 m) is refused unless you pass `--allow-private-ends`, which keeps a stretch well clear of that place, as above; a `--zone` is always kept clear the same way. For the race print, a private place is also where two or more of your other runs start or end close to where this one does, within 1.5 zone radii of its first or last 300 m (a new home, a holiday flat, before years of history learn it). A loop from a place none of your other runs starts from (a first run from a hotel) cannot be told from a race start: it is drawn whole, and Pinless says so, so you can add a `--zone` there.
4. It never prints coordinates or start points, and nothing outside the visible map is written into the PNG, PDF or SVG. Dates are printed only on the race print, where they are the point, and never an activity's own name: the race headline is the city or what you write with `--title`. The favourite street is never one within 1 km of a private zone.
5. Your files never leave your computer. Only the area is sent out: its bounding box to OpenStreetMap's Overpass service for the streets, and its centre, rounded to about 1 km, to OpenStreetMap's Nominatim for the city name (in the atlas, the same for each city). A second export (`--with`) adds no request. When the main Overpass server is busy, Pinless tries the public mirror overpass.private.coffee; to use another server, set `PINLESS_OVERPASS` to its API address.

Stray GPS fixes at the start or end of a file (a watch that opens with yesterday's position, then jumps) are dropped on import. A run that starts in one place and ends in another, once, only has its ends trimmed. This reduces the risk; it is not a guarantee of anonymity. A map of where you run still says something about you. Look at the poster before you share it, and hide more places with `--zone LAT,LON,RADIUS_M` (repeatable).

## Options

| Option | What it does |
|---|---|
| `--year 2025` | Only that year |
| `--type Run` | Activity type from the export (`Run`, `Ride`, `Walk`, `Hike`...) |
| `--place "Valencia, Spain"` | Use a named place instead of the area where you run most |
| `--bbox S,W,N,E` | Use an exact box |
| `--title`, `--name`, `--headline city\|name` | Headline text and inscription |
| `--layout editorial\|atlas\|race`, `--max-cities 9`, `--activity longest`, `--allow-private-ends` | One city, one small map per city, or one race |
| `--color-by heat\|single\|month\|pace\|elevation`, `--graphs` | What the line colour says; the band of charts |
| `--style huella\|carbon\|midnight\|...`, `--size A4\|A3\|50x70` | One of twelve styles; paper |
| `--lang en\|es` | Labels in English or Spanish |
| `--out`, `--pdf`, `--svg`, `--preview` | Outputs |
| `--zone LAT,LON,R` | Extra private zone (with `--with`, it hides both people's lines) |
| `--with EXPORT`, `--with-name NAME` | A second person's export and name: one poster for two, one ink each |
| `--with-type Run` | Activity type in the second export, when it names the sport differently (default: `--type`) |

## Use it from a coding agent

`skills/pinless/SKILL.md` lets Claude Code, Codex, Cursor or any agent that reads Agent Skills make the poster for you. The skill only runs the command above. It tells the agent not to open your export, so your GPS points stay out of the conversation.

## How it compares

- [running_page](https://github.com/yihong0618/running_page) builds a personal running website synced from Strava or Garmin. Pinless makes one printable poster.
- [GitHubPoster](https://github.com/yihong0618/GitHubPoster) draws calendar and circular posters of activity. Pinless draws your routes on your city's streets.
- [Heatmapster](https://heatmapster.com) is a free web service that connects to your Strava account. Pinless needs no account connection and runs locally.

## Development

```bash
git clone https://github.com/Lostmanu/pinless && cd pinless
pip install -e ".[dev]"
pytest
```

The demo export in `demo/` is a synthetic runner generated on real OpenStreetMap streets of Madrid by `tools/simulate_runner.py`. Its true home and work points are stored apart in `demo/truth.json`, only so the tests can check that no published line comes near them. The partner demo (`demo/strava_export_madrid_partner_demo.zip`) is a second synthetic runner with their own home and workplace, who runs some weekend runs side by side with the first one, from the first one's door; its places and the number of runs together are in `demo/truth_partner.json`.

The tests run offline. The street statistics use `tests/data/madrid_streets.json.gz`, a small extract of the named OpenStreetMap streets around the demo runs (ODbL). `tools/make_street_fixture.py` rebuilds it from the full Madrid graph, and its `--check` option confirms that it gives the same street counts as the full graph.

## Credits and license

Map data © OpenStreetMap contributors, available under the ODbL; every poster carries the attribution. Fonts: Barlow Condensed and Manrope, SIL Open Font License 1.1 (`pinless/fonts`). Code: MIT.

Not affiliated with Strava.
