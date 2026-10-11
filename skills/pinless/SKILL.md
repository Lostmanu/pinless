---
name: pinless
description: Make a print-ready poster of a runner's year from their Strava export zip (or a folder of GPX/TCX/FIT files), locally and with privacy zones applied. Use when the user asks for a poster, map or print of their runs, rides or walks, or of two people's runs on one map.
---

# Pinless poster

## Rules
- Only run the `pinless` command. Do NOT open, read, unzip, summarise or print the export or any GPX/TCX/FIT file:
  they contain the user's exact locations and must stay out of this conversation.
- Do not upload the export anywhere. Pinless works locally; it only sends the area's bounding box and its centre
  rounded to about 1 km (for the atlas, each city's) to OpenStreetMap services, or to a public Overpass mirror when the
  main server is busy. If it prints `[osm] STOP`, the map servers are busy: suggest running it again later.
- If the export path is unknown, ask the user for it. Do not search their disk for it.
- Add `--relief shade` only when the user asks for terrain or relief; it also downloads Copernicus elevation data
  from Amazon S3, which learns the 1-degree cells around the map and, where the land is hilly, the blocks of about
  30 km inside them that cover it (never the routes); without it nothing is sent there.

## Steps
1. Make sure it is installed: `pinless --help`. If not: `pip install git+https://github.com/Lostmanu/pinless`.
2. Ask only for what you need: the export path, the year, the name to print (optional) and the style
   (`huella` warm paper, `carbon` dark green; also `midnight`, `blueprint`, `ink`, `graphite`, `neon`, `lavender`,
   `sunset`, `forest`, `blush`, `ocean`).
3. Run:
   `pinless "<export.zip>" --year <YEAR> --name "<NAME>" --style <STYLE> --out poster.png --pdf poster.pdf --preview poster.jpg`
   Add `--place "<City, Country>"` only if the user wants a city other than where they run most,
   and `--lang es` for Spanish labels.
   If they want every city they ran in on one print (trips, holidays), add `--layout atlas` instead of `--place`.
   `--color-by month|pace|elevation|single` changes what the line colour says (default `heat`, how often), and
   `--graphs` adds charts of km per month, time of day and direction under the map (single-city poster only).
   For one race as a finisher print: `--layout race --activity <YYYY-MM-DD | latest | longest | id>`, with
   `--title "<race name>"` if they want its name printed (the activity's own name is never printed by default).
   For one poster of two people (a couple, two friends), ask the user for both export paths and both names; take
   them from the user only, never from a search or from the files. Do not open either export. Run:
   `pinless "<first.zip>" --with "<second.zip>" --name "<FIRST NAME>" --with-name "<SECOND NAME>" --year <YEAR> --style <STYLE> --out poster.png --pdf poster.pdf --preview poster.jpg`
   The first export takes the warm ink. If the second export names the sport differently, add `--with-type <TYPE>`.
   Tell the user to use the other person's export only with their consent, and to look at the poster together
   before printing or sharing it. `--color-by` and `--graphs` do not apply to a poster for two.
4. If the command prints `[privacy] STOP`, tell the user nothing was exported and why; do not try to bypass it.
   Never run it again with a smaller `--trim` or `--zone-radius`. With `--with`, a STOP or a refusal that names the
   second person refers to the second export (for example: nothing of theirs inside the map, or nothing left after
   the privacy cuts): suggest `--place "<City, Country>"` for a city both ran in, never weaker privacy.
   If it says a `--zone` does not touch the map, ask the user to check that zone (latitude first, then longitude);
   never drop it to get a poster.
   For `--layout race`, a STOP that says the run starts or ends at a private place means it began at home or work:
   suggest another run. Add `--allow-private-ends` only if the user asks for that run anyway, and tell them the print
   will show only a stretch well clear of that place and describe only that stretch. If the log says the run is a
   loop from a place no other run starts from, ask whether that place is private (a hotel, a friend's home); if it
   is, add `--zone`.
5. Show the user `poster.jpg` and tell them where `poster.pdf` is. Remind them to look at it before sharing.
