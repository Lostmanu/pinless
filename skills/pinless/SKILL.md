---
name: pinless
description: Make a print-ready poster of a runner's year from their Strava export zip (or a folder of GPX/TCX/FIT files), locally and with privacy zones applied. Use when the user asks for a poster, map or print of their runs, rides or walks.
---

# Pinless poster

## Rules
- Only run the `pinless` command. Do NOT open, read, unzip, summarise or print the export or any GPX/TCX/FIT file:
  they contain the user's exact locations and must stay out of this conversation.
- Do not upload the export anywhere. Pinless works locally; it only sends the area's bounding box (for the atlas, each
  city's box and its centre rounded to about 1 km) to OpenStreetMap services, or to a public Overpass mirror when the
  main server is busy. If it prints `[osm] STOP`, the map servers are busy: suggest running it again later.
- If the export path is unknown, ask the user for it. Do not search their disk for it.

## Steps
1. Make sure it is installed: `pinless --help`. If not: `pip install git+https://github.com/Lostmanu/pinless`.
2. Ask only for what you need: the export path, the year, the name to print (optional) and the style
   (`huella` warm paper or `carbon` dark).
3. Run:
   `pinless "<export.zip>" --year <YEAR> --name "<NAME>" --style <STYLE> --out poster.png --pdf poster.pdf --preview poster.jpg`
   Add `--place "<City, Country>"` only if the user wants a city other than where they run most,
   and `--lang es` for Spanish labels.
   If they want every city they ran in on one print (trips, holidays), add `--layout atlas` instead of `--place`.
4. If the command prints `[privacy] STOP`, tell the user nothing was exported and why; do not try to bypass it.
5. Show the user `poster.jpg` and tell them where `poster.pdf` is. Remind them to look at it before sharing.
