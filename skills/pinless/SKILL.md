---
name: pinless
description: Make a print-ready poster of someone's runs, rides, walks or hikes from their Strava export or a GPX, TCX or FIT file (a year in their city, an atlas of every city, one race, one trail, or one poster for two), on their own computer with privacy built in. Guides them with short multiple-choice questions (colours, size, words) and installs what it needs. Use when the user asks for a poster, map, print or gift of their runs, rides, walks or hikes, or mentions Pinless.
argument-hint: "[the path of your Strava export or GPX file: paste the path, do not attach the file]"
---

# Pinless poster

Take the user from nothing to a poster they love. You install and run everything; they only answer short questions
and look at previews. Never ask them to type a command, and do not show commands unless they ask.

Paths below: `${CLAUDE_SKILL_DIR}` is the folder of this file and `${CLAUDE_PLUGIN_DATA}` the plugin's data folder.
Claude Code fills them in; an agent that does not should use this file's folder and `~/.pinless` instead. The flags
of each poster and what to do with each message of the command are in [reference.md](reference.md): read it before
the first run. In PowerShell, start a command that begins with a quoted path with `& ` (`& "<PINLESS>" ...`).

## Privacy rules (always)

- Never open, read, unzip, list, copy or summarise the export or any GPX, TCX or FIT file, and never search the disk
  for one: they hold the user's exact locations and must stay out of this conversation. Ask for the file's path,
  never for the file: an attached or @-mentioned file is read into the conversation. In a terminal, dragging the
  file onto the window types its path. In an app with a chat box, it attaches the file: ask them to copy the path
  instead (Windows: Shift + right-click the file, "Copy as path"; macOS: right-click, hold Option, "Copy ... as
  Pathname"). Only check that the path exists, and pass it to `pinless`. If they attached the file anyway, do not use
  what you were shown; ask for the path.
- Never ask for the coordinates of a home or any other private place. A single file that starts or ends at a home
  gets `--private-ends`, which hides its own ends without anyone typing where the door is.
- Nothing is uploaded. Pinless sends only the map's area and its centre rounded to about 1 km (for an atlas, each
  city's) to OpenStreetMap, or to a public Overpass mirror when the main server is busy. The trail print and
  `--relief shade` also fetch elevation and land cover tiles for the area from Amazon S3 (never the routes). Say so
  in one line before the first poster.
- When Pinless refuses for privacy (a `[privacy]` line that says STOP, nothing exported or nothing left, or a
  `--zone` that does not touch the map), do what reference.md says. Never run it again with a smaller `--trim` or
  `--zone-radius`, without a `--zone` the user gave, or with `--allow-private-ends` they did not ask for.
- A poster for two uses someone else's export: only with their consent, which you ask about.

## How to ask

- Speak the user's language. Pass `--lang es` when they write in Spanish; otherwise English labels (Pinless prints
  only English or Spanish).
- Ask with your multiple-choice question tool when you have one (in Claude Code, AskUserQuestion: up to 4 questions
  per call, 2 to 4 options each, and the user can always type their own answer in "Other"). Without one, ask in one
  short message with numbered choices. Put the recommended option first.
- Skip any question already answered ("a blue poster of my half marathon" answers the kind and the colours), and
  keep each question to one line. A name, a title, a city or a date is typed in "Other": say so in the question.

## 1. Get ready (quietly, before the first question is answered)

Run the setup script with a 10-minute timeout; the first time it downloads a few hundred MB. Try these in order and
stop at the first that exits 0:

1. The environment of an earlier session, if it exists: `"${CLAUDE_PLUGIN_DATA}/venv/Scripts/python.exe"` on
   Windows, `"${CLAUDE_PLUGIN_DATA}/venv/bin/python"` elsewhere.
2. `py -3` on Windows; `python3`, then `python` (on Windows, a "Python was not found" message about the Microsoft
   Store means that command is not a real Python).
3. `uv run --python-preference only-managed --python 3.12 --no-project`, with `uv`, `~/.local/bin/uv` or
   `~/.cargo/bin/uv`: uv brings its own Python.

Each runs `"${CLAUDE_SKILL_DIR}/scripts/setup.py" --home "${CLAUDE_PLUGIN_DATA}"`. Exit 2 (or any failure other
than 3) means that Python cannot do it: go on to the next. Exit 3 means the download failed or another install is
running: tell the user what it printed and offer to try again later.

If nothing works, tell the user Pinless needs Python and ask: "Install it for me (recommended)" runs uv's official
installer (`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` on Windows,
`curl -LsSf https://astral.sh/uv/install.sh | sh` elsewhere), then step 3 with `~/.local/bin/uv`; "I'll install Python
myself" points to python.org/downloads (on Windows, tick "Add python.exe to PATH"). On Debian or Ubuntu, when the
script said `sudo apt install python3-venv`, they can also run that themselves.

Keep the lines it prints: `PINLESS` (the command), `PYTHON` (its Python), `STYLES` (a picture of the twelve styles)
and the `DEMO_*` files. Run it on every new conversation: it takes a second when nothing changed.

## 2. What poster, from which file

Ask together:

- **What should the poster show?** "My year in my city" (every run of a year on the streets where they run most;
  an atlas of every city is offered next), "One race" (a finisher print with time, pace and splits), "A hike or
  trail" (one route on a mountain map with relief and the colours of the land), "For two" (their runs and another
  person's on one map, one colour each).
- **Do you have the file?** "My Strava export (.zip)", "A GPX, FIT or TCX file", "Not yet".

Then ask for its path as the privacy rules say. "Not yet": explain how to get it (reference.md, "Getting the file")
and offer to try the whole thing now with a demo, so they can pick colours while they wait: `DEMO_YEAR` for a year
or a race, `DEMO_ATLAS` for an atlas, `DEMO_YEAR` with `DEMO_TWO` for two, `DEMO_TRAIL` for a trail. The demos are a
made-up runner in 2025: use `--year 2025`, and skip the questions about home and names (offer "Demo" as the name).
If setup printed no `DEMO_*`, there is no demo to offer.

For two, ask for both paths, both names, and whether the other person agreed.

## 3. Colours and size

Open the sample first, `"<PYTHON>" "${CLAUDE_SKILL_DIR}/scripts/show.py" "<STYLES>"`. On `OPENED`, say you opened
it; on `SHOW` (no screen), say where it is. Without `STYLES`, describe the styles from reference.md. Then ask:

- **Which colours?** (say "or another from the picture, or your own colours" in the question)
  "Huella, warm paper" (cream paper, brick-red lines), "Carbon, dark green" (deep green, cream and copper lines),
  "Midnight, night blue" (glowing light blue on dark blue), "Ink, black and white". In each option's preview, list
  its paper, line and text colours from reference.md. A name from the picture is `--style <name>`; their own colours
  ("green and gold", "my club's colours") become a style file (reference.md, "Their own colours").
- **What size?** "A3, 30 x 42 cm (recommended)", "A4, 21 x 30 cm", "50 x 70 cm".
- Year poster: **What should the lines show?** "How often I ran each street (recommended)", "My pace", "The month",
  "One colour". And **Which area?** "Where I run most (recommended)", "Every city I ran in (an atlas)", or a city
  typed in "Other" ("Valencia, Spain").
- Race or trail from an export (not a single file): **Which sport?** for a race "Run", "Ride", "Walk", "Hike"; for
  a trail "Hike", "Walk", "Trail run", "Run". And **Which one?** "The longest (recommended)", "The latest", or a date
  typed in "Other" (YYYY-MM-DD).

## 4. Words

- Every kind but "For two": **Name on the poster?** "No name", "Yes (type it in Other)". If they pick "Yes" without
  typing one, ask for it in one line.
- Year poster and for two: **Which year?** this year, last year (from today's date), "All my years". If they did not
  say runs, **Which sport?** "Runs", "Rides", "Walks", "Hikes".
- Race: **Title?** "No title", "The race's name (type it in Other)". **Official distance?** "Half marathon",
  "Marathon", "10K", or km typed in "Other": it makes pace and splits match the race, not the GPS.
- Trail: **Title?** "No title", "The route's name (type it in Other)".
- Race or trail from a single GPX, FIT or TCX file: **Does it start or end at a home?** "No, at a trailhead, a car
  park or a station", "Yes, at a home". If yes, add `--private-ends`, and tell them the print will leave out the
  stretch near each end and give numbers for the drawn part only. (A year poster or an atlas needs an export.)

## 5. Make it

Run `"<PINLESS>" "<file>" <flags>` with the flags from the answers (reference.md) and the three outputs in
`~/Pinless` (Pinless makes the folder): `--out "~/Pinless/<name>.png" --pdf "~/Pinless/<name>.pdf" --preview
"~/Pinless/<name>.jpg"`. The name is the title or name plus the style, such as `madrid-2025-carbon`; when a later
version would reuse a name, add `-2`, `-3`. Use a 10-minute timeout, and tell the user the first map of a place
takes a few minutes (its streets are downloaded once). Read what it prints: on `STOP`, a refusal or an error, follow
reference.md. When it works, open the preview with `show.py` and tell them the line it closes with (reference.md
says what each poster prints).

## 6. Look and adjust

Ask **How does it look?** "I love it, done", "Other colours", "Change the words", "Something else". Make the change
and run again (much faster now), keeping each version's files, until they are happy. Do not look at the poster image
yourself unless they ask you to.

## 7. Done

Tell them where the files are and what each is for: the PNG (300 dpi) and the PDF for printing, the JPG to share on a
phone. Print tips: order exactly the size you chose, on matte paper, with no automatic enhancement or crop; an A3 in a
40 x 50 cm frame with a mat looks best. Ask them to look at it closely before printing or sharing it. Offer to open the
folder (`show.py ~/Pinless`).
