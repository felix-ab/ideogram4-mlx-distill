# Prompt notes — Ideogram v4 structured JSON

Ideogram v4 takes a structured JSON caption (see `example_prompt.json`). These notes are
the distilled results of a measured ablation study (fixed seed, field-by-field deltas
against a reference render) on this model family. Effect sizes: ≥0.35 = changes identity,
0.20–0.30 = restages the scene, 0.09–0.15 = regrades without recomposing, ≤0.07 = subtle.

## Field ranking (strongest first)

1. `high_level_description` and background/scene wording (~0.38) — composition lives here.
2. `style_description.aesthetics` — the look's identity sentence.
3. `style_description.color_palette` (~0.40) — an array of UPPERCASE hex strings; acts
   like a production-design brief, not a tint. The single strongest *dial*.
4. `style_description.lighting` — restages light; motivated/named sources work best.
5. `style_description.photo` (~0.13) — camera/film/grain vocabulary; grades more than it
   recomposes.

## Behaviors worth knowing

- **Unknown JSON keys are read at full strength.** Custom keys like `film_emulation` or
  `atmosphere` steer as strongly as (sometimes stronger than) folding the same words into
  prose. Invent dials freely.
- **Fine repeating structures must be NAMED as structures** ("knotted net mesh in a
  criss-cross lattice", "venetian-blind stripes") — topology commits from the
  conditioning at mid-noise; no amount of extra steps builds a lattice you didn't name.
- **Placement**: element bboxes are y-first, ×1000 normalized. Omit for full-frame.
- **Motion reads as alive when placed somewhere specific**: "1/30s shutter drag on the
  skirt hem, eyes stay sharp" beats "dynamic motion blur".
- **Film-scan physicality renders when named**: halation rims, black scan borders, base
  fog — the model knows the medium's physics by name.
- "no text" in aesthetics suppresses incidental garbled signage.

## A five-part recipe for movie-still energy

(1) backlight as the antagonist (low sun/practical *against* the subject), (2) film-scan
physicality named (stock, halation, border), (3) one warm accent inside a cool world
(or vice versa) via the palette array, (4) narrative candidness — motion in named places,
nobody looking at camera, (5) layered occlusion (something soft in the near foreground).
