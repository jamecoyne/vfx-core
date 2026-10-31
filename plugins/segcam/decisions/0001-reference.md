# ADR 1: segcam 1.1 is the reference

Status: proposed. Date: 2026-10-30. Proposed by: Strongtut. Decides: jamecoyne.

## Context

Three versions of segcam exist: jamecoyne's macOS app (segcam 1.1), jamecoyne's C++ After Effects effect, and
the Rust port in this repo, written line for line from the C++ effect. Without one reference, each
change gets checked against whichever version its author had at hand, and they disagree in places.

## Decision

segcam 1.1 is the reference for look and behaviour. Where an editing host can't follow the app,
because it renders frames in any order and keeps no state, the C++ effect's rules apply: output
depends only on the frames and the settings, and window IDs are per frame. The Rust code implements
both and is never the reference itself.

## Options considered

| Option | For | Against |
|---|---|---|
| segcam 1.1 | the effect people know and love; jamecoyne's own design | keeps state, which editors can't |
| The C++ effect | already the check in this repo; built for editors | differs from the app in places |
| The Rust port | no second program to compare against | a bug in the port would count as correct, since nothing outside it says otherwise |

## Consequences

- Behaviour changes are judged by "closer to 1.1, still correct in editors" (SPEC section 6).
- Known differences get listed here for jamecoyne. One today: the app converts to luma with integers,
  the C++ effect with floats. Proposed: floats in the plugins, as the C++ effect does.
