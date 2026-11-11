# ADR 5: Windows plugins ship as an installer and zips

Status: proposed. Date: 2026-11-10 (first proposed 2026-10-30). Proposed by: Strongtut. Decides: jamecoyne.

## Context

`release.yml` builds an unsigned Windows installer. An unsigned installer makes Windows
SmartScreen warn "unknown publisher" before it runs. The hosts load unsigned plugin files
without a warning. v0.1.0 shipped the unsigned installer.

## Decision

The Windows download is one installer with segcam and the example effects as options, plus
segcam's zips, one per host, each with a readme naming its folder:

| Plugin | Folder |
|---|---|
| OpenFX | `C:\Program Files\Common Files\OFX\Plugins` |
| After Effects | `C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\segcam` |
| Resolume | `Documents\Resolume Arena\Extra Effects` (or Avenue's) |

The DXC files sit next to the plugin; the OpenFX bundle carries them and `DirectML.dll`
inside. The README says how to get past the SmartScreen warning until there is code signing.

## Options considered

| Option | For | Against |
|---|---|---|
| Zips | no warning, no cost | a manual copy, admin rights for two folders |
| Unsigned installer | one click | a SmartScreen warning on every install |
| Signed installer | one click, no warning | a certificate, identity checks and a yearly cost |

## Consequences

- The install docs explain the installer's warning and the copy for each host.
- Code signing becomes a later decision for jamecoyne.
