# ADR 5: Windows plugins ship as zips

Status: proposed. Date: 2026-10-30. Decides: jamecoyne.

## Context

`release.yml` builds an unsigned Windows installer and a zip of the same files. An unsigned
installer makes Windows SmartScreen warn "unknown publisher" before it runs. The hosts load
unsigned plugin files without a warning.

## Decision

For 1.0, the Windows download is the zip, one per plugin, each with a readme naming its folder:

| Plugin | Folder |
|---|---|
| OpenFX | `C:\Program Files\Common Files\OFX\Plugins` |
| After Effects | `C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore` |
| Resolume | `Documents\Resolume Arena\Extra Effects` (or Avenue's) |

`DirectML.dll` sits next to the plugin. The installer comes back once there is code signing.

## Options considered

| Option | For | Against |
|---|---|---|
| Zips | no warning, no cost | a manual copy, admin rights for two folders |
| Unsigned installer | one click | a SmartScreen warning on every install |
| Signed installer | one click, no warning | a certificate, identity checks and a yearly cost |

## Consequences

- The install docs explain the copy for each host.
- Code signing becomes a later decision for jamecoyne.
