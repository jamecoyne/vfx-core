# Notes for agents working on this repo

People start at [README.md](README.md); the developer guide is [docs/DEVELOPING.md](docs/DEVELOPING.md).

- Working on segcam: read [`plugins/segcam/SPEC.md`](plugins/segcam/SPEC.md) first, the 1.0
  requirements with their status. Decisions are in `plugins/segcam/decisions/`, the work order in
  `ROADMAP.md`, and speed numbers follow `BENCHMARK.md`, all in the same folder.
- Pull requests: small, one topic each. Request review from jamecoyne, and from whoever wrote the
  code you change (`git log` on its files). A PR that touches speed adds the benchmark rows for the
  hosts it changes; one that finishes a segcam requirement changes its status in SPEC.md.
- Effect logic belongs in an effect crate (`vfx-core/effects/`, `plugins/`) or the core
  (`vfx-core/src/effects`). A target crate (`vfx-core/wrappers/`) only translates parameters
  and moves pixels, for any effect; what two targets share belongs in
  `vfx-core/wrappers/common`. Nothing in `vfx-core/` depends on `plugins/` except the
  preview's list of effects.
- Read the README in each target or effect (`vfx-core/wrappers/ffgl/`,
  `vfx-core/wrappers/adobe/ae-tests/`, `plugins/segcam/segcam-motion/`) before changing it, and run its harness afterwards. The After
  Effects harnesses test the *installed* plugins, so `cargo vfx package` and reinstall
  first (installing to system folders needs the user's password).
- Keep labels under 16 bytes: FFGL hosts cut longer ones, and each effect crate has a test.
  Never rename a parameter's `name` once shipped; change its `label` instead.
- A new effect crate: `cargo vfx new NAME`. An effect crate the preview's `Cargo.toml`
  depends on shows up in it with no code; `cargo vfx check` reports one it doesn't.
- `cargo fmt --all -- --check` and clippy with `-D warnings` (see `ci.yml` for the crates
  and features) must pass; CI runs latest stable, whose clippy can be stricter.
- wgpu is pinned to `=30.0.1`; the Metal interop uses `wgpu-hal` types and must be checked
  when it moves. The web build needs wgpu's `fragile-send-sync-non-atomic-wasm` feature
  (set in `vfx-core/examples/preview/Cargo.toml`) because effects are `Send`.
