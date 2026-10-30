//! The one effect a plugin binary holds, registered by its `export!` macro.

use std::sync::OnceLock;

use vfx_core::{Gpu, ParamKind, ParamSpec, ParamValue, ParamValues, Processor, Requires};

use crate::diag::Log;

/// Who a plugin is in every host, from `[package.metadata.vfx]`.
#[derive(Clone, Copy, Debug)]
pub struct Info {
    /// In each host's effect list.
    pub name: &'static str,
    /// Where a host allows only 16 bytes (FFGL); `name` unless the crate sets one.
    pub short_name: &'static str,
    /// After Effects category and OpenFX grouping.
    pub category: &'static str,
    pub description: &'static str,
    /// Bundles, DLLs, installers and logs: no spaces.
    pub file: &'static str,
    /// OpenFX identifier; macOS bundle IDs start with it.
    pub id: &'static str,
    pub ae_match_name: &'static str,
    /// FFGL plugin code, exactly 4 characters.
    pub ffgl_code: &'static str,
    pub version: [u32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Gpu,
    Cpu,
}

/// A plugin: its identity and how to make an instance of its effect.
pub struct Plugin {
    pub info: Info,
    pub kind: Kind,
    pub make: fn() -> Processor,
    outline: OnceLock<Outline>,
}

/// What hosts ask about an effect before rendering anything, from one instance made for
/// the purpose.
pub struct Outline {
    pub params: Vec<ParamSpec>,
    pub layers: usize,
    pub layer_name: String,
}

impl Plugin {
    pub const fn new(info: Info, kind: Kind, make: fn() -> Processor) -> Self {
        Self {
            info,
            kind,
            make,
            outline: OnceLock::new(),
        }
    }

    pub fn outline(&self) -> &Outline {
        self.outline.get_or_init(|| {
            let processor = (self.make)();
            let (layers, layer_name) = match &processor {
                Processor::Gpu(e) => (e.layer_inputs(), e.layer_name().to_owned()),
                Processor::Cpu(_) => (0, String::new()),
            };
            Outline {
                params: processor.params().to_vec(),
                layers,
                layer_name,
            }
        })
    }

    pub fn params(&self) -> &[ParamSpec] {
        &self.outline().params
    }

    /// Default values for every parameter.
    pub fn defaults(&self) -> ParamValues {
        ParamValues::new(self.params())
    }
}

static PLUGIN: OnceLock<&'static Plugin> = OnceLock::new();

/// Called by the export macros at every host entry point; the first call wins.
pub fn register(plugin: &'static Plugin) {
    let _ = PLUGIN.set(plugin);
}

/// The registered plugin.
pub fn plugin() -> &'static Plugin {
    PLUGIN
        .get()
        .expect("the export macro registers the plugin at every entry point")
}

pub(crate) fn registered() -> Option<&'static Plugin> {
    PLUGIN.get().copied()
}

/// One device for the whole plugin, created on first use. `None` if no GPU adapter could
/// be opened; renders then pass the image through rather than fail.
pub fn gpu(log: &Log) -> Option<&'static Gpu> {
    static GPU: OnceLock<Option<Gpu>> = OnceLock::new();
    GPU.get_or_init(|| match pollster::block_on(Gpu::headless()) {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            log.note(&format!(
                "warning: no GPU available, passing frames through: {e}"
            ));
            None
        }
    })
    .as_ref()
}

/// Whether a host can serve a parameter: one that needs a second layer only where there
/// is one, one that grows the output only where the host allows it.
pub fn served(spec: &ParamSpec, layer: bool, region: bool) -> bool {
    match spec.requires {
        Requires::Nothing => true,
        Requires::Layer => layer,
        Requires::Region => region,
    }
}

/// Rewrites every point parameter with `f`, from host coordinates to the normalised canvas
/// coordinates effects take.
pub fn map_points(values: &mut ParamValues, specs: &[ParamSpec], f: impl Fn([f32; 2]) -> [f32; 2]) {
    for spec in specs {
        if matches!(spec.kind, ParamKind::Point { .. })
            && let Some(ParamValue::Point(p)) = values.get(spec.name)
        {
            let _ = values.set(spec.name, ParamValue::Point(f(p)));
        }
    }
}

/// Compares strings at compile time, so an export macro can check its effect's kind
/// against the crate's metadata.
pub const fn same(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requirements_and_points() {
        let layer = ParamSpec::float("amount", 0.0, 1.0, 1.0).requires(Requires::Layer);
        assert!(!served(&layer, false, true) && served(&layer, true, false));
        let specs = [
            ParamSpec::point("centre", [0.5, 0.5]),
            ParamSpec::float("x", 0.0, 1.0, 0.5),
        ];
        let mut values = ParamValues::new(&specs);
        map_points(&mut values, &specs, |[x, y]| [x * 2.0, 1.0 - y]);
        assert_eq!(values.point("centre"), [1.0, 0.5]);
        assert_eq!(values.float("x"), 0.5);
        assert!(same("gpu", "gpu") && !same("gpu", "cpu"));
    }
}
