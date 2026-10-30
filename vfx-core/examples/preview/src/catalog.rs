//! The effects a preview offers, one at a time. [`crate::run`] takes any list; the
//! repository's preview runs [`repository`]: every effect crate here (through its
//! `VFX_PLUGIN`, the same static its plugins export) and each core effect.
//!
//! The effect crates are listed by the build script, from this crate's dependencies that
//! have `[package.metadata.vfx]`, and the panel, expressions and export all come from an
//! effect's parameter list: a new effect crate needs no code here.

use std::sync::OnceLock;

use vfx_core::Processor;
use vfx_core::effects::{Blend, ColorAdjust, FeedbackTrail, Flip, GaussianBlur, Vignette};

pub struct Entry {
    pub name: String,
    /// How `--effect` and `?effect=` name it.
    pub slug: String,
    pub about: String,
    pub make: fn() -> Processor,
    /// The effect works on premultiplied alpha, so the preview converts its straight
    /// frames before it and back after (GPU effects).
    pub premultiplied: bool,
    /// Starts with a pattern as its second layer, for effects that show nothing without
    /// one. Otherwise the layer starts at None, as a host's layer menu does.
    pub layer_by_default: bool,
}

impl Entry {
    /// An effect crate's plugin: its name, description and effect.
    pub fn plugin(plugin: &'static vfx_host::Plugin) -> Self {
        Self {
            name: plugin.info.name.to_owned(),
            slug: slug(plugin.info.short_name),
            about: plugin.info.description.to_owned(),
            make: plugin.make,
            premultiplied: false,
            layer_by_default: false,
        }
    }

    pub fn new(name: &str, about: &str, make: fn() -> Processor) -> Self {
        Self {
            name: name.to_owned(),
            slug: slug(name),
            about: about.to_owned(),
            make,
            premultiplied: false,
            layer_by_default: false,
        }
    }

    fn premultiplied(mut self) -> Self {
        self.premultiplied = true;
        self
    }

    fn layer_by_default(mut self) -> Self {
        self.layer_by_default = true;
        self
    }
}

// `effect_crates()`: an `Entry::plugin` for each effect crate this crate depends on.
include!(concat!(env!("OUT_DIR"), "/effect_crates.rs"));

/// Every effect in this repository: the effect crates, then the core effects.
pub fn repository() -> Vec<Entry> {
    let mut all = effect_crates();
    all.extend([
        Entry::new(
            "Colour Adjust",
            "Brightness, contrast and saturation.",
            || Processor::Gpu(Box::<ColorAdjust>::default()),
        ),
        Entry::new(
            "Gaussian Blur",
            "Separable blur, horizontal, vertical or both.",
            || Processor::Gpu(Box::<GaussianBlur>::default()),
        )
        .premultiplied(),
        Entry::new(
            "Vignette",
            "Fades towards a colour away from a centre. Click the image to move the centre.",
            || Processor::Gpu(Box::<Vignette>::default()),
        ),
        Entry::new(
            "Blend",
            "Blends a second layer over the input. Pick the layer under Input.",
            || Processor::Gpu(Box::<Blend>::default()),
        )
        .layer_by_default(),
        Entry::new(
            "Flip",
            "Mirrors the frame. Flips when a value is above 0.5.",
            || Processor::Gpu(Box::<Flip>::default()),
        ),
        Entry::new(
            "Feedback Trail",
            "Keeps the brighter of each frame and a fading copy of the last result.",
            || Processor::Gpu(Box::<FeedbackTrail>::default()),
        ),
    ]);
    all
}

static ENTRIES: OnceLock<Vec<Entry>> = OnceLock::new();

/// Sets the catalog, once, before the preview starts.
pub fn set(entries: Vec<Entry>) {
    let _ = ENTRIES.set(entries);
}

/// The catalog: the one [`set`] was given, or the repository's.
pub fn entries() -> &'static [Entry] {
    ENTRIES.get_or_init(repository)
}

/// The index of the effect called `name`.
#[cfg(test)]
pub fn find(name: &str) -> Option<usize> {
    entries().iter().position(|e| e.name == name)
}

/// How effects are named in links and on the command line: runs of letters and digits,
/// lower case, joined by hyphens. "segcam Motion" is `segcam-motion`. `cargo vfx` names
/// them the same way.
pub fn slug(name: &str) -> String {
    let mut out = String::new();
    for c in name.chars() {
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    out.trim_end_matches('-').to_owned()
}

/// The index of the effect a link or `--effect` names, by slug.
pub fn find_slug(slug_text: &str) -> Option<usize> {
    let wanted = slug(slug_text);
    entries()
        .iter()
        .position(|e| e.slug == wanted || slug(&e.name) == wanted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugs_name_effects() {
        assert_eq!(slug("segcam Motion"), "segcam-motion");
        assert_eq!(slug("segcam Motion (Rust)"), "segcam-motion-rust");
        assert!(find_slug("segcam-motion").is_some());
        assert!(find_slug("vfx-preview").is_some());
        assert!(find_slug("gaussian-blur").is_some());
    }

    #[test]
    fn names_are_unique() {
        let all = entries();
        for (i, a) in all.iter().enumerate() {
            assert!(all[i + 1..].iter().all(|b| b.name != a.name), "{}", a.name);
        }
    }
}
