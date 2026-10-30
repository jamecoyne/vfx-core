//! How an effect's parameters are laid out as a host's controls: which a host can serve,
//! which sit in a group, and where the layer picker goes. Each host target walks the layout
//! and makes its own controls, so the rules live here, once, without any host's API.

use vfx_core::{ParamSpec, Requires};

use crate::served;

/// One control, or a group of them, in host order.
#[derive(Clone, Debug, PartialEq)]
pub enum Item<'a> {
    Control(&'a ParamSpec),
    /// The second layer's picker, for hosts that show one.
    Layer,
    Group {
        name: &'a str,
        items: Vec<Item<'a>>,
    },
}

/// What a host offers.
#[derive(Clone, Copy, Debug, Default)]
pub struct Host<'a> {
    /// The host can hand over a second layer.
    pub layer: bool,
    /// The host lets an effect draw outside the frame.
    pub region: bool,
    /// The layer picker's name, for hosts that show one as a control (After Effects).
    pub layer_picker: Option<&'a str>,
    pub groups: Groups,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Groups {
    /// Every control at the top level.
    #[default]
    None,
    Shown,
    /// Groups, except one named like a control or the layer picker. For hosts whose scripts
    /// find controls by name (After Effects), where the group would hide the control.
    UnlessNamedLikeAControl,
}

/// The controls for `specs` in `host`, in order: those it can serve, consecutive ones of a
/// group in that group, and the layer picker before the first control that needs a layer
/// (or last, if none does).
pub fn layout<'a>(specs: &'a [ParamSpec], host: &Host) -> Vec<Item<'a>> {
    let served: Vec<&ParamSpec> = specs
        .iter()
        .filter(|s| served(s, host.layer, host.region))
        .collect();
    let shows = |group: &str| match host.groups {
        Groups::None => false,
        Groups::Shown => !group.is_empty(),
        Groups::UnlessNamedLikeAControl => {
            !group.is_empty()
                && host.layer_picker != Some(group)
                && !served.iter().any(|s| s.display_label() == group)
        }
    };
    let mut picker = host.layer_picker.is_some();
    let mut items = Vec::new();
    let mut i = 0;
    while i < served.len() {
        let group = served[i].group;
        let end = served[i..]
            .iter()
            .position(|s| s.group != group)
            .map_or(served.len(), |n| i + n);
        let mut run = Vec::new();
        for spec in &served[i..end] {
            if picker && spec.requires == Requires::Layer {
                run.push(Item::Layer);
                picker = false;
            }
            run.push(Item::Control(spec));
        }
        if shows(group) {
            items.push(Item::Group {
                name: group,
                items: run,
            });
        } else {
            items.extend(run);
        }
        i = end;
    }
    if picker {
        items.push(Item::Layer);
    }
    items
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPECS: &[ParamSpec] = &[
        ParamSpec::toggle("linear", false).label("Linear Light"),
        ParamSpec::float("brightness", -1.0, 1.0, 0.0).group("Color"),
        ParamSpec::float("contrast", 0.0, 2.0, 1.0).group("Color"),
        ParamSpec::toggle("vignette", false)
            .label("Vignette")
            .group("Vignette"),
        ParamSpec::float("radius", 0.0, 1.0, 0.5).group("Vignette"),
        ParamSpec::float("blend_amount", 0.0, 1.0, 1.0)
            .group("Blend")
            .requires(Requires::Layer),
        ParamSpec::toggle("grow", false).requires(Requires::Region),
    ];

    fn names(items: &[Item]) -> Vec<String> {
        items
            .iter()
            .map(|item| match item {
                Item::Control(s) => s.name.to_owned(),
                Item::Layer => "<layer>".into(),
                Item::Group { name, items } => format!("{name}[{}]", names(items).join(" ")),
            })
            .collect()
    }

    #[test]
    fn after_effects_keeps_controls_reachable_by_name() {
        let host = Host {
            layer: true,
            region: true,
            layer_picker: Some("Blend Layer"),
            groups: Groups::UnlessNamedLikeAControl,
        };
        assert_eq!(
            names(&layout(SPECS, &host)),
            [
                "linear",
                "Color[brightness contrast]",
                "vignette",
                "radius",
                "Blend[<layer> blend_amount]",
                "grow"
            ]
        );
    }

    #[test]
    fn hosts_without_a_layer_or_region_leave_those_out() {
        let host = Host {
            groups: Groups::Shown,
            ..Host::default()
        };
        assert_eq!(
            names(&layout(SPECS, &host)),
            [
                "linear",
                "Color[brightness contrast]",
                "Vignette[vignette radius]"
            ]
        );
        assert_eq!(layout(SPECS, &Host::default()).len(), 5, "flat");
    }

    #[test]
    fn the_picker_comes_last_when_nothing_needs_it() {
        let host = Host {
            layer: true,
            layer_picker: Some("Layer"),
            ..Host::default()
        };
        let items = layout(&SPECS[..2], &host);
        assert_eq!(names(&items), ["linear", "brightness", "<layer>"]);
    }
}
