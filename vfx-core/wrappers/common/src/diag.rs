//! One-line notes about how a plugin is rendering: which path (CPU or GPU-native),
//! which bit depths, and any fallback. Each distinct note is written once per session,
//! so the file stays a few lines long.
//!
//! The file is `<plugin file><suffix>`, say `VfxPreview-OpenFX.log`: on macOS in
//! `~/Library/Logs`, elsewhere in the temp directory.

use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

pub struct Log {
    /// After the plugin's file name, say "-OpenFX.log".
    suffix: &'static str,
    seen: Mutex<Option<HashSet<String>>>,
}

impl Log {
    pub const fn new(suffix: &'static str) -> Self {
        Self {
            suffix,
            seen: Mutex::new(None),
        }
    }

    pub fn note(&self, message: &str) {
        let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        if !seen
            .get_or_insert_with(HashSet::new)
            .insert(message.to_owned())
        {
            return;
        }
        drop(seen);
        let Some(path) = self.path() else { return };
        // Logging must never disturb a render, so write errors are ignored.
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(file, "[pid {}] {message}", std::process::id());
        }
    }

    fn path(&self) -> Option<PathBuf> {
        let file = crate::plugin::registered().map_or("vfx", |p| p.info.file);
        let name = format!("{file}{}", self.suffix);
        if cfg!(target_os = "macos") {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Logs").join(name))
        } else {
            Some(std::env::temp_dir().join(name))
        }
    }
}
