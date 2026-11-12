//! Values of `FFMixed`, the 64-bit union of an integer and a pointer that a
//! host passes to `plugMain(code, input, instance)` and gets back. A float
//! travels as its bit pattern in the integer.

use std::ffi::c_void;

use crate::sys::{FF_FAIL, FF_SUCCESS, FFMixed};

impl FFMixed {
    pub fn uint(v: u32) -> Self {
        // Zero the whole union first, so the high half of a 64-bit read is defined.
        let mut m = FFMixed {
            PointerValue: std::ptr::null_mut(),
        };
        m.UIntValue = v;
        m
    }

    pub fn float(v: f32) -> Self {
        Self::uint(v.to_bits())
    }

    pub fn ptr<T>(p: *const T) -> Self {
        FFMixed {
            PointerValue: p as *mut c_void,
        }
    }

    pub fn fail() -> Self {
        Self::uint(FF_FAIL)
    }

    pub fn success() -> Self {
        Self::uint(FF_SUCCESS)
    }

    pub fn ok(ok: bool) -> Self {
        if ok { Self::success() } else { Self::fail() }
    }
}
