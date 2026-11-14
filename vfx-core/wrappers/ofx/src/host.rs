//! The host's suites, fetched once at load, and small helpers over the property suite.

use std::ffi::{CStr, CString, c_char, c_double, c_int, c_void};
use std::ptr;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicPtr, Ordering};

use crate::sys::*;

pub struct Suites {
    pub prop: &'static OfxPropertySuiteV1,
    pub effect: &'static OfxImageEffectSuiteV1,
    pub param: &'static OfxParameterSuiteV1,
}

static HOST: AtomicPtr<OfxHost> = AtomicPtr::new(ptr::null_mut());
static SUITES: OnceLock<Suites> = OnceLock::new();

/// Records the host passed before loading the plugin.
///
/// # Safety
/// `host` and its suites must remain valid until the plugin unloads.
pub unsafe extern "C" fn set_host(host: *mut OfxHost) {
    HOST.store(host, Ordering::Release);
}

/// Whether the host is DaVinci Resolve, from the name it gives itself.
pub fn is_resolve() -> bool {
    let host = HOST.load(Ordering::Acquire);
    if host.is_null() {
        return false;
    }
    // SAFETY: the host pointer stays valid while the plugin is loaded.
    let Some(fetch_suite) = (unsafe { (*host).fetchSuite }) else {
        return false;
    };
    // SAFETY: the host pointer and its property set stay valid while the plugin is loaded.
    let props = unsafe { fetch_suite((*host).host, kOfxPropertySuite.as_ptr(), 1) }
        .cast::<OfxPropertySuiteV1>();
    if props.is_null() {
        return false;
    }
    // SAFETY: a property suite of the requested version, valid while the plugin is loaded.
    let Some(get_string) = (unsafe { (*props).propGetString }) else {
        return false;
    };
    let mut name: *mut c_char = ptr::null_mut();
    // SAFETY: the host's property set handle; the host writes one string pointer.
    let named = unsafe { get_string((*host).host, kOfxPropName.as_ptr(), 0, &mut name) }
        == kOfxStatOK
        && !name.is_null();
    // SAFETY: a NUL-terminated string the host owns, checked non-null above.
    named
        && unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .to_lowercase()
            .contains("resolve")
}

/// Fetches the suites. Called from the load action; false if the host lacks one.
pub fn fetch_suites() -> bool {
    let host = HOST.load(Ordering::Acquire);
    if host.is_null() {
        return false;
    }
    // SAFETY: the host pointer stays valid while the plugin is loaded.
    let Some(fetch_suite) = (unsafe { (*host).fetchSuite }) else {
        return false;
    };
    let fetch = |name: &CStr| -> *const c_void {
        // SAFETY: the host pointer stays valid while the plugin is loaded.
        unsafe { fetch_suite((*host).host, name.as_ptr(), 1) }
    };
    let (prop, effect, param) = (
        fetch(kOfxPropertySuite),
        fetch(kOfxImageEffectSuite),
        fetch(kOfxParameterSuite),
    );
    if prop.is_null() || effect.is_null() || param.is_null() {
        return false;
    }
    // SAFETY: the host returns suites of the requested version, valid while loaded.
    let _ = SUITES.set(unsafe {
        Suites {
            prop: &*prop.cast(),
            effect: &*effect.cast(),
            param: &*param.cast(),
        }
    });
    true
}

pub fn suites() -> &'static Suites {
    SUITES.get().expect("suites are fetched in the load action")
}

/// The function a suite field holds, or `kOfxStatErrMissingHostFeature` when the host
/// left the field null.
pub fn required<F>(function: Option<F>) -> Result<F, OfxStatus> {
    function.ok_or(kOfxStatErrMissingHostFeature)
}

/// Turns a host status into a Result, so `?` can carry failures up to the action.
pub fn check_status(status: OfxStatus) -> Result<(), OfxStatus> {
    if status == kOfxStatOK {
        Ok(())
    } else {
        Err(status)
    }
}

/// A property set handle with typed accessors.
#[derive(Clone, Copy)]
pub struct Props(pub OfxPropertySetHandle);

impl Props {
    fn suite() -> &'static OfxPropertySuiteV1 {
        suites().prop
    }

    pub fn set_str(&self, name: &CStr, index: usize, value: &CStr) -> Result<(), OfxStatus> {
        // SAFETY: valid handle and NUL-terminated strings; the host copies the value.
        check_status(unsafe {
            (required(Self::suite().propSetString)?)(
                self.0,
                name.as_ptr(),
                index as c_int,
                value.as_ptr(),
            )
        })
    }

    pub fn set_strs(&self, name: &CStr, values: &[&CStr]) -> Result<(), OfxStatus> {
        for (i, v) in values.iter().enumerate() {
            self.set_str(name, i, v)?;
        }
        Ok(())
    }

    /// Sets a string built at run time, such as a label.
    pub fn set_string(&self, name: &CStr, value: &str) -> Result<(), OfxStatus> {
        let value = CString::new(value).map_err(|_| kOfxStatFailed)?;
        self.set_str(name, 0, &value)
    }

    pub fn set_int(&self, name: &CStr, index: usize, value: i32) -> Result<(), OfxStatus> {
        // SAFETY: valid handle and property name.
        check_status(unsafe {
            (required(Self::suite().propSetInt)?)(self.0, name.as_ptr(), index as c_int, value)
        })
    }

    pub fn set_double(&self, name: &CStr, index: usize, value: f64) -> Result<(), OfxStatus> {
        // SAFETY: valid handle and property name.
        check_status(unsafe {
            (required(Self::suite().propSetDouble)?)(
                self.0,
                name.as_ptr(),
                index as c_int,
                value as c_double,
            )
        })
    }

    pub fn set_doubles(&self, name: &CStr, values: &[f64]) -> Result<(), OfxStatus> {
        for (i, v) in values.iter().enumerate() {
            self.set_double(name, i, *v)?;
        }
        Ok(())
    }

    pub fn get_int(&self, name: &CStr, index: usize) -> Result<i32, OfxStatus> {
        let mut v = 0;
        // SAFETY: valid handle; the host writes one int.
        check_status(unsafe {
            (required(Self::suite().propGetInt)?)(self.0, name.as_ptr(), index as c_int, &mut v)
        })?;
        Ok(v)
    }

    pub fn get_double(&self, name: &CStr, index: usize) -> Result<f64, OfxStatus> {
        let mut v = 0.0;
        // SAFETY: valid handle; the host writes one double.
        check_status(unsafe {
            (required(Self::suite().propGetDouble)?)(self.0, name.as_ptr(), index as c_int, &mut v)
        })?;
        Ok(v)
    }

    pub fn get_pointer(&self, name: &CStr) -> Result<*mut c_void, OfxStatus> {
        let mut v = ptr::null_mut();
        // SAFETY: valid handle; the host writes one pointer.
        check_status(unsafe {
            (required(Self::suite().propGetPointer)?)(self.0, name.as_ptr(), 0, &mut v)
        })?;
        Ok(v)
    }

    /// Compares a string property with `expected`.
    pub fn str_is(&self, name: &CStr, expected: &CStr) -> Result<bool, OfxStatus> {
        let mut v: *mut c_char = ptr::null_mut();
        // SAFETY: valid handle; the host returns a string it owns.
        check_status(unsafe {
            (required(Self::suite().propGetString)?)(self.0, name.as_ptr(), 0, &mut v)
        })?;
        // SAFETY: a NUL-terminated string the host keeps alive for this call.
        Ok(!v.is_null() && unsafe { CStr::from_ptr(v) } == expected)
    }

    pub fn get_rect_i(&self, name: &CStr) -> Result<crate::image::Rect, OfxStatus> {
        Ok(crate::image::Rect {
            x1: self.get_int(name, 0)?,
            y1: self.get_int(name, 1)?,
            x2: self.get_int(name, 2)?,
            y2: self.get_int(name, 3)?,
        })
    }

    pub fn get_rect_d(&self, name: &CStr) -> Result<[f64; 4], OfxStatus> {
        Ok([
            self.get_double(name, 0)?,
            self.get_double(name, 1)?,
            self.get_double(name, 2)?,
            self.get_double(name, 3)?,
        ])
    }
}
