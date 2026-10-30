//! The parts of the OpenFX 1.4 C API this plugin uses, transcribed from `ofxCore.h`,
//! `ofxProperty.h`, `ofxImageEffect.h` and `ofxParam.h`. Field order in the suites must
//! match the headers exactly: hosts hand over tables of function pointers.

#![allow(non_snake_case)]

use std::ffi::{CStr, c_char, c_double, c_int, c_uint, c_void};

pub type OfxStatus = c_int;
pub type OfxTime = c_double;
pub type PropSet = *mut c_void;
pub type ImageEffect = *mut c_void;
pub type Clip = *mut c_void;
pub type ParamSet = *mut c_void;
pub type Param = *mut c_void;

pub const STAT_OK: OfxStatus = 0;
pub const STAT_FAILED: OfxStatus = 1;
pub const STAT_ERR_MISSING_HOST_FEATURE: OfxStatus = 4;
pub const STAT_REPLY_DEFAULT: OfxStatus = 14;
pub const STAT_ERR_IMAGE_FORMAT: OfxStatus = 1000;

#[repr(C)]
pub struct OfxHost {
    pub host: PropSet,
    pub fetchSuite:
        unsafe extern "C" fn(host: PropSet, name: *const c_char, version: c_int) -> *const c_void,
}

pub type EntryPoint = unsafe extern "C" fn(
    action: *const c_char,
    handle: *const c_void,
    in_args: PropSet,
    out_args: PropSet,
) -> OfxStatus;

#[repr(C)]
pub struct OfxPlugin {
    pub pluginApi: *const c_char,
    pub apiVersion: c_int,
    pub pluginIdentifier: *const c_char,
    pub pluginVersionMajor: c_uint,
    pub pluginVersionMinor: c_uint,
    pub setHost: unsafe extern "C" fn(host: *mut OfxHost),
    pub mainEntry: EntryPoint,
}

// The struct only holds pointers to static strings and functions.
unsafe impl Sync for OfxPlugin {}

#[repr(C)]
pub struct PropertySuiteV1 {
    pub propSetPointer:
        unsafe extern "C" fn(PropSet, *const c_char, c_int, *mut c_void) -> OfxStatus,
    pub propSetString:
        unsafe extern "C" fn(PropSet, *const c_char, c_int, *const c_char) -> OfxStatus,
    pub propSetDouble: unsafe extern "C" fn(PropSet, *const c_char, c_int, c_double) -> OfxStatus,
    pub propSetInt: unsafe extern "C" fn(PropSet, *const c_char, c_int, c_int) -> OfxStatus,
    pub propSetPointerN:
        unsafe extern "C" fn(PropSet, *const c_char, c_int, *const *mut c_void) -> OfxStatus,
    pub propSetStringN:
        unsafe extern "C" fn(PropSet, *const c_char, c_int, *const *const c_char) -> OfxStatus,
    pub propSetDoubleN:
        unsafe extern "C" fn(PropSet, *const c_char, c_int, *const c_double) -> OfxStatus,
    pub propSetIntN: unsafe extern "C" fn(PropSet, *const c_char, c_int, *const c_int) -> OfxStatus,
    pub propGetPointer:
        unsafe extern "C" fn(PropSet, *const c_char, c_int, *mut *mut c_void) -> OfxStatus,
    pub propGetString:
        unsafe extern "C" fn(PropSet, *const c_char, c_int, *mut *mut c_char) -> OfxStatus,
    pub propGetDouble:
        unsafe extern "C" fn(PropSet, *const c_char, c_int, *mut c_double) -> OfxStatus,
    pub propGetInt: unsafe extern "C" fn(PropSet, *const c_char, c_int, *mut c_int) -> OfxStatus,
    pub propGetPointerN:
        unsafe extern "C" fn(PropSet, *const c_char, c_int, *mut *mut c_void) -> OfxStatus,
    pub propGetStringN:
        unsafe extern "C" fn(PropSet, *const c_char, c_int, *mut *mut c_char) -> OfxStatus,
    pub propGetDoubleN:
        unsafe extern "C" fn(PropSet, *const c_char, c_int, *mut c_double) -> OfxStatus,
    pub propGetIntN: unsafe extern "C" fn(PropSet, *const c_char, c_int, *mut c_int) -> OfxStatus,
    pub propReset: unsafe extern "C" fn(PropSet, *const c_char) -> OfxStatus,
    pub propGetDimension: unsafe extern "C" fn(PropSet, *const c_char, *mut c_int) -> OfxStatus,
}

#[repr(C)]
pub struct ImageEffectSuiteV1 {
    pub getPropertySet: unsafe extern "C" fn(ImageEffect, *mut PropSet) -> OfxStatus,
    pub getParamSet: unsafe extern "C" fn(ImageEffect, *mut ParamSet) -> OfxStatus,
    pub clipDefine: unsafe extern "C" fn(ImageEffect, *const c_char, *mut PropSet) -> OfxStatus,
    pub clipGetHandle:
        unsafe extern "C" fn(ImageEffect, *const c_char, *mut Clip, *mut PropSet) -> OfxStatus,
    pub clipGetPropertySet: unsafe extern "C" fn(Clip, *mut PropSet) -> OfxStatus,
    pub clipGetImage: unsafe extern "C" fn(Clip, OfxTime, *const c_void, *mut PropSet) -> OfxStatus,
    pub clipReleaseImage: unsafe extern "C" fn(PropSet) -> OfxStatus,
    pub clipGetRegionOfDefinition:
        unsafe extern "C" fn(Clip, OfxTime, *mut [c_double; 4]) -> OfxStatus,
    pub abort: unsafe extern "C" fn(ImageEffect) -> c_int,
    pub imageMemoryAlloc: unsafe extern "C" fn(ImageEffect, usize, *mut *mut c_void) -> OfxStatus,
    pub imageMemoryFree: unsafe extern "C" fn(*mut c_void) -> OfxStatus,
    pub imageMemoryLock: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> OfxStatus,
    pub imageMemoryUnlock: unsafe extern "C" fn(*mut c_void) -> OfxStatus,
}

#[repr(C)]
pub struct ParameterSuiteV1 {
    pub paramDefine:
        unsafe extern "C" fn(ParamSet, *const c_char, *const c_char, *mut PropSet) -> OfxStatus,
    pub paramGetHandle:
        unsafe extern "C" fn(ParamSet, *const c_char, *mut Param, *mut PropSet) -> OfxStatus,
    pub paramSetGetPropertySet: unsafe extern "C" fn(ParamSet, *mut PropSet) -> OfxStatus,
    pub paramGetPropertySet: unsafe extern "C" fn(Param, *mut PropSet) -> OfxStatus,
    pub paramGetValue: unsafe extern "C" fn(Param, ...) -> OfxStatus,
    pub paramGetValueAtTime: unsafe extern "C" fn(Param, OfxTime, ...) -> OfxStatus,
    pub paramGetDerivative: unsafe extern "C" fn(Param, OfxTime, ...) -> OfxStatus,
    pub paramGetIntegral: unsafe extern "C" fn(Param, OfxTime, OfxTime, ...) -> OfxStatus,
    pub paramSetValue: unsafe extern "C" fn(Param, ...) -> OfxStatus,
    pub paramSetValueAtTime: unsafe extern "C" fn(Param, OfxTime, ...) -> OfxStatus,
    pub paramGetNumKeys: unsafe extern "C" fn(Param, *mut c_uint) -> OfxStatus,
    pub paramGetKeyTime: unsafe extern "C" fn(Param, c_uint, *mut OfxTime) -> OfxStatus,
    pub paramGetKeyIndex: unsafe extern "C" fn(Param, OfxTime, c_int, *mut c_int) -> OfxStatus,
    pub paramDeleteKey: unsafe extern "C" fn(Param, OfxTime) -> OfxStatus,
    pub paramDeleteAllKeys: unsafe extern "C" fn(Param) -> OfxStatus,
    pub paramCopy: unsafe extern "C" fn(Param, Param, OfxTime, *const c_void) -> OfxStatus,
    pub paramEditBegin: unsafe extern "C" fn(ParamSet, *const c_char) -> OfxStatus,
    pub paramEditEnd: unsafe extern "C" fn(ParamSet) -> OfxStatus,
}

// Suite names.
pub const PROPERTY_SUITE: &CStr = c"OfxPropertySuite";
pub const IMAGE_EFFECT_SUITE: &CStr = c"OfxImageEffectSuite";
pub const PARAMETER_SUITE: &CStr = c"OfxParameterSuite";

// Actions.
pub const ACTION_LOAD: &CStr = c"OfxActionLoad";
pub const ACTION_DESCRIBE: &CStr = c"OfxActionDescribe";
pub const ACTION_UNLOAD: &CStr = c"OfxActionUnload";
pub const ACTION_CREATE_INSTANCE: &CStr = c"OfxActionCreateInstance";
pub const ACTION_DESTROY_INSTANCE: &CStr = c"OfxActionDestroyInstance";
pub const ACTION_DESCRIBE_IN_CONTEXT: &CStr = c"OfxImageEffectActionDescribeInContext";
pub const ACTION_GET_ROD: &CStr = c"OfxImageEffectActionGetRegionOfDefinition";
pub const ACTION_GET_ROIS: &CStr = c"OfxImageEffectActionGetRegionsOfInterest";
pub const ACTION_RENDER: &CStr = c"OfxImageEffectActionRender";
pub const ACTION_GET_FRAMES_NEEDED: &CStr = c"OfxImageEffectActionGetFramesNeeded";

// Plugin and contexts.
pub const IMAGE_EFFECT_PLUGIN_API: &CStr = c"OfxImageEffectPluginAPI";
pub const CONTEXT_FILTER: &CStr = c"OfxImageEffectContextFilter";
pub const CONTEXT_GENERAL: &CStr = c"OfxImageEffectContextGeneral";

// Generic properties.
pub const PROP_LABEL: &CStr = c"OfxPropLabel";
pub const PROP_PLUGIN_DESCRIPTION: &CStr = c"OfxPropPluginDescription";
pub const PROP_TIME: &CStr = c"OfxPropTime";

// Image effect properties.
pub const PROP_GROUPING: &CStr = c"OfxImageEffectPluginPropGrouping";
pub const PROP_SUPPORTED_CONTEXTS: &CStr = c"OfxImageEffectPropSupportedContexts";
pub const PROP_SUPPORTED_PIXEL_DEPTHS: &CStr = c"OfxImageEffectPropSupportedPixelDepths";
pub const PROP_SUPPORTED_COMPONENTS: &CStr = c"OfxImageEffectPropSupportedComponents";
pub const PROP_SUPPORTS_MULTI_RESOLUTION: &CStr = c"OfxImageEffectPropSupportsMultiResolution";
pub const PROP_SUPPORTS_TILES: &CStr = c"OfxImageEffectPropSupportsTiles";
pub const PROP_TEMPORAL_CLIP_ACCESS: &CStr = c"OfxImageEffectPropTemporalClipAccess";
pub const PROP_MULTIPLE_CLIP_DEPTHS: &CStr = c"OfxImageEffectPropMultipleClipDepths";
pub const PROP_RENDER_THREAD_SAFETY: &CStr = c"OfxImageEffectPluginRenderThreadSafety";
pub const RENDER_FULLY_SAFE: &CStr = c"OfxImageEffectRenderFullySafe";
pub const RENDER_INSTANCE_SAFE: &CStr = c"OfxImageEffectRenderInstanceSafe";
pub const PROP_INSTANCE_DATA: &CStr = c"OfxPropInstanceData";
pub const PROP_HOST_FRAME_THREADING: &CStr = c"OfxImageEffectPluginPropHostFrameThreading";
pub const PROP_CONTEXT: &CStr = c"OfxImageEffectPropContext";
pub const PROP_RENDER_WINDOW: &CStr = c"OfxImageEffectPropRenderWindow";
pub const PROP_RENDER_SCALE: &CStr = c"OfxImageEffectPropRenderScale";
pub const PROP_FRAME_RATE: &CStr = c"OfxImageEffectPropFrameRate";
pub const PROP_FRAME_RANGE: &CStr = c"OfxImageEffectPropFrameRange";
pub const PROP_ROD: &CStr = c"OfxImageEffectPropRegionOfDefinition";
pub const PROP_ROI: &CStr = c"OfxImageEffectPropRegionOfInterest";
pub const PROP_PREMULTIPLICATION: &CStr = c"OfxImageEffectPropPreMultiplication";
pub const PROP_PIXEL_DEPTH: &CStr = c"OfxImageEffectPropPixelDepth";
pub const PROP_COMPONENTS: &CStr = c"OfxImageEffectPropComponents";
pub const PREMULTIPLIED: &CStr = c"OfxImageAlphaPremultiplied";

// Clips and images.
pub const CLIP_OUTPUT: &CStr = c"Output";
pub const CLIP_SOURCE: &CStr = c"Source";
pub const CLIP_LAYER: &CStr = c"Layer";
pub const CLIP_ROI_SOURCE: &CStr = c"OfxImageClipPropRoI_Source";
pub const CLIP_ROI_LAYER: &CStr = c"OfxImageClipPropRoI_Layer";
pub const CLIP_FRAME_RANGE_SOURCE: &CStr = c"OfxImageClipPropFrameRange_Source";
pub const CLIP_PROP_OPTIONAL: &CStr = c"OfxImageClipPropOptional";
pub const CLIP_PROP_CONNECTED: &CStr = c"OfxImageClipPropConnected";
pub const IMAGE_PROP_DATA: &CStr = c"OfxImagePropData";
pub const IMAGE_PROP_BOUNDS: &CStr = c"OfxImagePropBounds";
pub const IMAGE_PROP_ROW_BYTES: &CStr = c"OfxImagePropRowBytes";
pub const COMPONENT_RGBA: &CStr = c"OfxImageComponentRGBA";
pub const DEPTH_BYTE: &CStr = c"OfxBitDepthByte";
pub const DEPTH_SHORT: &CStr = c"OfxBitDepthShort";
pub const DEPTH_FLOAT: &CStr = c"OfxBitDepthFloat";

// Parameters.
pub const PARAM_TYPE_DOUBLE: &CStr = c"OfxParamTypeDouble";
pub const PARAM_TYPE_INTEGER: &CStr = c"OfxParamTypeInteger";
pub const PARAM_TYPE_BOOLEAN: &CStr = c"OfxParamTypeBoolean";
pub const PARAM_TYPE_CHOICE: &CStr = c"OfxParamTypeChoice";
pub const PARAM_TYPE_RGBA: &CStr = c"OfxParamTypeRGBA";
pub const PARAM_TYPE_DOUBLE_2D: &CStr = c"OfxParamTypeDouble2D";
pub const PARAM_TYPE_GROUP: &CStr = c"OfxParamTypeGroup";
pub const PARAM_PROP_DEFAULT: &CStr = c"OfxParamPropDefault";
pub const PARAM_PROP_MIN: &CStr = c"OfxParamPropMin";
pub const PARAM_PROP_MAX: &CStr = c"OfxParamPropMax";
pub const PARAM_PROP_DISPLAY_MIN: &CStr = c"OfxParamPropDisplayMin";
pub const PARAM_PROP_DISPLAY_MAX: &CStr = c"OfxParamPropDisplayMax";
pub const PARAM_PROP_DIGITS: &CStr = c"OfxParamPropDigits";
pub const PARAM_PROP_INCREMENT: &CStr = c"OfxParamPropIncrement";
pub const PARAM_PROP_CHOICE_OPTION: &CStr = c"OfxParamPropChoiceOption";
pub const PARAM_PROP_PARENT: &CStr = c"OfxParamPropParent";
pub const PARAM_PROP_SCRIPT_NAME: &CStr = c"OfxParamPropScriptName";
pub const PARAM_PROP_DOUBLE_TYPE: &CStr = c"OfxParamPropDoubleType";
pub const DOUBLE_TYPE_XY_ABSOLUTE: &CStr = c"OfxParamDoubleTypeXYAbsolute";
pub const PARAM_PROP_DEFAULT_COORDINATES: &CStr = c"OfxParamPropDefaultCoordinateSystem";
pub const COORDINATES_NORMALISED: &CStr = c"OfxParamCoordinatesNormalised";
