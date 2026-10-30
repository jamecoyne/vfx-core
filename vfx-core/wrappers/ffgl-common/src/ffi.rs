//! The FFGL 2.1 plugin ABI, transcribed from Resolume's `FFGL.h`:
//!
//!   Copyright (c) 2018 www.freeframe.org. All rights reserved.
//!   Redistribution and use in source and binary forms, with or without modification,
//!   are permitted provided that the following conditions are met: redistributions of
//!   source code must retain the above copyright notice, this list of conditions and
//!   the following disclaimer; redistributions in binary form must reproduce the above
//!   copyright notice, this list of conditions and the following disclaimer in the
//!   documentation and/or other materials provided with the distribution; neither the
//!   name of FreeFrame nor the names of its contributors may be used to endorse or
//!   promote products derived from this software without specific prior written
//!   permission. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//!   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.
//!
//! Hosts call one exported function, `plugMain(code, input, instance)`. `input` and the
//! result are a 64-bit union of an integer and a pointer; float values travel as their
//! bit pattern in the integer.

use std::ffi::{c_char, c_void};

pub type GLuint = u32;

#[repr(C)]
#[derive(Clone, Copy)]
pub union FFMixed {
    pub uint: u32,
    pub ptr: *mut c_void,
}

impl FFMixed {
    pub fn uint(v: u32) -> Self {
        // Zero the whole union first, so the high half of a 64-bit read is defined.
        let mut m = FFMixed {
            ptr: std::ptr::null_mut(),
        };
        m.uint = v;
        m
    }

    pub fn float(v: f32) -> Self {
        Self::uint(v.to_bits())
    }

    pub fn ptr<T>(p: *const T) -> Self {
        FFMixed {
            ptr: p as *mut c_void,
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

// Function codes.
pub const FF_GET_INFO: u32 = 0;
pub const FF_INITIALISE_V2: u32 = 34;
pub const FF_DEINITIALISE: u32 = 2;
pub const FF_GET_NUM_PARAMETERS: u32 = 4;
pub const FF_GET_PARAMETER_NAME: u32 = 5;
pub const FF_GET_PARAMETER_DEFAULT: u32 = 6;
pub const FF_GET_PARAMETER_DISPLAY: u32 = 7;
pub const FF_SET_PARAMETER: u32 = 8;
pub const FF_GET_PARAMETER: u32 = 9;
pub const FF_GET_PLUGIN_CAPS: u32 = 10;
pub const FF_ENABLE_PLUGIN_CAP: u32 = 49;
pub const FF_GET_EXTENDED_INFO: u32 = 13;
pub const FF_GET_PARAMETER_TYPE: u32 = 15;
pub const FF_GET_INPUT_STATUS: u32 = 16;
pub const FF_PROCESS_OPENGL: u32 = 17;
pub const FF_INSTANTIATE_GL: u32 = 18;
pub const FF_DEINSTANTIATE_GL: u32 = 19;
pub const FF_SET_TIME: u32 = 20;
pub const FF_CONNECT: u32 = 21;
pub const FF_DISCONNECT: u32 = 22;
pub const FF_RESIZE: u32 = 23;
pub const FF_GET_NUM_PARAMETER_ELEMENTS: u32 = 31;
pub const FF_GET_PARAMETER_ELEMENT_NAME: u32 = 35;
pub const FF_GET_PARAMETER_ELEMENT_VALUE: u32 = 36;
pub const FF_SET_PARAMETER_ELEMENT_VALUE: u32 = 37;
pub const FF_GET_PARAMETER_USAGE: u32 = 32;
pub const FF_GET_PLUGIN_SHORT_NAME: u32 = 33;
pub const FF_SET_BEATINFO: u32 = 38;
pub const FF_SET_HOSTINFO: u32 = 39;
pub const FF_SET_SAMPLERATE: u32 = 40;
pub const FF_GET_RANGE: u32 = 41;
pub const FF_GET_PARAM_GROUP: u32 = 50;
pub const FF_GET_PARAM_DISPLAY_NAME: u32 = 51;
pub const FF_GET_PARAMETER_VISIBILITY: u32 = 45;
pub const FF_GET_PARAMETER_EVENTS: u32 = 46;
pub const FF_GET_NUM_ELEMENT_SEPARATORS: u32 = 47;

pub const FF_SUCCESS: u32 = 0;
pub const FF_FAIL: u32 = 0xFFFF_FFFF;
pub const FF_TRUE: u32 = 1;
pub const FF_FALSE: u32 = 0;

pub const FF_EFFECT: u32 = 0;

// Capabilities.
pub const FF_CAP_SET_TIME: u32 = 5;
pub const FF_CAP_MINIMUM_INPUT_FRAMES: u32 = 10;
pub const FF_CAP_MAXIMUM_INPUT_FRAMES: u32 = 11;

// Parameter types.
pub const FF_TYPE_BOOLEAN: u32 = 0;
pub const FF_TYPE_RED: u32 = 2;
pub const FF_TYPE_GREEN: u32 = 3;
pub const FF_TYPE_BLUE: u32 = 4;
pub const FF_TYPE_XPOS: u32 = 5;
pub const FF_TYPE_YPOS: u32 = 6;
pub const FF_TYPE_STANDARD: u32 = 10;
pub const FF_TYPE_OPTION: u32 = 11;

pub const FF_INPUT_INUSE: u32 = 1;
pub const FF_USAGE_STANDARD: u32 = 0;

#[repr(C)]
pub struct PluginInfoStruct {
    pub api_major_version: u32,
    pub api_minor_version: u32,
    pub plugin_unique_id: [u8; 4],
    pub plugin_name: [u8; 16],
    pub plugin_type: u32,
}

#[repr(C)]
pub struct PluginExtendedInfoStruct {
    pub plugin_major_version: u32,
    pub plugin_minor_version: u32,
    pub description: *const c_char,
    pub about: *const c_char,
    pub free_frame_extended_data_size: u32,
    pub free_frame_extended_data_block: *mut c_void,
}

// Both structs only point at static data.
unsafe impl Sync for PluginInfoStruct {}
unsafe impl Sync for PluginExtendedInfoStruct {}

#[repr(C)]
pub struct SetParameterStruct {
    pub parameter_number: u32,
    pub new_parameter_value: FFMixed,
}

#[repr(C)]
pub struct RangeStruct {
    pub min: f32,
    pub max: f32,
}

#[repr(C)]
pub struct GetRangeStruct {
    pub parameter_number: u32,
    pub range: RangeStruct,
}

#[repr(C)]
pub struct StringBufferStruct {
    pub address: *mut c_char,
    pub max_to_write: u32,
}

#[repr(C)]
pub struct GetStringStruct {
    pub parameter_number: u32,
    pub string_buffer: StringBufferStruct,
}

#[repr(C)]
pub struct GetParameterElementStruct {
    pub parameter_number: u32,
    pub element_number: u32,
}

#[repr(C)]
pub struct GetParamEventsStruct {
    pub num_events: u32,
    pub events: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FFGLViewportStruct {
    pub x: GLuint,
    pub y: GLuint,
    pub width: GLuint,
    pub height: GLuint,
}

#[repr(C)]
pub struct FFGLTextureStruct {
    /// Size of the image in the texture.
    pub width: u32,
    pub height: u32,
    /// Allocated size, which may be larger (for example a power of two).
    pub hardware_width: u32,
    pub hardware_height: u32,
    pub handle: GLuint,
}

#[repr(C)]
pub struct ProcessOpenGLStruct {
    pub num_input_textures: u32,
    pub input_textures: *mut *mut FFGLTextureStruct,
    /// The host's framebuffer, already bound, which the plugin draws into.
    pub host_fbo: GLuint,
}
