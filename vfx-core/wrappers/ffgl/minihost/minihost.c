/*
 * A minimal FFGL 2 host for testing the plugin without Resolume.
 *
 * Makes a headless OpenGL 4.1 core context, loads the plugin binary, queries its info
 * and parameters, then renders generated textures through FF_PROCESS_OPENGL into a
 * framebuffer and checks the pixels and the GL state the plugin leaves behind.
 *
 * FFGL_MINIHOST_SOFTWARE=1 uses Apple's software renderer, as on CI runners.
 *
 *   minihost <plugin binary> [case]   cases: all (default), info, passthrough,
 *                                     brightness, padding, blur, vignette, state, timing
 * Windows also checks context rejection and sharing/RAM pixel parity (context, sharing).
 * FFGL_MINIHOST_REQUIRE_INTEROP=1 rejects fallback during an automatic-path run.
 * Exit status 0 when every check passes.
 */
#define GL_SILENCE_DEPRECATION
#ifndef PLUGIN_NAME
#define PLUGIN_NAME "VFX Preview" /* run.sh passes the effect crate's short name */
#endif
#ifndef PLUGIN_LOG
#define PLUGIN_LOG "VfxPreview-FFGL.log" /* run.sh passes <file>-FFGL.log */
#endif

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <GL/gl.h>

/* Windows ships GL 1.1 headers. Declare the few newer calls and constants here. */
#define GL_RGBA32F 0x8814
#define GL_FRAMEBUFFER 0x8D40
#define GL_DRAW_FRAMEBUFFER 0x8CA9
#define GL_READ_FRAMEBUFFER 0x8CA8
#define GL_READ_FRAMEBUFFER_BINDING 0x8CAA
#define GL_COLOR_ATTACHMENT0 0x8CE0
#define GL_FRAMEBUFFER_COMPLETE 0x8CD5
#define GL_CURRENT_PROGRAM 0x8B8D
#define GL_VERTEX_ARRAY_BINDING 0x85B5
#define GL_ACTIVE_TEXTURE 0x84E0
#define GL_TEXTURE0 0x84C0
#define GL_FRAMEBUFFER_BINDING 0x8CA6

static PROC gl_proc(const char *name) {
    PROC p = wglGetProcAddress(name);
    uintptr_t address = (uintptr_t)p;
    if (!p || address == 1 || address == 2 || address == 3 || address == UINTPTR_MAX)
        p = (PROC)GetProcAddress(GetModuleHandleA("opengl32.dll"), name);
    return p;
}
typedef void (APIENTRY *glGenTexturesProc)(GLsizei n, GLuint *textures);
static glGenTexturesProc p_glGenTextures;
#define glGenTextures p_glGenTextures
typedef void (APIENTRY *glBindTextureProc)(GLenum target, GLuint texture);
static glBindTextureProc p_glBindTexture;
#define glBindTexture p_glBindTexture
typedef void (APIENTRY *glTexParameteriProc)(GLenum target, GLenum name, GLint value);
static glTexParameteriProc p_glTexParameteri;
#define glTexParameteri p_glTexParameteri
typedef void (APIENTRY *glTexImage2DProc)(GLenum target, GLint level, GLint internal, GLsizei w, GLsizei h, GLint border, GLenum format, GLenum kind, const void *data);
static glTexImage2DProc p_glTexImage2D;
#define glTexImage2D p_glTexImage2D
typedef void (APIENTRY *glGenFramebuffersProc)(GLsizei n, GLuint *fbos);
static glGenFramebuffersProc p_glGenFramebuffers;
#define glGenFramebuffers p_glGenFramebuffers
typedef void (APIENTRY *glBindFramebufferProc)(GLenum target, GLuint fbo);
static glBindFramebufferProc p_glBindFramebuffer;
#define glBindFramebuffer p_glBindFramebuffer
typedef void (APIENTRY *glFramebufferTexture2DProc)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level);
static glFramebufferTexture2DProc p_glFramebufferTexture2D;
#define glFramebufferTexture2D p_glFramebufferTexture2D
typedef GLenum (APIENTRY *glCheckFramebufferStatusProc)(GLenum target);
static glCheckFramebufferStatusProc p_glCheckFramebufferStatus;
#define glCheckFramebufferStatus p_glCheckFramebufferStatus
typedef void (APIENTRY *glViewportProc)(GLint x, GLint y, GLsizei w, GLsizei h);
static glViewportProc p_glViewport;
#define glViewport p_glViewport
typedef void (APIENTRY *glClearColorProc)(GLfloat r, GLfloat g, GLfloat b, GLfloat a);
static glClearColorProc p_glClearColor;
#define glClearColor p_glClearColor
typedef void (APIENTRY *glClearProc)(GLbitfield mask);
static glClearProc p_glClear;
#define glClear p_glClear
typedef void (APIENTRY *glReadPixelsProc)(GLint x, GLint y, GLsizei w, GLsizei h, GLenum format, GLenum kind, void *data);
static glReadPixelsProc p_glReadPixels;
#define glReadPixels p_glReadPixels
typedef void (APIENTRY *glGetIntegervProc)(GLenum name, GLint *data);
static glGetIntegervProc p_glGetIntegerv;
#define glGetIntegerv p_glGetIntegerv
typedef GLenum (APIENTRY *glGetErrorProc)(void);
static glGetErrorProc p_glGetError;
#define glGetError p_glGetError
typedef void (APIENTRY *glFinishProc)(void);
static glFinishProc p_glFinish;
#define glFinish p_glFinish
typedef const GLubyte * (APIENTRY *glGetStringProc)(GLenum name);
static glGetStringProc p_glGetString;
#define glGetString p_glGetString

static int load_gl(void) {
    p_glGenTextures = (glGenTexturesProc)gl_proc("glGenTextures");
    if (!p_glGenTextures) { fprintf(stderr, "missing OpenGL function glGenTextures\n"); return 0; }
    p_glBindTexture = (glBindTextureProc)gl_proc("glBindTexture");
    if (!p_glBindTexture) { fprintf(stderr, "missing OpenGL function glBindTexture\n"); return 0; }
    p_glTexParameteri = (glTexParameteriProc)gl_proc("glTexParameteri");
    if (!p_glTexParameteri) { fprintf(stderr, "missing OpenGL function glTexParameteri\n"); return 0; }
    p_glTexImage2D = (glTexImage2DProc)gl_proc("glTexImage2D");
    if (!p_glTexImage2D) { fprintf(stderr, "missing OpenGL function glTexImage2D\n"); return 0; }
    p_glGenFramebuffers = (glGenFramebuffersProc)gl_proc("glGenFramebuffers");
    if (!p_glGenFramebuffers) { fprintf(stderr, "missing OpenGL function glGenFramebuffers\n"); return 0; }
    p_glBindFramebuffer = (glBindFramebufferProc)gl_proc("glBindFramebuffer");
    if (!p_glBindFramebuffer) { fprintf(stderr, "missing OpenGL function glBindFramebuffer\n"); return 0; }
    p_glFramebufferTexture2D = (glFramebufferTexture2DProc)gl_proc("glFramebufferTexture2D");
    if (!p_glFramebufferTexture2D) { fprintf(stderr, "missing OpenGL function glFramebufferTexture2D\n"); return 0; }
    p_glCheckFramebufferStatus = (glCheckFramebufferStatusProc)gl_proc("glCheckFramebufferStatus");
    if (!p_glCheckFramebufferStatus) { fprintf(stderr, "missing OpenGL function glCheckFramebufferStatus\n"); return 0; }
    p_glViewport = (glViewportProc)gl_proc("glViewport");
    if (!p_glViewport) { fprintf(stderr, "missing OpenGL function glViewport\n"); return 0; }
    p_glClearColor = (glClearColorProc)gl_proc("glClearColor");
    if (!p_glClearColor) { fprintf(stderr, "missing OpenGL function glClearColor\n"); return 0; }
    p_glClear = (glClearProc)gl_proc("glClear");
    if (!p_glClear) { fprintf(stderr, "missing OpenGL function glClear\n"); return 0; }
    p_glReadPixels = (glReadPixelsProc)gl_proc("glReadPixels");
    if (!p_glReadPixels) { fprintf(stderr, "missing OpenGL function glReadPixels\n"); return 0; }
    p_glGetIntegerv = (glGetIntegervProc)gl_proc("glGetIntegerv");
    if (!p_glGetIntegerv) { fprintf(stderr, "missing OpenGL function glGetIntegerv\n"); return 0; }
    p_glGetError = (glGetErrorProc)gl_proc("glGetError");
    if (!p_glGetError) { fprintf(stderr, "missing OpenGL function glGetError\n"); return 0; }
    p_glFinish = (glFinishProc)gl_proc("glFinish");
    if (!p_glFinish) { fprintf(stderr, "missing OpenGL function glFinish\n"); return 0; }
    p_glGetString = (glGetStringProc)gl_proc("glGetString");
    if (!p_glGetString) { fprintf(stderr, "missing OpenGL function glGetString\n"); return 0; }
    return 1;
}

static HWND window;
static HDC dc;
static HGLRC ctx;

static int create_context(void) {
    WNDCLASSA wc = {0};
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = DefWindowProcA;
    wc.hInstance = GetModuleHandleA(NULL);
    wc.lpszClassName = "VfxFfglMinihost";
    if (!RegisterClassA(&wc)) return 0;
    /* No WS_VISIBLE: this window only supplies a device context. */
    window = CreateWindowA(wc.lpszClassName, "FFGL mini host", WS_OVERLAPPEDWINDOW,
                           0, 0, 16, 16, NULL, NULL, wc.hInstance, NULL);
    if (!window || !(dc = GetDC(window))) return 0;
    PIXELFORMATDESCRIPTOR pf = {0};
    pf.nSize = sizeof pf;
    pf.nVersion = 1;
    pf.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pf.iPixelType = PFD_TYPE_RGBA;
    pf.cColorBits = 32;
    int format = ChoosePixelFormat(dc, &pf);
    if (!format || !SetPixelFormat(dc, format, &pf)) return 0;
    HGLRC bootstrap = wglCreateContext(dc);
    if (!bootstrap) return 0;
    if (!wglMakeCurrent(dc, bootstrap)) { wglDeleteContext(bootstrap); return 0; }
    typedef HGLRC (WINAPI *CreateContext)(HDC, HGLRC, const int *);
    CreateContext create = (CreateContext)gl_proc("wglCreateContextAttribsARB");
    /* WGL_CONTEXT_{MAJOR,MINOR}_VERSION_ARB, PROFILE_MASK_ARB, CORE_PROFILE_BIT_ARB. */
    int attributes[] = {0x2091, 4, 0x2092, 1, 0x9126, 1, 0};
    if (create) ctx = create(dc, NULL, attributes);
    wglMakeCurrent(NULL, NULL);
    wglDeleteContext(bootstrap);
    if (!ctx || !wglMakeCurrent(dc, ctx)) return 0;
    return load_gl();
}

static void *open_plugin(const char *path) { return (void *)LoadLibraryA(path); }
static void *plugin_symbol(void *lib, const char *name) { return (void *)GetProcAddress((HMODULE)lib, name); }
static void open_failed(void) { fprintf(stderr, "LoadLibrary failed (error %lu)\n", GetLastError()); }
#else
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl3.h>
#include <dlfcn.h>
static void *open_plugin(const char *path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
static void *plugin_symbol(void *lib, const char *name) { return dlsym(lib, name); }
static void open_failed(void) { fprintf(stderr, "dlopen: %s\n", dlerror()); }
#endif

/* The FFGL 2.1 ABI (from Resolume's FFGL.h). */
typedef union { uint32_t u; void *p; } FFMixed;
typedef FFMixed (*PlugMain)(uint32_t, FFMixed, void *);
typedef struct { uint32_t major, minor; char id[4]; char name[16]; uint32_t type; } PluginInfo;
typedef struct { uint32_t number; FFMixed value; } SetParameter;
typedef struct { uint32_t number; float min, max; } GetRange;
typedef struct { uint32_t number; char *address; uint32_t max; } GetString;
typedef struct { uint32_t param, element; } GetElement;
typedef struct { GLuint x, y, w, h; } Viewport;
typedef struct { uint32_t w, h, hw, hh; GLuint handle; } Texture;
typedef struct { uint32_t count; Texture **textures; GLuint fbo; } Process;

enum {
    GET_INFO = 0, INITIALISE_V2 = 34, DEINITIALISE = 2, GET_NUM_PARAMETERS = 4, GET_PARAMETER_NAME = 5,
    GET_PARAMETER_DEFAULT = 6, GET_PARAMETER_DISPLAY = 7, SET_PARAMETER = 8, GET_PARAMETER = 9,
    GET_PLUGIN_CAPS = 10, GET_PARAMETER_TYPE = 15, PROCESS_OPENGL = 17, INSTANTIATE_GL = 18,
    DEINSTANTIATE_GL = 19, SET_TIME = 20, GET_NUM_PARAMETER_ELEMENTS = 31, GET_PARAMETER_ELEMENT_NAME = 35,
    GET_RANGE = 41, GET_PARAM_GROUP = 50,
};
#define FF_FAIL 0xFFFFFFFFu
#define TYPE_OPTION 11u

static PlugMain plug;
static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static FFMixed u(uint32_t v) { FFMixed m = {0}; m.u = v; return m; }
static FFMixed ptr(void *p) { FFMixed m; m.p = p; return m; }
static FFMixed call(uint32_t code, FFMixed in, void *inst) { return plug(code, in, inst); }

/* Parameter index by name, from the plugin's own table. */
static int param(const char *name) {
    uint32_t n = call(GET_NUM_PARAMETERS, u(0), NULL).u;
    for (uint32_t i = 0; i < n; i++) {
        const char *p = call(GET_PARAMETER_NAME, u(i), NULL).p;
        if (p && !strcmp(p, name)) return (int)i;
    }
    failures++;
    printf("  FAIL: no parameter %s\n", name);
    return 0;
}

static void set(void *inst, const char *name, float v) {
    SetParameter s;
    s.number = (uint32_t)param(name);
    memset(&s.value, 0, sizeof s.value);
    memcpy(&s.value.u, &v, 4);
    CHECK(call(SET_PARAMETER, ptr(&s), inst).u == 0, "set %s failed", name);
}

/* ---- GL helpers -------------------------------------------------------------------- */

static GLuint make_texture(int w, int h, const float *pixels) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, pixels);
    glBindTexture(GL_TEXTURE_2D, 0);
    return t;
}

typedef struct { GLuint fbo, color; int w, h; } Target;

static Target make_target(int w, int h) {
    Target t = {0, make_texture(w, h, NULL), w, h};
    glGenFramebuffers(1, &t.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t.color, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) { fprintf(stderr, "FBO incomplete\n"); exit(2); }
    return t;
}

/* Pixel (x, y), y up, of an RGBA float image `w` wide. */
static float *px(float *img, int w, int x, int y) { return &img[((size_t)y * w + x) * 4]; }

/* Straight-alpha pattern: red ramps with x, green with y, blue checker, alpha 1 with a
   half-transparent band on the right. Stored premultiplied, as FFGL textures are. */
static void pattern(float *straight, float *premult, int w, int h, int stride) {
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            float *s = px(straight, w, x, y), *p = px(premult, stride, x, y);
            s[0] = (float)x / (w - 1);
            s[1] = (float)y / (h - 1);
            s[2] = ((x / 8 + y / 8) % 2) ? 0.8f : 0.2f;
            s[3] = x < w * 3 / 4 ? 1.0f : 0.5f;
            for (int c = 0; c < 3; c++) p[c] = s[c] * s[3];
            p[3] = s[3];
        }
}

typedef struct {
    int w, h, hw, hh;
    float *straight;
    GLuint input;
    Target target;
    float *out; /* straight, w x h, y up */
} Frame;

static Frame frame(int w, int h, int hw, int hh) {
    Frame f = {w, h, hw, hh, calloc((size_t)w * h * 4, 4), 0, make_target(w, h), calloc((size_t)w * h * 4, 4)};
    float *texels = calloc((size_t)hw * hh * 4, 4);
    for (size_t i = 0; i < (size_t)hw * hh * 4; i++) texels[i] = 9.0f; /* padding the plugin must ignore */
    pattern(f.straight, texels, w, h, hw);
    f.input = make_texture(hw, hh, texels);
    free(texels);
    return f;
}

static void check_context(void) {
#ifdef _WIN32
    if (wglGetCurrentContext() != ctx || wglGetCurrentDC() != dc) {
        fprintf(stderr, "plugin changed the host context or device context\n");
        exit(2);
    }
#endif
}

static int render(void *inst, Frame *f) {
    Texture t = {(uint32_t)f->w, (uint32_t)f->h, (uint32_t)f->hw, (uint32_t)f->hh, f->input};
    Texture *list[1] = {&t};
    Process p = {1, list, f->target.fbo};
    glBindFramebuffer(GL_FRAMEBUFFER, f->target.fbo);
    glViewport(0, 0, f->w, f->h);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    uint32_t r = call(PROCESS_OPENGL, ptr(&p), inst).u;
    check_context();
    glReadPixels(0, 0, f->w, f->h, GL_RGBA, GL_FLOAT, f->out);
    for (int i = 0; i < f->w * f->h; i++) {
        float *o = &f->out[i * 4];
        if (o[3] > 0) for (int c = 0; c < 3; c++) o[c] /= o[3];
    }
    return r == 0;
}

static void *instantiate(int w, int h) {
    Viewport v = {0, 0, (GLuint)w, (GLuint)h};
    FFMixed r = call(INSTANTIATE_GL, ptr(&v), NULL);
    if (!r.p || (uintptr_t)r.p == FF_FAIL) {
        fprintf(stderr, "instantiate failed (see the plugin's FFGL log)\n");
        exit(2);
    }
    double t = 0;
    call(SET_TIME, ptr(&t), r.p);
    return r.p;
}

static float max_diff(Frame *f) {
    float worst = 0;
    for (int i = 0; i < f->w * f->h * 4; i++) {
        float d = fabsf(f->out[i] - f->straight[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

/* ---- Cases ------------------------------------------------------------------------- */

#ifdef _WIN32
static void case_context(void) {
    CHECK(wglMakeCurrent(NULL, NULL), "detach context");
    PluginInfo *info = call(GET_INFO, u(0), NULL).p;
    CHECK(info && info->major == 2, "metadata needs no GL context");
    Viewport v = {0, 0, 64, 48};
    CHECK(call(INSTANTIATE_GL, ptr(&v), NULL).u == FF_FAIL, "instantiate without a context should fail");
    if (!wglMakeCurrent(dc, ctx)) { fprintf(stderr, "restore context failed\n"); exit(2); }
    void *inst = instantiate(64, 48);
    CHECK(call(DEINSTANTIATE_GL, u(0), inst).u == 0, "delete instance after restoring context");
}
#endif

static void case_info(void) {
    PluginInfo *info = call(GET_INFO, u(0), NULL).p;
    CHECK(info && info->major == 2 && info->minor >= 1, "API version %u.%u", info ? info->major : 0, info ? info->minor : 0);
    CHECK(info && !strncmp(info->name, PLUGIN_NAME, sizeof info->name), "plugin name");
    CHECK(info && info->type == 0, "plugin type is effect");
    CHECK(call(GET_PLUGIN_CAPS, u(10), NULL).u == 1 && call(GET_PLUGIN_CAPS, u(11), NULL).u == 1, "one input");
    uint32_t n = call(GET_NUM_PARAMETERS, u(0), NULL).u;
    CHECK(n >= 10 && n < 64, "%u parameters", n);
    for (uint32_t i = 0; i < n; i++) {
        const char *name = call(GET_PARAMETER_NAME, u(i), NULL).p;
        CHECK(name && *name, "parameter %u has no name", i);
        GetRange r = {i, -99, -99};
        CHECK(call(GET_RANGE, ptr(&r), NULL).u == 0 && r.min < r.max, "%s range %g..%g", name, r.min, r.max);
        FFMixed d = call(GET_PARAMETER_DEFAULT, u(i), NULL);
        float dv;
        memcpy(&dv, &d.u, 4);
        CHECK(dv >= r.min && dv <= r.max, "%s default %g outside %g..%g", name, dv, r.min, r.max);
        if (call(GET_PARAMETER_TYPE, u(i), NULL).u == TYPE_OPTION) {
            uint32_t k = call(GET_NUM_PARAMETER_ELEMENTS, u(i), NULL).u;
            CHECK(k >= 2, "%s has %u options", name, k);
            for (uint32_t e = 0; e < k; e++) {
                GetElement ge = {i, e};
                const char *label = call(GET_PARAMETER_ELEMENT_NAME, ptr(&ge), NULL).p;
                CHECK(label && *label, "%s option %u has no label", name, e);
            }
        }
    }
    char group[64] = {0};
    int segcam = info && !memcmp(info->id, "SGCM", 4);
    GetString g = {(uint32_t)param(segcam ? "Sensitivity" : "Brightness"), group, sizeof group - 1};
    CHECK(call(GET_PARAM_GROUP, ptr(&g), NULL).u == 0 && !strcmp(group, segcam ? "Motion" : "Color"), "parameter group is '%s'", group);
}

static void case_passthrough(void) {
    void *inst = instantiate(64, 48);
    Frame f = frame(64, 48, 64, 48);
    CHECK(render(inst, &f), "process failed");
    CHECK(max_diff(&f) < 1e-4, "defaults changed the image (max diff %g)", max_diff(&f));
    call(DEINSTANTIATE_GL, u(0), inst);
}

static void case_brightness(void) {
    void *inst = instantiate(64, 48);
    set(inst, "Brightness", 0.25f);
    FFMixed got = call(GET_PARAMETER, u((uint32_t)param("Brightness")), inst);
    float v; memcpy(&v, &got.u, 4);
    CHECK(fabsf(v - 0.25f) < 1e-6, "Brightness reads back %g", v);
    const char *shown = call(GET_PARAMETER_DISPLAY, u((uint32_t)param("Brightness")), inst).p;
    CHECK(shown && !strcmp(shown, "0.25"), "Brightness displays '%s'", shown ? shown : "(null)");
    Frame f = frame(64, 48, 64, 48);
    render(inst, &f);
    float want = px(f.straight, 64, 10, 5)[0] + 0.25f, gotr = px(f.out, 64, 10, 5)[0];
    CHECK(fabsf(gotr - want) < 1e-4, "red at (10,5) %g, expected %g", gotr, want);
    CHECK(px(f.out, 64, 10, 40)[1] > px(f.out, 64, 10, 5)[1], "image upside down");
    /* The half-transparent band: straight colour is brightened, alpha kept. */
    CHECK(fabsf(px(f.out, 64, 60, 20)[3] - 0.5f) < 1e-4, "alpha changed (%g)", px(f.out, 64, 60, 20)[3]);
    CHECK(fabsf(px(f.out, 64, 60, 20)[0] - (px(f.straight, 64, 60, 20)[0] + 0.25f)) < 1e-3, "half-transparent pixel wrong");
    /* Float transfers must keep HDR/negative colour and discard colour at zero alpha. */
    Frame hdr = frame(2, 2, 2, 2);
    float texels[] = {1.0f, -0.25f, 0.125f, 0.5f, 9, 8, 7, 0,
                     0, 0, 0, 1, 0.5f, 0.25f, 0, 1};
    hdr.input = make_texture(2, 2, texels);
    CHECK(render(inst, &hdr), "HDR process failed");
    CHECK(fabsf(hdr.out[0] - 2.25f) < 1e-4 && fabsf(hdr.out[1] + 0.25f) < 1e-4,
          "HDR/negative colour changed: %g, %g", hdr.out[0], hdr.out[1]);
    CHECK(hdr.out[4] == 0 && hdr.out[5] == 0 && hdr.out[6] == 0 && hdr.out[7] == 0,
          "transparent pixel retained colour");
    call(DEINSTANTIATE_GL, u(0), inst);
}

static void case_padding(void) {
    /* A 50x40 image in a 64x64 allocation: the plugin must use only the image part. */
    void *inst = instantiate(50, 40);
    Frame f = frame(50, 40, 64, 64);
    CHECK(render(inst, &f), "process failed");
    CHECK(max_diff(&f) < 1e-4, "padded texture changed the image (max diff %g)", max_diff(&f));
    /* Reuse one instance across size changes, including rows needing wgpu padding. */
    Frame odd = frame(37, 29, 64, 32);
    CHECK(render(inst, &odd), "resized process failed");
    CHECK(max_diff(&odd) < 1e-4, "resized padded texture changed the image (max diff %g)", max_diff(&odd));
    CHECK(render(inst, &f), "process after restoring size failed");
    CHECK(max_diff(&f) < 1e-4, "restored texture changed the image (max diff %g)", max_diff(&f));
    call(DEINSTANTIATE_GL, u(0), inst);
}

static void case_blur(void) {
    void *inst = instantiate(64, 48);
    set(inst, "Blur Radius", 4);
    Frame f = frame(64, 48, 64, 48);
    render(inst, &f);
    float edge = px(f.out, 64, 8, 4)[2];
    CHECK(edge > 0.25f && edge < 0.75f, "blue at the checker edge %g, expected a blend", edge);
    set(inst, "Blur Direction", 2); /* Vertical: no horizontal spread */
    render(inst, &f);
    float h = px(f.out, 64, 8, 4)[2];
    CHECK(fabsf(h - px(f.straight, 64, 8, 4)[2]) < 0.05f || fabsf(px(f.out, 64, 8, 4)[2] - px(f.out, 64, 9, 4)[2]) < 1e-3,
          "vertical blur spread horizontally");
    call(DEINSTANTIATE_GL, u(0), inst);
}

static void case_vignette(void) {
    void *inst = instantiate(64, 48);
    set(inst, "Vignette", 1);
    set(inst, "Vignette X", 0.25f);
    set(inst, "Vignette Y", 0.25f); /* bottom-left quadrant, y up */
    Frame f = frame(64, 48, 64, 48);
    render(inst, &f);
    /* Brightness relative to the input: near the centre ~1, the far corner darker. */
    float near_brightness = px(f.out, 64, 16, 12)[1] / fmaxf(px(f.straight, 64, 16, 12)[1], 1e-3f);
    float far_brightness = px(f.out, 64, 63, 47)[1] / fmaxf(px(f.straight, 64, 63, 47)[1], 1e-3f);
    CHECK(near_brightness > 0.95f, "vignette centre darkened (%g)", near_brightness);
    CHECK(far_brightness < 0.6f, "far corner not darkened (%g)", far_brightness);
    call(DEINSTANTIATE_GL, u(0), inst);
}

static void free_frame(Frame *f) {
#ifdef _WIN32
    typedef void (APIENTRY *DeleteFramebuffers)(GLsizei, const GLuint *);
    typedef void (APIENTRY *DeleteTextures)(GLsizei, const GLuint *);
    ((DeleteFramebuffers)gl_proc("glDeleteFramebuffers"))(1, &f->target.fbo);
    DeleteTextures delete_textures = (DeleteTextures)gl_proc("glDeleteTextures");
    delete_textures(1, &f->target.color);
    delete_textures(1, &f->input);
#else
    glDeleteFramebuffers(1, &f->target.fbo);
    glDeleteTextures(1, &f->target.color);
    glDeleteTextures(1, &f->input);
#endif
    free(f->straight);
    free(f->out);
}

#ifdef SEGCAM_CHECKS
static void case_segcam(void) {
    case_info();
    void *inst = instantiate(63, 37);
    Frame f = frame(63, 37, 128, 64);
    set(inst, "Output", 0);
    CHECK(render(inst, &f), "segcam first frame failed");
    CHECK(max_diff(&f) <= 1.0f / 255, "segcam first overlay should preserve the input");
    set(inst, "Output", 1);
    CHECK(render(inst, &f), "segcam matte failed");
    for (int i = 0; i < f.w * f.h; i++)
        if (f.out[i * 4 + 3] != 0) { CHECK(0, "still image produced motion"); break; }
    call(DEINSTANTIATE_GL, u(0), inst);
    free_frame(&f);
}
#endif

static void case_state(void) {
    /* FFGL hosts expect the plugin to leave GL state as it found it. */
    void *inst = instantiate(64, 48);
    PluginInfo *info = call(GET_INFO, u(0), NULL).p;
    set(inst, !memcmp(info->id, "SGCM", 4) ? "Output" : "Blur Radius", 2);
    Frame f = frame(64, 48, 64, 48);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, f.target.fbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    Texture tex = {64, 48, 64, 48, f.input};
    Texture *list[1] = {&tex};
    Process p = {1, list, f.target.fbo};
    glViewport(3, 5, 50, 35);
    CHECK(call(PROCESS_OPENGL, ptr(&p), inst).u == 0, "process with offset viewport failed");
    check_context();
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    CHECK(viewport[0] == 3 && viewport[1] == 5 && viewport[2] == 50 && viewport[3] == 35,
          "host viewport changed: %d,%d %dx%d", viewport[0], viewport[1], viewport[2], viewport[3]);
    GLint v;
    glGetIntegerv(GL_CURRENT_PROGRAM, &v); CHECK(v == 0, "program %d left bound", v);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &v); CHECK(v == 0, "texture %d left bound", v);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &v); CHECK(v == 0, "vertex array %d left bound", v);
    glGetIntegerv(GL_ACTIVE_TEXTURE, &v); CHECK(v == GL_TEXTURE0, "active texture unit changed");
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &v); CHECK(v == (GLint)f.target.fbo, "host framebuffer unbound");
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &v); CHECK(v == 0, "host read framebuffer changed");
    GLint pack; glGetIntegerv(GL_PACK_ALIGNMENT, &pack); CHECK(pack == 4, "pack alignment %d", pack);
    CHECK(glGetError() == GL_NO_ERROR, "GL error after process");
    call(DEINSTANTIATE_GL, u(0), inst);
    free_frame(&f);
}

static double now(void) {
#ifdef _WIN32
    LARGE_INTEGER t, frequency;
    QueryPerformanceCounter(&t);
    QueryPerformanceFrequency(&frequency);
    return (double)t.QuadPart / frequency.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
#endif
}

#ifndef _WIN32
/* The plugin's own device opens on a thread of its own and frames pass through until it's open.
 * The cases compare the effect's output, so wait for a frame the effect changed first. Windows
 * waits in its own cases, the first of which checks that pass through. */
static void wait_for_effect(void) {
    void *inst = instantiate(64, 48);
    set(inst, "Brightness", 0.25f);
    Frame f = frame(64, 48, 64, 48);
    double started = now();
    struct timespec pause = {0, 5000000};
    do {
        render(inst, &f);
        if (max_diff(&f) > 1e-3f) break;
        nanosleep(&pause, NULL);
    } while (now() - started < 10);
    if (max_diff(&f) <= 1e-3f) printf("  the effect still passed frames through after 10 s\n");
    free_frame(&f);
    call(DEINSTANTIATE_GL, u(0), inst);
}
#endif

/* segcam Motion's own checks, from plugins/segcam (its ffgl-tests/run.sh passes the folder). */
#ifdef SEGCAM_CHECKS
#include "segcam.h"
#endif

static void case_timing(void) {
    int sizes[][2] = {{1280, 720}, {1920, 1080}};
    for (int s = 0; s < 2; s++) {
        int w = sizes[s][0], h = sizes[s][1];
        void *inst = instantiate(w, h);
        set(inst, "Blur Radius", 8);
        set(inst, "Vignette", 1);
        Frame f = frame(w, h, w, h);
        render(inst, &f); /* warm up: chain built on first use */
        int n = 30;
        double t0 = now();
        for (int i = 0; i < n; i++) {
            double t = i / 60.0;
            call(SET_TIME, ptr(&t), inst);
            Texture tex = {(uint32_t)w, (uint32_t)h, (uint32_t)w, (uint32_t)h, f.input};
            Texture *list[1] = {&tex};
            Process p = {1, list, f.target.fbo};
            CHECK(call(PROCESS_OPENGL, ptr(&p), inst).u == 0, "timed process failed");
            check_context();
        }
        glFinish();
        double ms = (now() - t0) * 1000 / n;
        printf("  %dx%d: %.1f ms per frame (blur 8 + vignette)\n", w, h, ms);
        call(DEINSTANTIATE_GL, u(0), inst);
    }
}

#ifdef _WIN32
#define LOG_PATH_MAX MAX_PATH
static unsigned long process_id(void) { return GetCurrentProcessId(); }
static int log_path(char *name, size_t size) {
    DWORD n = GetTempPathA((DWORD)size, name);
    if (!n || n + strlen(PLUGIN_LOG) >= size) return 0;
    strcat(name, PLUGIN_LOG);
    return 1;
}
#else
#include <unistd.h>
#define LOG_PATH_MAX 1024
static unsigned long process_id(void) { return (unsigned long)getpid(); }
static int log_path(char *name, size_t size) {
    const char *home = getenv("HOME");
    return home && snprintf(name, size, "%s/Library/Logs/%s", home, PLUGIN_LOG) < (int)size;
}
#endif

/* The log's length before the plugin loads: this process's entries all come after it. */
static long log_start = 0;

static void mark_log_start(void) {
    char name[LOG_PATH_MAX];
    FILE *log = log_path(name, sizeof name) ? fopen(name, "rb") : NULL;
    if (!log) return;
    if (!fseek(log, 0, SEEK_END)) log_start = ftell(log);
    fclose(log);
}

/* Check only this process's log entries, never a successful path from an earlier run. With
 * `number`, it takes the number right after `message` in the latest entry that has one. */
static int logged_entry(const char *message, double *number) {
    char name[LOG_PATH_MAX], prefix[64], line[2048];
    if (!log_path(name, sizeof name)) return 0;
    FILE *log = fopen(name, "rb");
    if (!log) return 0;
    if (log_start > 0) fseek(log, log_start, SEEK_SET);
    snprintf(prefix, sizeof prefix, "[pid %lu] ", process_id());
    int found = 0;
    while (fgets(line, sizeof line, log)) {
        const char *at = strstr(line, message);
        if (strncmp(line, prefix, strlen(prefix)) || !at) continue;
        if (!number) { found = 1; continue; }
        char *end;
        double parsed = strtod(at + strlen(message), &end);
        if (end != at + strlen(message)) { *number = parsed; found = 1; }
    }
    fclose(log);
    return found;
}

static int logged_message(const char *message) { return logged_entry(message, NULL); }

#ifdef _WIN32

/* The rest of the line after `message` in this process's latest entry that has it, in `rest`;
 * empty if none does. */
static void logged_rest(const char *message, char *rest, size_t size) {
    char name[MAX_PATH], prefix[64], line[2048];
    rest[0] = 0;
    if (!log_path(name, sizeof name)) return;
    FILE *log = fopen(name, "rb");
    if (!log) return;
    if (log_start > 0) fseek(log, log_start, SEEK_SET);
    snprintf(prefix, sizeof prefix, "[pid %lu] ", process_id());
    while (fgets(line, sizeof line, log)) {
        const char *at = strstr(line, message);
        if (strncmp(line, prefix, strlen(prefix)) || !at) continue;
        snprintf(rest, size, "%s", at + strlen(message));
        rest[strcspn(rest, "\r\n")] = 0;
    }
    fclose(log);
}

/* The number of the plugin's own device that opened last in this process, from 1; 0 before any. */
static double own_device_number(void) {
    double number = 0;
    logged_entry("own device ", &number);
    return number;
}

/* A GPU effect's RAM path renders on a device of the plugin's own, opened on a thread of its own,
 * and passes frames through until it is open. Renders `ram` until a device numbered above `after`
 * is open, so the frame it renders next is a RAM reference. */
static void wait_for_ram_device(void *ram, Frame *f, double after) {
    double started = now();
    while (own_device_number() <= after && now() - started < 10) {
        CHECK(render(ram, f), "render while the RAM path's device opens");
        Sleep(5);
    }
    CHECK(own_device_number() > after, "the RAM path's device never opened");
}

static void case_sharing(void) {
    const char *original = getenv("VFX_FFGL_PATH");
    char *saved = original ? strdup(original) : NULL;
    void *automatic = instantiate(63, 37);
    void *peer = instantiate(63, 37);
    _putenv_s("VFX_FFGL_PATH", "ram");
    void *ram = instantiate(63, 37);
    _putenv_s("VFX_FFGL_PATH", saved ? saved : "");
    free(saved);
    /* The first RAM frames of a process come before the device opens: they pass through unchanged,
     * with no wait on the open. The effect changes the image, so its output can't be mistaken for
     * the input. A device that opened before this render leaves nothing to check. */
    if (own_device_number() == 0) {
        Frame f = frame(63, 37, 80, 48);
        set(ram, "Brightness", 0.2f);
        double started = now();
        CHECK(render(ram, &f), "first RAM render");
        double took = now() - started;
        if (!logged_message("render path: GPU through RAM")) {
            CHECK(logged_message("render path: passing frames through: no device for the RAM path"),
                  "the first RAM frame didn't say it passed through");
            CHECK(max_diff(&f) < 1e-4f, "the first RAM frame wasn't the input: %g", max_diff(&f));
            CHECK(took < 0.1, "the first RAM frame took %.1f ms", took * 1000);
        } else {
            printf("  the RAM path's device opened before the first RAM frame\n");
        }
        wait_for_ram_device(ram, &f, 0);
        free_frame(&f);
    }
    /* Reuse instances across sizes, settings, padding, transparent and HDR pixels. */
    int sizes[][2] = {{63, 37}, {17, 13}, {63, 37}};
    for (int s = 0; s < 3; s++) {
        Frame f = frame(sizes[s][0], sizes[s][1], 80, 48);
        float *pixels = calloc((size_t)f.hw * f.hh * 4, sizeof(float));
        pattern(f.straight, pixels, f.w, f.h, f.hw);
        for (int y = 0; y < f.h; y++) for (int x = 0; x < f.w; x++) {
            float *p = px(pixels, f.hw, x, y);
            if (x == 0) { p[0] = 4.0f; p[1] = 2.0f; }
            if (y == 0) { p[0] = p[1] = p[2] = p[3] = 0; }
        }
        glBindTexture(GL_TEXTURE_2D, f.input);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, f.hw, f.hh, 0, GL_RGBA, GL_FLOAT, pixels);
        glBindTexture(GL_TEXTURE_2D, 0);
        free(pixels);
        void *instances[] = {automatic, peer, ram};
        for (int i = 0; i < 3; i++) {
            set(instances[i], "Brightness", s ? 0.2f : 0);
            set(instances[i], "Blur Radius", s == 2 ? 3 : 0);
            set(instances[i], "Vignette", s == 1 ? 1 : 0);
            set(instances[i], "Linear Light", s == 2 ? 1 : 0);
        }
        float *reference = malloc((size_t)f.w * f.h * 4 * sizeof(float));
        for (int repeat = 0; repeat < 3; repeat++) {
            CHECK(render(automatic, &f), "automatic sharing render");
            memcpy(reference, f.out, (size_t)f.w * f.h * 4 * sizeof(float));
            CHECK(render(peer, &f), "second shared instance render");
            float peer_worst = 0;
            for (int i = 0; i < f.w * f.h * 4; i++) {
                float diff = fabsf(reference[i] - f.out[i]);
                if (!isfinite(f.out[i])) { peer_worst = INFINITY; break; }
                if (diff > peer_worst) peer_worst = diff;
            }
            CHECK(peer_worst < 1e-4f, "shared instance mismatch: %g", peer_worst);
            CHECK(render(ram, &f), "forced RAM reference render");
            float worst = 0;
            for (int i = 0; i < f.w * f.h * 4; i++) {
                float diff = fabsf(reference[i] - f.out[i]);
                if (!isfinite(reference[i]) || !isfinite(f.out[i])) { worst = INFINITY; break; }
                if (diff > worst) worst = diff;
            }
            CHECK(worst < 1e-4f, "sharing/RAM mismatch at size %d repeat %d: %g", s, repeat, worst);
            CHECK(glGetError() == GL_NO_ERROR, "sharing left a GL error");
        }
        free(reference);
        free_frame(&f);
    }
    call(DEINSTANTIATE_GL, u(0), automatic);
    call(DEINSTANTIATE_GL, u(0), peer);
    call(DEINSTANTIATE_GL, u(0), ram);
    if (getenv("FFGL_MINIHOST_REQUIRE_INTEROP")) {
        CHECK(logged_message("render path: GPU-native (Vulkan"), "Vulkan shared path did not run in this process");
    }
}

/* GL errors left on the context by the host or another plugin aren't the plugin's: it stays on the
 * shared path, with the right pixels. An invalid target to glBindTexture is GL_INVALID_ENUM. */
static void case_pending_gl_error(void) {
    _putenv_s("VFX_FFGL_PATH", "ram");
    void *ram = instantiate(63, 37);
    _putenv_s("VFX_FFGL_PATH", "");
    Frame f = frame(63, 37, 80, 48);
    float *reference = malloc((size_t)f.w * f.h * 4 * sizeof(float));
    wait_for_ram_device(ram, &f, 0);
    CHECK(render(ram, &f), "forced RAM reference render");
    memcpy(reference, f.out, (size_t)f.w * f.h * 4 * sizeof(float));
    glBindTexture(0xFFFF, 0);
    void *shared = instantiate(63, 37);
    for (int i = 0; i < 3; i++) {
        glBindTexture(0xFFFF, 0);
        CHECK(render(shared, &f), "render %d with an error pending", i);
        float worst = 0;
        for (int k = 0; k < f.w * f.h * 4; k++) {
            float diff = fabsf(reference[k] - f.out[k]);
            if (diff > worst) worst = diff;
        }
        CHECK(worst < 1e-4f, "render %d differs from RAM: %g", i, worst);
    }
    free(reference);
    free_frame(&f);
    call(DEINSTANTIATE_GL, u(0), shared);
    call(DEINSTANTIATE_GL, u(0), ram);
    if (getenv("FFGL_MINIHOST_REQUIRE_INTEROP")) {
        CHECK(!logged_message("zero-copy path unavailable"), "a pending GL error stopped setup");
        CHECK(!logged_message("shared GPU render failed"), "a pending GL error failed a render");
        CHECK(logged_message("render path: GPU-native (Vulkan"), "the shared path did not run");
    }
}

/* A device lost, by destroy(), which goes another way through wgpu than a driver's loss: frames stay
 * right through RAM, and the instance goes back to shared frames on a new device. */
static void case_device_reopen(void) {
    _putenv_s("VFX_FFGL_PATH", "ram");
    void *ram = instantiate(63, 37);
    _putenv_s("VFX_FFGL_PATH", "");
    Frame f = frame(63, 37, 80, 48);
    float *reference = malloc((size_t)f.w * f.h * 4 * sizeof(float));
    wait_for_ram_device(ram, &f, 0);
    CHECK(render(ram, &f), "forced RAM reference render");
    memcpy(reference, f.out, (size_t)f.w * f.h * 4 * sizeof(float));
    /* A new instance's setup and first render on a healthy device: GL objects and shaders of its
     * own, which a slow machine can take long over and which no device open is part of. */
    double setup_started = now();
    void *healthy = instantiate(63, 37);
    CHECK(render(healthy, &f), "render of an instance on the healthy device");
    double setup = now() - setup_started;
    call(DEINSTANTIATE_GL, u(0), healthy);
    _putenv_s("VFX_FFGL_LOSE_DEVICE_AFTER", "20");
    void *lost = instantiate(63, 37);
    /* Instances made while the device is out, which mustn't wait for it: one as soon as the loss
     * is logged, and one once the first try is due, which may land while that try is opening. */
    void *during = NULL, *during_try = NULL;
    /* No call that a device open could reach may wait for one: the gap's instances setting up, and
     * every render of the lost instance from the one that finds the loss on. The bound is far above
     * a frame this small through RAM and below the time a device takes to open. A render can still
     * be slowed by the open without waiting for it: while the open's thread creates the Vulkan
     * device, the driver holds back every GL and Vulkan wait in the process, the host's included,
     * for about 40 to 60 ms on an RTX 4070, a fraction of the whole open. A gap instance's setup and
     * first render get three times a healthy one's, at least the bound, and so does the render that
     * finishes the new device's setup and builds the effect on it. */
    const double bound = 0.1;
    const double setup_bound = 3 * setup > bound ? 3 * setup : bound;
    double lost_at = 0, worst = 0, worst_setup = 0;
    /* Read when the first render sets up the shared images; the new device stays. The first try
     * is a second after the loss. */
    int rendered = 0;
    for (int back = 0; rendered < 2000 && !back; rendered++) {
        double started = now();
        CHECK(render(lost, &f), "render %d", rendered);
        double took = now() - started;
        back = logged_message("shared frames back on a new device");
        if (back && took > worst_setup) worst_setup = took;
        _putenv_s("VFX_FFGL_LOSE_DEVICE_AFTER", "");
        if (!lost_at && logged_message("warning: GPU device lost")) lost_at = now();
        if (lost_at && !back && took > worst) worst = took;
        if (lost_at && !during) {
            started = now();
            during = instantiate(63, 37);
            CHECK(render(during, &f), "render of an instance made in the gap");
            if (now() - started > worst_setup) worst_setup = now() - started;
            started = now();
            CHECK(render(during, &f), "second render of an instance made in the gap");
            if (now() - started > worst) worst = now() - started;
        }
        if (lost_at && !during_try && now() - lost_at >= 1.0) {
            started = now();
            during_try = instantiate(63, 37);
            CHECK(render(during_try, &f), "render of an instance made during the first try");
            if (now() - started > worst_setup) worst_setup = now() - started;
            started = now();
            CHECK(render(during_try, &f), "second render of an instance made during the first try");
            if (now() - started > worst) worst = now() - started;
        }
        float diff_worst = 0;
        for (int k = 0; k < f.w * f.h * 4; k++) {
            float diff = fabsf(reference[k] - f.out[k]);
            if (diff > diff_worst) diff_worst = diff;
        }
        CHECK(diff_worst < 1e-4f, "render %d differs from RAM: %g", rendered, diff_worst);
        CHECK(glGetError() == GL_NO_ERROR, "render %d left a GL error", rendered);
        Sleep(2);
    }
    for (int i = 0; i < 10; i++) {
        double started = now();
        CHECK(render(lost, &f), "render %d after the new device", i);
        if (now() - started > worst) worst = now() - started;
        float diff_worst = 0;
        for (int k = 0; k < f.w * f.h * 4; k++) {
            float diff = fabsf(reference[k] - f.out[k]);
            if (diff > diff_worst) diff_worst = diff;
        }
        CHECK(diff_worst < 1e-4f, "render %d on the new device differs from RAM: %g", i, diff_worst);
    }
    if (lost_at) {
        printf("  worst call a device open could reach: %.1f ms, a setup on the way back %.1f ms "
               "(healthy %.1f ms)\n",
               worst * 1000, worst_setup * 1000, setup * 1000);
        CHECK(worst < bound, "a call waited %.1f ms, over the %.0f ms bound", worst * 1000, bound * 1000);
        CHECK(worst_setup < setup_bound, "a setup on the way back took %.1f ms, over %.0f ms",
              worst_setup * 1000, setup_bound * 1000);
    }
    free(reference);
    free_frame(&f);
    CHECK(call(DEINSTANTIATE_GL, u(0), lost).u == 0, "delete after the new device");
    if (during) CHECK(call(DEINSTANTIATE_GL, u(0), during).u == 0, "delete the gap's instance");
    if (during_try) CHECK(call(DEINSTANTIATE_GL, u(0), during_try).u == 0, "delete the try's instance");
    call(DEINSTANTIATE_GL, u(0), ram);
    if (getenv("FFGL_MINIHOST_REQUIRE_INTEROP")) {
        CHECK(lost_at, "the loss was never logged");
        CHECK(logged_message("shared frames back on a new device"), "no new device after the loss");
    }
}

/* The plugin's own device lost, by destroy(): the RAM path passes frames through unchanged without
 * waiting, opens a new device after the backoff, and renders on it again under a new pool key. */
static void case_own_device_loss(void) {
    _putenv_s("VFX_FFGL_PATH", "ram");
    void *ram = instantiate(63, 37);
    _putenv_s("VFX_FFGL_LOSE_OWN_DEVICE_AFTER", "3");
    void *losing = instantiate(63, 37);
    _putenv_s("VFX_FFGL_LOSE_OWN_DEVICE_AFTER", "");
    _putenv_s("VFX_FFGL_PATH", "");
    Frame f = frame(63, 37, 80, 48);
    /* Settings that change the image, so a frame passed through can't pass for the effect's. */
    set(ram, "Brightness", 0.2f);
    set(losing, "Brightness", 0.2f);
    wait_for_ram_device(ram, &f, 0);
    double before = own_device_number();
    char key_before[64], key_after[64];
    logged_rest("pool key ", key_before, sizeof key_before);
    CHECK(render(ram, &f), "RAM reference render");
    float *reference = malloc((size_t)f.w * f.h * 4 * sizeof(float));
    memcpy(reference, f.out, (size_t)f.w * f.h * 4 * sizeof(float));
    CHECK(max_diff(&f) > 1e-3f, "the settings left the image unchanged");
    /* A render that builds the effect on a device, at a size not rendered yet: the first render on
     * the new device does that too, on the render thread. */
    Frame other = frame(61, 35, 80, 48);
    double build_started = now();
    CHECK(render(ram, &other), "RAM render at a new size");
    double build = now() - build_started;
    free_frame(&other);
    /* Every render: the effect's frame before the loss and on the new device, the input between.
     * The first renders on the new device get three times a build, at least the bound. */
    const double bound = 0.1;
    const double build_bound = 3 * build > bound ? 3 * build : bound;
    double lost_at = 0, back_at = 0, worst = 0, worst_build = 0, worst_losing = 0;
    int after_back = 0;
    for (int n = 0; n < 5000 && after_back < 10; n++) {
        double started = now();
        CHECK(render(losing, &f), "render %d", n);
        double took = now() - started;
        /* Up to the render that finds the loss: one of them destroys the device, which only the
         * test does, and its render fails with a panic. Not bounded. */
        int losing_renders = !lost_at;
        if (!lost_at && logged_message("warning: GPU lost, passing frames through")) lost_at = now();
        if (lost_at && !back_at && own_device_number() > before) back_at = now();
        /* The render that found the new device and the one after may be the first on it. */
        if (losing_renders) {
            if (took > worst_losing) worst_losing = took;
        } else if (back_at && after_back < 2) {
            if (took > worst_build) worst_build = took;
        } else if (took > worst) {
            worst = took;
        }
        float effect = 0;
        for (int k = 0; k < f.w * f.h * 4; k++) {
            float diff = fabsf(reference[k] - f.out[k]);
            if (diff > effect) effect = diff;
        }
        if (lost_at && !back_at) {
            CHECK(max_diff(&f) < 1e-4f, "render %d in the gap wasn't the input: %g", n, max_diff(&f));
        } else if (back_at) {
            /* The render that found the new device may be its first; the ones after draw on it. */
            if (after_back++) CHECK(effect < 1e-4f, "render %d on the new device differs: %g", n, effect);
        } else {
            CHECK(effect < 1e-4f || max_diff(&f) < 1e-4f, "render %d is neither: %g", n, effect);
        }
        Sleep(2);
    }
    logged_rest("pool key ", key_after, sizeof key_after);
    printf("  worst RAM render from the loss on: %.1f ms, the first on the new device %.1f ms (a build "
           "%.1f ms), up to the loss %.1f ms, new device after %.0f ms, pool keys %s then %s\n",
           worst * 1000, worst_build * 1000, build * 1000, worst_losing * 1000,
           (back_at - lost_at) * 1000, key_before, key_after);
    CHECK(lost_at, "the loss was never logged");
    CHECK(back_at, "no new device after the loss");
    /* The first try is a second after the loss, and an open takes far less than the rest. */
    CHECK(back_at - lost_at < 4, "the new device took %.0f ms", (back_at - lost_at) * 1000);
    CHECK(key_before[0] && strcmp(key_after, key_before), "the new device kept the pool key");
    CHECK(worst < bound, "a render waited %.1f ms, over the %.0f ms bound", worst * 1000, bound * 1000);
    CHECK(worst_build < build_bound, "a first render on the new device took %.1f ms, over %.0f ms",
          worst_build * 1000, build_bound * 1000);
    free(reference);
    free_frame(&f);
    CHECK(call(DEINSTANTIATE_GL, u(0), losing).u == 0, "delete after the new device");
    call(DEINSTANTIATE_GL, u(0), ram);
}

/* A frame whose GPU wait runs out, as on a GPU that hangs without being lost: the instance goes on
 * through RAM with the right pixels, and deleting it returns. */
static void case_gpu_wait(void) {
    void *timing_out = instantiate(63, 37);
    _putenv_s("VFX_FFGL_PATH", "ram");
    void *ram = instantiate(63, 37);
    _putenv_s("VFX_FFGL_PATH", "");
    Frame f = frame(63, 37, 80, 48);
    float *reference = malloc((size_t)f.w * f.h * 4 * sizeof(float));
    wait_for_ram_device(ram, &f, 0);
    CHECK(render(ram, &f), "forced RAM reference render");
    memcpy(reference, f.out, (size_t)f.w * f.h * 4 * sizeof(float));
    /* A strict run requires one shared frame in the process, and the frames below may have none. */
    void *normal = instantiate(63, 37);
    CHECK(render(normal, &f), "render at the normal wait");
    call(DEINSTANTIATE_GL, u(0), normal);
    /* Read when the first render sets up the shared images. A wait of 0 ms can still find a frame
     * this small finished, so it takes many frames to be sure of one that isn't. */
    _putenv_s("VFX_FFGL_GPU_WAIT_MS", "0");
    for (int i = 0; i < 500; i++) {
        CHECK(render(timing_out, &f), "render %d after the wait ran out", i);
        float worst = 0;
        for (int k = 0; k < f.w * f.h * 4; k++) {
            float diff = fabsf(reference[k] - f.out[k]);
            if (diff > worst) worst = diff;
        }
        CHECK(worst < 1e-4f, "render %d differs from RAM: %g", i, worst);
        CHECK(glGetError() == GL_NO_ERROR, "render %d left a GL error", i);
    }
    _putenv_s("VFX_FFGL_GPU_WAIT_MS", "");
    free(reference);
    free_frame(&f);
    CHECK(call(DEINSTANTIATE_GL, u(0), timing_out).u == 0, "delete after the wait ran out");
    call(DEINSTANTIATE_GL, u(0), ram);
    if (getenv("FFGL_MINIHOST_REQUIRE_INTEROP")) {
        CHECK(logged_message("the GPU did not finish in time"), "the wait never ran out");
    }
}
#endif

#if defined(SEGCAM_CHECKS) && defined(_WIN32)
/* A CPU effect's output on the GPU, with the loss found by the present: frames go through RAM until
 * a new device is up, renders go on across it without waiting for the open, and the effect draws on
 * the new device. A panic there fails a render and starts the effect over. */
static void segcam_device_reopen(void) {
    _putenv_s("SEGCAM_GPU", "1");
    _putenv_s("VFX_FFGL_LOSE_DEVICE_IN_PRESENT", "30");
    int w = 1920, h = 1080;
    void *inst = instantiate(w, h);
    segcam_options(inst, 3, 0);
    Frame f = frame(w, h, w, h);
    GLuint textures[2] = {segcam_texture(w, h, 0), segcam_texture(w, h, 1)};
    GLuint original = f.input;
    /* What a render through RAM takes on this machine and load, from an instance forced there: the
     * renders in the gap go that way, and none of them may wait for the open on top. */
    _putenv_s("VFX_FFGL_PATH", "ram");
    void *ram = instantiate(w, h);
    _putenv_s("VFX_FFGL_PATH", "");
    segcam_options(ram, 3, 0);
    double ram_worst = 0;
    for (int n = 0; n < 20; n++) {
        f.input = textures[n & 1];
        double time = n / 30.0;
        CHECK(call(SET_TIME, ptr(&time), ram).u == 0, "set time failed");
        double started = now();
        CHECK(render(ram, &f), "RAM render %d failed", n);
        if (n >= 5 && now() - started > ram_worst) ram_worst = now() - started;
    }
    CHECK(call(DEINSTANTIATE_GL, u(0), ram).u == 0, "delete the RAM instance");
    int back_frames = 0;
    double waited_at = 0, worst_render = 0;
    for (int n = 0; n < 3000 && back_frames < 60; n++) {
        f.input = textures[n & 1];
        double time = n / 30.0;
        CHECK(call(SET_TIME, ptr(&time), inst).u == 0, "set time failed");
        double started = now();
        CHECK(render(inst, &f), "render %d failed", n);
        /* What the host's render thread waits for through the reopen: every render from the one
         * whose present finds the loss on. */
        double took = now() - started;
        _putenv_s("VFX_FFGL_LOSE_DEVICE_IN_PRESENT", "");
        if (!waited_at && logged_message("shared GPU render failed, using RAM")) waited_at = now();
        if (waited_at && took > worst_render) worst_render = took;
        if (logged_message("shared frames back on a new device")) back_frames++;
        Sleep(2);
    }
    /* A render that waited for the new device to open would take a RAM render plus what was left
     * of the open. The bound is three RAM renders, at least 100 ms as in device_reopen, since a
     * render that overlaps the open's Vulkan device creation is held back by the driver (see
     * device_reopen) with nothing waiting. The one render that finishes the setup on the render
     * thread adds the time the log line gives. */
    double open_ms = 0, finish_ms = 0;
    int timed = logged_entry("shared frames back on a new device: ", &open_ms)
                && logged_entry(" ms on a thread, ", &finish_ms);
    printf("  worst render through the reopen: %.1f ms, the open on its thread: %.0f ms, finishing "
           "it on the render thread: %.0f ms, a forced RAM render: %.1f ms\n",
           worst_render * 1000, open_ms, finish_ms, ram_worst * 1000);
    CHECK(waited_at, "the present never found the loss");
    CHECK(back_frames >= 60, "no new device after the loss");
    CHECK(timed, "no time for the open in the log");
    double render_bound_ms = (3 * ram_worst > 0.1 ? 3 * ram_worst : 0.1) * 1000 + finish_ms;
    CHECK(worst_render * 1000 < render_bound_ms, "a render waited %.1f ms, over %.0f ms",
          worst_render * 1000, render_bound_ms);
    /* The interop being back isn't enough: the swarm must be drawn on the new device again. The
     * note is written once per process, which holds here since this check runs alone in its own
     * process (`segcam-device-reopen` in main). */
    CHECK(logged_message("a CPU effect's frame drawn on a new device reached the host"),
          "the effect didn't draw on the new device");
    CHECK(!logged_message("CPU render failed"), "a render panicked after the new device");
    CHECK(call(DEINSTANTIATE_GL, u(0), inst).u == 0, "delete after the new device");
    glDeleteTextures(2, textures);
    f.input = original;
    free_frame(&f);
    CHECK(glGetError() == GL_NO_ERROR, "GL error after the new device");
}
#endif

typedef struct { const char *name; void (*run)(void); } Case;
static Case cases[] = {
#ifdef _WIN32
    {"context", case_context},
    /* The first case that renders: its check of the first RAM frame needs a process whose own
     * device hasn't opened yet. */
    {"sharing", case_sharing},
    /* Before gpu_wait, whose failed renders this one's log check would find. */
    {"pending_gl_error", case_pending_gl_error},
    {"gpu_wait", case_gpu_wait},
    {"device_reopen", case_device_reopen},
    {"own_device_loss", case_own_device_loss},
#endif
    {"info", case_info}, {"passthrough", case_passthrough}, {"brightness", case_brightness},
    {"padding", case_padding}, {"blur", case_blur}, {"vignette", case_vignette},
    {"state", case_state}, {"timing", case_timing},
};

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: minihost <plugin binary> [case]\n"); return 2; }
#ifdef _WIN32
    if (!create_context()) { fprintf(stderr, "no OpenGL 4.1 core context (error %lu)\n", GetLastError()); return 2; }
#else
    /* Prefer the GPU. Virtual machines (CI runners) have no accelerated OpenGL, so fall
       back to Apple's software renderer; the plugin then takes its RAM path. */
    CGLPixelFormatAttribute accelerated[] = {
        kCGLPFAOpenGLProfile, (CGLPixelFormatAttribute)kCGLOGLPVersion_GL4_Core,
        kCGLPFAAccelerated, kCGLPFAAllowOfflineRenderers, (CGLPixelFormatAttribute)0,
    };
    CGLPixelFormatAttribute software[] = {
        kCGLPFAOpenGLProfile, (CGLPixelFormatAttribute)kCGLOGLPVersion_GL4_Core,
        kCGLPFARendererID, (CGLPixelFormatAttribute)kCGLRendererGenericFloatID, (CGLPixelFormatAttribute)0,
    };
    CGLPixelFormatObj pf = NULL; GLint npf; CGLContextObj ctx;
    /* FFGL_MINIHOST_SOFTWARE=1 forces the software renderer, to test that path here. */
    int force_software = getenv("FFGL_MINIHOST_SOFTWARE") != NULL;
    if (force_software || CGLChoosePixelFormat(accelerated, &pf, &npf) || !pf) {
        pf = NULL;
        if (CGLChoosePixelFormat(software, &pf, &npf) || !pf) {
            fprintf(stderr, "no OpenGL 4.1 core context\n"); return 2;
        }
        printf("no accelerated OpenGL; using the software renderer\n");
    }
    if (CGLCreateContext(pf, NULL, &ctx)) { fprintf(stderr, "CGLCreateContext failed\n"); return 2; }
    CGLSetCurrentContext(ctx);
#endif
    printf("GL %s on %s\n", glGetString(GL_VERSION), glGetString(GL_RENDERER));

    mark_log_start();
    void *lib = open_plugin(argv[1]);
    if (!lib) { open_failed(); return 2; }
    plug = (PlugMain)plugin_symbol(lib, "plugMain");
    if (!plug) { fprintf(stderr, "no plugMain export\n"); return 2; }
    if (call(INITIALISE_V2, u(0), NULL).u != 0) { fprintf(stderr, "initialise failed\n"); return 2; }

    const char *only = argc > 2 ? argv[2] : "all";
#ifdef SEGCAM_CHECKS
    PluginInfo *info = call(GET_INFO, u(0), NULL).p;
    int segcam = info && !memcmp(info->id, "SGCM", 4);
    if (segcam && !strcmp(only, "segcam-stream")) {
        int result = segcam_stream(argc, argv);
        call(DEINITIALISE, u(0), NULL);
        return result;
    }
#ifdef _WIN32
    if (segcam && !strcmp(only, "segcam-device-reopen")) {
        segcam_device_reopen();
        printf("segcam-device-reopen: %d failure(s)\n", failures);
        call(DEINITIALISE, u(0), NULL);
        return failures ? 1 : 0;
    }
#endif
    if (segcam && (!strcmp(only, "segcam") || !strcmp(only, "all") || !strcmp(only, "segcam-timing"))) {
        if (strcmp(only, "segcam-timing")) { case_segcam(); case_state(); }
        if (strcmp(only, "segcam")) segcam_timing();
        printf("segcam: %d failure(s)\n", failures);
        call(DEINITIALISE, u(0), NULL);
        return failures ? 1 : 0;
    }
#endif
#ifndef _WIN32
    wait_for_effect();
#endif
    int ran = 0;
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        if (strcmp(only, "all") && strcmp(only, cases[i].name)) continue;
        int before = failures;
        cases[i].run();
        printf("%-12s %s\n", cases[i].name, failures == before ? "ok" : "FAILED");
        ran++;
    }
    call(DEINITIALISE, u(0), NULL);
#ifdef _WIN32
    if (getenv("FFGL_MINIHOST_REQUIRE_INTEROP")) {
        CHECK(logged_message("render path: GPU-native (Vulkan"), "required Vulkan shared path absent from this process's log");
        CHECK(!logged_message("warning: zero-copy path unavailable"), "shared setup fell back during this run");
        CHECK(!logged_message("warning: shared GPU render failed, using RAM: GL")
                  && !logged_message("the render panicked"), "shared rendering fell back during this run");
    }
#elif defined(__APPLE__)
    if (getenv("FFGL_MINIHOST_REQUIRE_INTEROP")) {
        CHECK(logged_message("render path: GPU-native (IOSurface"), "IOSurface shared path absent from this process's log");
        CHECK(!logged_message("warning: zero-copy path unavailable")
                  && !logged_message("warning: zero-copy path not set up"), "IOSurface setup fell back during this run");
        CHECK(!logged_message("warning: GPU render failed"), "IOSurface rendering fell back during this run");
    }
#endif
    if (!ran) { fprintf(stderr, "unknown case %s\n", only); return 2; }
    printf("%d case(s), %d failure(s)\n", ran, failures);
#ifdef _WIN32
    wglMakeCurrent(NULL, NULL);
    wglDeleteContext(ctx);
    ReleaseDC(window, dc);
    DestroyWindow(window);
#endif
    return failures ? 1 : 0;
}
