/*
 * A minimal OpenFX host for testing the plugin without Resolve or Nuke.
 *
 * It loads an .ofx binary, runs the actions a host runs (load, describe, describe in
 * context, create instance, regions of definition and interest, render), feeds it a
 * generated image and checks the pixels. Built against the real OpenFX headers, so the
 * suite layouts are the host's, not the plugin's transcription of them.
 *
 *   minihost <plugin.ofx> [case]      cases: all (default), passthrough, brightness,
 *                                     blur, scale, premult, vignette, blend, grow, depths
 * Exit status 0 when every check passes.
 */
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxParam.h"
#include "ofxProperty.h"

/* ---- Loading the plugin ----------------------------------------------------------- */

#ifdef _WIN32
static void *open_plugin(const char *path) { return (void *)LoadLibraryA(path); }
static void *plugin_symbol(void *lib, const char *name) { return (void *)GetProcAddress((HMODULE)lib, name); }
static void open_failed(void) { fprintf(stderr, "LoadLibrary failed (error %lu)\n", GetLastError()); }
#else
static void *open_plugin(const char *path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
static void *plugin_symbol(void *lib, const char *name) { return dlsym(lib, name); }
static void open_failed(void) { fprintf(stderr, "dlopen: %s\n", dlerror()); }
#endif

/* ---- Property sets ---------------------------------------------------------------- */

enum Kind { K_NONE, K_STR, K_DBL, K_INT, K_PTR };
#define MAX_VALUES 16
typedef struct {
    char name[96];
    enum Kind kind;
    int count;
    char str[MAX_VALUES][96];
    double dbl[MAX_VALUES];
    int i[MAX_VALUES];
    void *ptr[MAX_VALUES];
} Prop;

struct OfxPropertySetStruct {
    Prop props[96];
    int count;
};

static OfxPropertySetHandle props_new(void) { return calloc(1, sizeof(struct OfxPropertySetStruct)); }

static Prop *find(OfxPropertySetHandle h, const char *name, int create) {
    for (int i = 0; i < h->count; i++)
        if (!strcmp(h->props[i].name, name)) return &h->props[i];
    if (!create || h->count == 96) return NULL;
    Prop *p = &h->props[h->count++];
    snprintf(p->name, sizeof p->name, "%s", name);
    return p;
}

/* Properties real hosts reject with kOfxStatErrUnknown, so the plugin must cope.
   DaVinci Resolve 21 doesn't know the plugin description. */
static const char *unknown_props[] = {"OfxPropPluginDescription", NULL};

static int host_knows(const char *name) {
    for (const char **u = unknown_props; *u; u++)
        if (!strcmp(*u, name)) return 0;
    return 1;
}

static Prop *slot(OfxPropertySetHandle h, const char *name, int index, enum Kind kind) {
    if (!h || index < 0 || index >= MAX_VALUES || !host_knows(name)) return NULL;
    Prop *p = find(h, name, 1);
    if (!p) return NULL;
    p->kind = kind;
    if (index >= p->count) p->count = index + 1;
    return p;
}

static OfxStatus setPointer(OfxPropertySetHandle h, const char *n, int i, void *v) {
    Prop *p = slot(h, n, i, K_PTR); if (!p) return kOfxStatErrUnknown; p->ptr[i] = v; return kOfxStatOK; }
static OfxStatus setString(OfxPropertySetHandle h, const char *n, int i, const char *v) {
    Prop *p = slot(h, n, i, K_STR); if (!p) return kOfxStatErrUnknown;
    snprintf(p->str[i], sizeof p->str[i], "%s", v); return kOfxStatOK; }
static OfxStatus setDouble(OfxPropertySetHandle h, const char *n, int i, double v) {
    Prop *p = slot(h, n, i, K_DBL); if (!p) return kOfxStatErrUnknown; p->dbl[i] = v; return kOfxStatOK; }
static OfxStatus setInt(OfxPropertySetHandle h, const char *n, int i, int v) {
    Prop *p = slot(h, n, i, K_INT); if (!p) return kOfxStatErrUnknown; p->i[i] = v; return kOfxStatOK; }
static OfxStatus setPointerN(OfxPropertySetHandle h, const char *n, int c, void *const *v) {
    for (int i = 0; i < c; i++) setPointer(h, n, i, v[i]); return kOfxStatOK; }
static OfxStatus setStringN(OfxPropertySetHandle h, const char *n, int c, const char *const *v) {
    for (int i = 0; i < c; i++) setString(h, n, i, v[i]); return kOfxStatOK; }
static OfxStatus setDoubleN(OfxPropertySetHandle h, const char *n, int c, const double *v) {
    for (int i = 0; i < c; i++) setDouble(h, n, i, v[i]); return kOfxStatOK; }
static OfxStatus setIntN(OfxPropertySetHandle h, const char *n, int c, const int *v) {
    for (int i = 0; i < c; i++) setInt(h, n, i, v[i]); return kOfxStatOK; }

static Prop *get(OfxPropertySetHandle h, const char *n, int i) {
    if (!h) return NULL;
    Prop *p = find(h, n, 0);
    return (p && i >= 0 && i < p->count) ? p : NULL;
}
static OfxStatus getPointer(OfxPropertySetHandle h, const char *n, int i, void **v) {
    Prop *p = get(h, n, i); if (!p) return kOfxStatErrUnknown; *v = p->ptr[i]; return kOfxStatOK; }
static OfxStatus getString(OfxPropertySetHandle h, const char *n, int i, char **v) {
    Prop *p = get(h, n, i); if (!p) return kOfxStatErrUnknown; *v = p->str[i]; return kOfxStatOK; }
static OfxStatus getDouble(OfxPropertySetHandle h, const char *n, int i, double *v) {
    Prop *p = get(h, n, i); if (!p) return kOfxStatErrUnknown; *v = p->dbl[i]; return kOfxStatOK; }
static OfxStatus getInt(OfxPropertySetHandle h, const char *n, int i, int *v) {
    Prop *p = get(h, n, i); if (!p) return kOfxStatErrUnknown; *v = p->i[i]; return kOfxStatOK; }
static OfxStatus getPointerN(OfxPropertySetHandle h, const char *n, int c, void **v) {
    for (int i = 0; i < c; i++) if (getPointer(h, n, i, &v[i])) return kOfxStatErrUnknown; return kOfxStatOK; }
static OfxStatus getStringN(OfxPropertySetHandle h, const char *n, int c, char **v) {
    for (int i = 0; i < c; i++) if (getString(h, n, i, &v[i])) return kOfxStatErrUnknown; return kOfxStatOK; }
static OfxStatus getDoubleN(OfxPropertySetHandle h, const char *n, int c, double *v) {
    for (int i = 0; i < c; i++) if (getDouble(h, n, i, &v[i])) return kOfxStatErrUnknown; return kOfxStatOK; }
static OfxStatus getIntN(OfxPropertySetHandle h, const char *n, int c, int *v) {
    for (int i = 0; i < c; i++) if (getInt(h, n, i, &v[i])) return kOfxStatErrUnknown; return kOfxStatOK; }
static OfxStatus propReset(OfxPropertySetHandle h, const char *n) { (void)h; (void)n; return kOfxStatOK; }
static OfxStatus getDimension(OfxPropertySetHandle h, const char *n, int *c) {
    Prop *p = h ? find(h, n, 0) : NULL; if (!p) return kOfxStatErrUnknown; *c = p->count; return kOfxStatOK; }

static OfxPropertySuiteV1 property_suite = {
    setPointer, setString, setDouble, setInt, setPointerN, setStringN, setDoubleN, setIntN,
    getPointer, getString, getDouble, getInt, getPointerN, getStringN, getDoubleN, getIntN,
    propReset, getDimension,
};

/* ---- Parameters ------------------------------------------------------------------- */

typedef struct OfxParamStruct {
    char name[64];
    char type[64];
    OfxPropertySetHandle props;
    double value[4];
} ParamRec;

struct OfxParamSetStruct {
    ParamRec params[48];
    int count;
    OfxPropertySetHandle props;
};

static ParamRec *param_find(OfxParamSetHandle s, const char *name) {
    for (int i = 0; i < s->count; i++)
        if (!strcmp(s->params[i].name, name)) return &s->params[i];
    return NULL;
}

static OfxStatus paramDefine(OfxParamSetHandle s, const char *type, const char *name, OfxPropertySetHandle *props) {
    if (param_find(s, name)) return kOfxStatErrExists;
    ParamRec *p = &s->params[s->count++];
    snprintf(p->name, sizeof p->name, "%s", name);
    snprintf(p->type, sizeof p->type, "%s", type);
    p->props = props_new();
    if (props) *props = p->props;
    return kOfxStatOK;
}

static OfxStatus paramGetHandle(OfxParamSetHandle s, const char *name, OfxParamHandle *param, OfxPropertySetHandle *props) {
    ParamRec *p = param_find(s, name);
    if (!p) return kOfxStatErrUnknown;
    *param = p;
    if (props) *props = p->props;
    return kOfxStatOK;
}

static OfxStatus paramSetGetPropertySet(OfxParamSetHandle s, OfxPropertySetHandle *props) { *props = s->props; return kOfxStatOK; }
static OfxStatus paramGetPropertySet(OfxParamHandle p, OfxPropertySetHandle *props) { *props = p->props; return kOfxStatOK; }

static int components(const ParamRec *p) {
    if (!strcmp(p->type, kOfxParamTypeDouble2D)) return 2;
    if (!strcmp(p->type, kOfxParamTypeRGBA)) return 4;
    if (!strcmp(p->type, kOfxParamTypeRGB)) return 3;
    return 1;
}

static int is_int_type(const ParamRec *p) {
    return !strcmp(p->type, kOfxParamTypeBoolean) || !strcmp(p->type, kOfxParamTypeChoice) ||
           !strcmp(p->type, kOfxParamTypeInteger);
}

/* A normalised Double2D default is relative to the project size; resolve it once. */
static double project_w = 64, project_h = 48;

static void get_values(ParamRec *p, va_list ap) {
    int n = components(p);
    for (int i = 0; i < n; i++) {
        if (is_int_type(p)) *va_arg(ap, int *) = (int)p->value[i];
        else *va_arg(ap, double *) = p->value[i];
    }
}

static OfxStatus paramGetValue(OfxParamHandle p, ...) {
    va_list ap; va_start(ap, p); get_values(p, ap); va_end(ap); return kOfxStatOK; }
static OfxStatus paramGetValueAtTime(OfxParamHandle p, OfxTime t, ...) {
    va_list ap; va_start(ap, t); get_values(p, ap); va_end(ap); return kOfxStatOK; }
static OfxStatus unsupported(void) { return kOfxStatErrUnsupported; }

static OfxParameterSuiteV1 parameter_suite = {
    paramDefine, paramGetHandle, paramSetGetPropertySet, paramGetPropertySet,
    paramGetValue, paramGetValueAtTime,
    (void *)unsupported, (void *)unsupported, (void *)unsupported, (void *)unsupported,
    (void *)unsupported, (void *)unsupported, (void *)unsupported, (void *)unsupported,
    (void *)unsupported, (void *)unsupported, (void *)unsupported, (void *)unsupported,
};

/* ---- Image effect ----------------------------------------------------------------- */

typedef struct OfxImageClipStruct {
    char name[32];
    OfxPropertySetHandle props;
    OfxPropertySetHandle image; /* the image handed out for this clip */
} ClipRec;

struct OfxImageEffectStruct {
    OfxPropertySetHandle props;
    struct OfxParamSetStruct params;
    ClipRec clips[4];
    int clip_count;
};

static OfxStatus getPropertySet(OfxImageEffectHandle e, OfxPropertySetHandle *props) { *props = e->props; return kOfxStatOK; }
static OfxStatus getParamSet(OfxImageEffectHandle e, OfxParamSetHandle *s) { *s = &e->params; return kOfxStatOK; }

static ClipRec *clip_find(OfxImageEffectHandle e, const char *name) {
    for (int i = 0; i < e->clip_count; i++)
        if (!strcmp(e->clips[i].name, name)) return &e->clips[i];
    return NULL;
}

static OfxStatus clipDefine(OfxImageEffectHandle e, const char *name, OfxPropertySetHandle *props) {
    ClipRec *c = &e->clips[e->clip_count++];
    snprintf(c->name, sizeof c->name, "%s", name);
    c->props = props_new();
    *props = c->props;
    return kOfxStatOK;
}

static OfxStatus clipGetHandle(OfxImageEffectHandle e, const char *name, OfxImageClipHandle *clip, OfxPropertySetHandle *props) {
    ClipRec *c = clip_find(e, name);
    if (!c) return kOfxStatErrBadHandle;
    *clip = c;
    if (props) *props = c->props;
    return kOfxStatOK;
}

static OfxStatus clipGetPropertySet(OfxImageClipHandle c, OfxPropertySetHandle *props) { *props = c->props; return kOfxStatOK; }

static int images_out = 0;
static OfxPropertySetHandle (*temporal_image)(OfxImageClipHandle, OfxTime) = NULL;
static OfxStatus clipGetImage(OfxImageClipHandle c, OfxTime t, const OfxRectD *r, OfxPropertySetHandle *image) {
    (void)t; (void)r;
    if (temporal_image) {
        *image = temporal_image(c, t);
        if (!*image) return kOfxStatFailed;
        images_out++;
        return kOfxStatOK;
    }
    if (!c->image) return kOfxStatFailed;
    images_out++;
    *image = c->image;
    return kOfxStatOK;
}
static OfxStatus clipReleaseImage(OfxPropertySetHandle image) { (void)image; images_out--; return kOfxStatOK; }

static double source_rod[4];
static OfxStatus clipGetRegionOfDefinition(OfxImageClipHandle c, OfxTime t, OfxRectD *bounds) {
    (void)c; (void)t;
    bounds->x1 = source_rod[0]; bounds->y1 = source_rod[1];
    bounds->x2 = source_rod[2]; bounds->y2 = source_rod[3];
    return kOfxStatOK;
}
static int effect_abort(OfxImageEffectHandle e) { (void)e; return 0; }

static OfxImageEffectSuiteV1 effect_suite = {
    getPropertySet, getParamSet, clipDefine, clipGetHandle, clipGetPropertySet,
    clipGetImage, clipReleaseImage, clipGetRegionOfDefinition, effect_abort,
    (void *)unsupported, (void *)unsupported, (void *)unsupported, (void *)unsupported,
};

static const void *fetchSuite(OfxPropertySetHandle host, const char *name, int version) {
    (void)host;
    if (version != 1) return NULL;
    if (!strcmp(name, kOfxPropertySuite)) return &property_suite;
    if (!strcmp(name, kOfxImageEffectSuite)) return &effect_suite;
    if (!strcmp(name, kOfxParameterSuite)) return &parameter_suite;
    return NULL;
}

/* ---- Test driver ------------------------------------------------------------------ */

static OfxPlugin *plugin;
static int failures = 0;

#define CHECK(cond, ...) do { if (!(cond)) { failures++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static OfxStatus action(const char *name, void *handle, OfxPropertySetHandle in, OfxPropertySetHandle out) {
    return plugin->mainEntry(name, handle, in, out);
}

/* Builds an instance: describes in `context`, then copies defaults into values. */
static OfxImageEffectHandle make_instance(const char *context) {
    OfxImageEffectHandle e = calloc(1, sizeof *e);
    e->props = props_new();
    e->params.props = props_new();
    OfxStatus s = action(kOfxActionDescribe, e, NULL, NULL);
    CHECK(s == kOfxStatOK, "describe returned %d", s);
    OfxPropertySetHandle in = props_new();
    setString(in, kOfxImageEffectPropContext, 0, context);
    s = action(kOfxImageEffectActionDescribeInContext, e, in, NULL);
    CHECK(s == kOfxStatOK, "describe in context returned %d", s);
    for (int i = 0; i < e->params.count; i++) {
        ParamRec *p = &e->params.params[i];
        Prop *d = find(p->props, kOfxParamPropDefault, 0);
        for (int c = 0; d && c < d->count && c < 4; c++)
            p->value[c] = d->kind == K_INT ? d->i[c] : d->dbl[c];
        Prop *coords = find(p->props, kOfxParamPropDefaultCoordinateSystem, 0);
        if (coords && !strcmp(coords->str[0], kOfxParamCoordinatesNormalised)) {
            p->value[0] *= project_w;
            p->value[1] *= project_h;
        }
    }
    setString(e->props, kOfxImageEffectPropContext, 0, context);
    setDouble(e->props, kOfxImageEffectPropFrameRate, 0, 24.0);
    s = action(kOfxActionCreateInstance, e, NULL, NULL);
    CHECK(s == kOfxStatOK, "create instance returned %d", s);
    return e;
}

static void set_param(OfxImageEffectHandle e, const char *name, double a, double b) {
    ParamRec *p = param_find(&e->params, name);
    if (!p) { failures++; printf("  FAIL: no parameter %s\n", name); return; }
    p->value[0] = a;
    p->value[1] = b;
}

typedef struct {
    int x1, y1, x2, y2;
    const char *depth;
    int premultiplied;
    int top_down; /* rows stored top first (negative row bytes) */
    float *pixels; /* always float RGBA, rows bottom-up, for checks */
    void *data;
    size_t bpc;
} Img;

static size_t depth_bytes(const char *depth) {
    return !strcmp(depth, kOfxBitDepthByte) ? 1 : !strcmp(depth, kOfxBitDepthShort) ? 2 : 4;
}

static Img img_new(int x1, int y1, int x2, int y2, const char *depth) {
    Img m = {x1, y1, x2, y2, depth, 0, 0, NULL, NULL, depth_bytes(depth)};
    size_t n = (size_t)(x2 - x1) * (y2 - y1) * 4;
    m.pixels = calloc(n, sizeof(float));
    m.data = calloc(n, m.bpc);
    return m;
}

static int img_w(const Img *m) { return m->x2 - m->x1; }
static float *at(const Img *m, int x, int y) { return &m->pixels[((size_t)(y - m->y1) * img_w(m) + (x - m->x1)) * 4]; }

/* Memory row for pixel row y, honouring top-down storage. */
static size_t mem_row(const Img *m, int y) {
    int h = m->y2 - m->y1;
    return m->top_down ? (size_t)(h - 1 - (y - m->y1)) : (size_t)(y - m->y1);
}

static void store(Img *m) {
    for (int y = m->y1; y < m->y2; y++)
        for (int x = m->x1; x < m->x2; x++) {
            float *p = at(m, x, y);
            size_t i = (mem_row(m, y) * img_w(m) + (x - m->x1)) * 4;
            for (int c = 0; c < 4; c++) {
                float v = p[c];
                if (m->premultiplied && c < 3) v *= p[3];
                if (m->bpc == 1) ((unsigned char *)m->data)[i + c] = (unsigned char)lroundf(fminf(fmaxf(v, 0), 1) * 255);
                else if (m->bpc == 2) ((unsigned short *)m->data)[i + c] = (unsigned short)lroundf(fminf(fmaxf(v, 0), 1) * 65535);
                else ((float *)m->data)[i + c] = v;
            }
        }
}

static void load(Img *m) {
    for (int y = m->y1; y < m->y2; y++)
        for (int x = m->x1; x < m->x2; x++) {
            float *p = at(m, x, y);
            size_t i = (mem_row(m, y) * img_w(m) + (x - m->x1)) * 4;
            for (int c = 0; c < 4; c++)
                p[c] = m->bpc == 1 ? ((unsigned char *)m->data)[i + c] / 255.0f
                     : m->bpc == 2 ? ((unsigned short *)m->data)[i + c] / 65535.0f
                     : ((float *)m->data)[i + c];
            if (m->premultiplied && p[3] > 0)
                for (int c = 0; c < 3; c++) p[c] /= p[3];
        }
}

static OfxPropertySetHandle image_props(Img *m) {
    OfxPropertySetHandle h = props_new();
    int w = img_w(m);
    long row = (long)w * 4 * (long)m->bpc;
    char *base = m->data;
    /* The data pointer is the bottom-left pixel; top-down storage walks rows backwards. */
    if (m->top_down) base += (size_t)(m->y2 - m->y1 - 1) * row;
    setPointer(h, kOfxImagePropData, 0, base);
    int bounds[4] = {m->x1, m->y1, m->x2, m->y2};
    setIntN(h, kOfxImagePropBounds, 4, bounds);
    setInt(h, kOfxImagePropRowBytes, 0, (int)(m->top_down ? -row : row));
    setString(h, kOfxImageEffectPropPixelDepth, 0, m->depth);
    setString(h, kOfxImageEffectPropComponents, 0, kOfxImageComponentRGBA);
    setString(h, kOfxImageEffectPropPreMultiplication, 0, m->premultiplied ? kOfxImagePreMultiplied : kOfxImageUnPreMultiplied);
    return h;
}

/* A gradient with an alpha ramp on the right. */
static void fill_pattern(Img *m) {
    for (int y = m->y1; y < m->y2; y++)
        for (int x = m->x1; x < m->x2; x++) {
            float *p = at(m, x, y);
            p[0] = (float)(x - m->x1) / (img_w(m) - 1);
            p[1] = (float)(y - m->y1) / (m->y2 - m->y1 - 1);
            p[2] = ((x / 8 + y / 8) % 2) ? 0.8f : 0.2f;
            p[3] = x - m->x1 < img_w(m) * 3 / 4 ? 1.0f : 0.5f;
        }
}

typedef struct {
    Img source, output, layer;
    int has_layer;
    double scale;
} Frame;

static OfxStatus render(OfxImageEffectHandle e, Frame *f, int window[4]) {
    store(&f->source);
    clip_find(e, kOfxImageEffectSimpleSourceClipName)->image = image_props(&f->source);
    clip_find(e, kOfxImageEffectOutputClipName)->image = image_props(&f->output);
    ClipRec *layer = clip_find(e, "Layer");
    if (layer) {
        setInt(layer->props, kOfxImageClipPropConnected, 0, f->has_layer);
        if (f->has_layer) { store(&f->layer); layer->image = image_props(&f->layer); }
    }
    OfxPropertySetHandle in = props_new();
    setDouble(in, kOfxPropTime, 0, 0);
    setIntN(in, kOfxImageEffectPropRenderWindow, 4, window);
    double scale[2] = {f->scale, f->scale};
    setDoubleN(in, kOfxImageEffectPropRenderScale, 2, scale);
    setString(in, kOfxImageEffectPropFieldToRender, 0, kOfxImageFieldNone);
    OfxStatus s = action(kOfxImageEffectActionRender, e, in, NULL);
    load(&f->output);
    return s;
}

static Frame frame(int w, int h, const char *depth) {
    Frame f = {img_new(0, 0, w, h, depth), img_new(0, 0, w, h, depth), img_new(0, 0, w, h, depth), 0, 1.0};
    fill_pattern(&f.source);
    store(&f.source);
    load(&f.source); /* quantise the reference to the depth */
    return f;
}

static int full[4] = {0, 0, 64, 48};

static float max_diff(const Img *a, const Img *b) {
    float worst = 0;
    for (int y = a->y1; y < a->y2; y++)
        for (int x = a->x1; x < a->x2; x++)
            for (int c = 0; c < 4; c++) {
                float d = fabsf(at(a, x, y)[c] - at(b, x, y)[c]);
                if (d > worst) worst = d;
            }
    return worst;
}

static void case_passthrough(void) {
    OfxImageEffectHandle e = make_instance(kOfxImageEffectContextFilter);
    Frame f = frame(64, 48, kOfxBitDepthFloat);
    OfxStatus s = render(e, &f, full);
    CHECK(s == kOfxStatOK, "render returned %d", s);
    CHECK(max_diff(&f.source, &f.output) < 1e-5, "defaults changed the image (max diff %g)", max_diff(&f.source, &f.output));
    CHECK(images_out == 0, "%d images not released", images_out);
}

static void case_brightness(void) {
    OfxImageEffectHandle e = make_instance(kOfxImageEffectContextFilter);
    set_param(e, "brightness", 0.25, 0);
    Frame f = frame(64, 48, kOfxBitDepthFloat);
    render(e, &f, full);
    float want = at(&f.source, 10, 5)[0] + 0.25f;
    float got = at(&f.output, 10, 5)[0];
    CHECK(fabsf(got - want) < 1e-5, "red at (10,5) is %g, expected %g", got, want);
    /* Rows must not be flipped: green rises with y. */
    CHECK(at(&f.output, 10, 40)[1] > at(&f.output, 10, 5)[1], "image came back upside down");
}

static void case_blur(void) {
    OfxImageEffectHandle e = make_instance(kOfxImageEffectContextFilter);
    set_param(e, "blur_radius", 4, 0);
    Frame f = frame(64, 48, kOfxBitDepthFloat);
    render(e, &f, full);
    /* The checker's edge at x = 8 softens. */
    float edge = at(&f.output, 8, 4)[2];
    CHECK(edge > 0.25f && edge < 0.75f, "blue at the checker edge is %g, expected a blend", edge);
    /* Interior of the opaque area stays opaque. */
    CHECK(fabsf(at(&f.output, 20, 20)[3] - 1) < 1e-4, "alpha changed inside the opaque area");
}

/* Opaque white on the left half, black on the right: one vertical edge. */
static void fill_edge(Img *m) {
    for (int y = m->y1; y < m->y2; y++)
        for (int x = m->x1; x < m->x2; x++) {
            float *p = at(m, x, y);
            p[0] = p[1] = p[2] = x - m->x1 < img_w(m) / 2 ? 1.0f : 0.0f;
            p[3] = 1;
        }
}

/* Pixels in a row that are part way between black and white. */
static int soft_pixels(const Img *m, int y) {
    int n = 0;
    for (int x = m->x1; x < m->x2; x++) { float v = at(m, x, y)[0]; if (v > 0.02f && v < 0.98f) n++; }
    return n;
}

static void case_scale(void) {
    /* At half render scale the same Blur Radius must soften half as many pixels. */
    OfxImageEffectHandle e = make_instance(kOfxImageEffectContextFilter);
    set_param(e, "blur_radius", 8, 0);
    Frame big = frame(64, 48, kOfxBitDepthFloat);
    fill_edge(&big.source);
    render(e, &big, full);
    Frame half = frame(32, 24, kOfxBitDepthFloat);
    fill_edge(&half.source);
    half.scale = 0.5;
    int w[4] = {0, 0, 32, 24};
    render(e, &half, w);
    int wide = soft_pixels(&big.output, 20), narrow = soft_pixels(&half.output, 10);
    CHECK(wide > 8 && abs(wide - 2 * narrow) <= 2,
          "soft pixels: %d at full scale, %d at half (expected half)", wide, narrow);
}

static void case_premult(void) {
    OfxImageEffectHandle e = make_instance(kOfxImageEffectContextFilter);
    Frame f = frame(64, 48, kOfxBitDepthFloat);
    f.source.premultiplied = f.output.premultiplied = 1;
    render(e, &f, full);
    CHECK(max_diff(&f.source, &f.output) < 1e-5, "premultiplied round trip changed the image (%g)", max_diff(&f.source, &f.output));
}

static void case_vignette(void) {
    OfxImageEffectHandle e = make_instance(kOfxImageEffectContextFilter);
    set_param(e, "vignette_enable", 1, 0);
    set_param(e, "brightness", 0, 0);
    Frame f = frame(64, 48, kOfxBitDepthFloat);
    for (int y = 0; y < 48; y++) for (int x = 0; x < 64; x++) { float *p = at(&f.source, x, y); p[0] = p[1] = p[2] = p[3] = 1; }
    render(e, &f, full);
    CHECK(at(&f.output, 32, 24)[0] > 0.99f, "centre darkened (%g)", at(&f.output, 32, 24)[0]);
    CHECK(at(&f.output, 0, 0)[0] < 0.8f, "corner not darkened (%g)", at(&f.output, 0, 0)[0]);
    /* Move the centre to the bottom-left quadrant (canonical y up): (16, 12). */
    set_param(e, "vignette_center", 16, 12);
    render(e, &f, full);
    CHECK(at(&f.output, 16, 12)[0] > 0.99f, "moved centre darkened (%g)", at(&f.output, 16, 12)[0]);
    CHECK(at(&f.output, 63, 47)[0] < at(&f.output, 0, 0)[0], "far corner should be darker than near one");
}

static void case_blend(void) {
    OfxImageEffectHandle e = make_instance(kOfxImageEffectContextGeneral);
    CHECK(clip_find(e, "Layer") != NULL, "general context has no Layer clip");
    set_param(e, "blend_mode", 1, 0); /* Add */
    set_param(e, "blend_amount", 1, 0);
    Frame f = frame(64, 48, kOfxBitDepthFloat);
    f.has_layer = 1;
    for (int y = 0; y < 48; y++) for (int x = 0; x < 64; x++) { float *p = at(&f.layer, x, y); p[0] = 0; p[1] = 0; p[2] = 0.1f; p[3] = 1; }
    render(e, &f, full);
    float want = at(&f.source, 10, 10)[2] + 0.1f;
    CHECK(fabsf(at(&f.output, 10, 10)[2] - want) < 1e-4, "add blend blue %g, expected %g", at(&f.output, 10, 10)[2], want);
    /* Unconnected layer: no change. */
    f.has_layer = 0;
    render(e, &f, full);
    CHECK(max_diff(&f.source, &f.output) < 1e-5, "unconnected layer changed the image");
}

static void case_grow(void) {
    OfxImageEffectHandle e = make_instance(kOfxImageEffectContextFilter);
    set_param(e, "blur_radius", 5, 0);
    set_param(e, "blur_grow", 1, 0);
    source_rod[0] = 0; source_rod[1] = 0; source_rod[2] = 64; source_rod[3] = 48;
    OfxPropertySetHandle in = props_new(), out = props_new();
    setDouble(in, kOfxPropTime, 0, 0);
    double scale[2] = {1, 1};
    setDoubleN(in, kOfxImageEffectPropRenderScale, 2, scale);
    OfxStatus s = action(kOfxImageEffectActionGetRegionOfDefinition, e, in, out);
    double rod[4] = {0};
    getDoubleN(out, kOfxImageEffectPropRegionOfDefinition, 4, rod);
    CHECK(s == kOfxStatOK && rod[0] == -5 && rod[3] == 53, "RoD with Blur Edges is %g %g %g %g (status %d)", rod[0], rod[1], rod[2], rod[3], s);

    /* Output bigger than the source by the blur: pixels outside get soft alpha. */
    Frame f = {img_new(0, 0, 64, 48, kOfxBitDepthFloat), img_new(-5, -5, 69, 53, kOfxBitDepthFloat), img_new(0, 0, 1, 1, kOfxBitDepthFloat), 0, 1};
    fill_pattern(&f.source);
    int window[4] = {-5, -5, 69, 53};
    render(e, &f, window);
    CHECK(at(&f.output, -2, 20)[3] > 0.05f, "no alpha outside the left edge (%g)", at(&f.output, -2, 20)[3]);
    CHECK(fabsf(at(&f.output, -2, 20)[3] - at(&f.output, 20, -2)[3]) < 0.05f, "growth differs left vs bottom");

    set_param(e, "blur_grow", 0, 0);
    s = action(kOfxImageEffectActionGetRegionOfDefinition, e, in, out);
    CHECK(s == kOfxStatReplyDefault, "RoD without Blur Edges should be the default (got %d)", s);

    OfxPropertySetHandle roi_in = props_new(), roi_out = props_new();
    setDouble(roi_in, kOfxPropTime, 0, 0);
    double roi[4] = {10, 10, 20, 20};
    setDoubleN(roi_in, kOfxImageEffectPropRegionOfInterest, 4, roi);
    setDoubleN(roi_in, kOfxImageEffectPropRenderScale, 2, scale);
    action(kOfxImageEffectActionGetRegionsOfInterest, e, roi_in, roi_out);
    double got[4] = {0};
    getDoubleN(roi_out, "OfxImageClipPropRoI_Source", 4, got);
    CHECK(got[0] == 5 && got[2] == 25, "source RoI %g..%g, expected 5..25", got[0], got[2]);
}

static void case_depths(void) {
    const char *depths[] = {kOfxBitDepthByte, kOfxBitDepthShort, kOfxBitDepthFloat};
    for (int d = 0; d < 3; d++) {
        OfxImageEffectHandle e = make_instance(kOfxImageEffectContextFilter);
        set_param(e, "brightness", 0.1, 0);
        Frame f = frame(64, 48, depths[d]);
        f.source.top_down = f.output.top_down = (d == 1); /* also try negative row bytes */
        render(e, &f, full);
        float tol = d == 0 ? 1.0f / 255 : 1e-4f;
        float want = fminf(at(&f.source, 30, 30)[0] + 0.1f, 1);
        CHECK(fabsf(at(&f.output, 30, 30)[0] - want) <= tol, "%s: red %g, expected %g", depths[d], at(&f.output, 30, 30)[0], want);
        CHECK(at(&f.output, 10, 40)[1] > at(&f.output, 10, 5)[1], "%s: image upside down", depths[d]);
    }
}

typedef struct { const char *name; void (*run)(void); } Case;
static Case cases[] = {
    {"passthrough", case_passthrough}, {"brightness", case_brightness}, {"blur", case_blur},
    {"scale", case_scale}, {"premult", case_premult}, {"vignette", case_vignette},
    {"blend", case_blend}, {"grow", case_grow}, {"depths", case_depths},
};

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: minihost <plugin.ofx> [case]\n"); return 2; }
    void *lib = open_plugin(argv[1]);
    if (!lib) { open_failed(); return 2; }
    int (*count)(void) = plugin_symbol(lib, "OfxGetNumberOfPlugins");
    OfxPlugin *(*getp)(int) = plugin_symbol(lib, "OfxGetPlugin");
    if (!count || !getp || count() != 1) { fprintf(stderr, "missing or unexpected OFX entry points\n"); return 2; }
    plugin = getp(0);
    printf("plugin %s %s v%u.%u\n", plugin->pluginApi, plugin->pluginIdentifier, plugin->pluginVersionMajor, plugin->pluginVersionMinor);
    if (strcmp(plugin->pluginApi, kOfxImageEffectPluginApi) || plugin->apiVersion != 1) { fprintf(stderr, "wrong plugin API\n"); return 2; }

    static OfxHost host;
    host.host = props_new();
    host.fetchSuite = fetchSuite;
    plugin->setHost(&host);
    if (action(kOfxActionLoad, NULL, NULL, NULL) != kOfxStatOK) { fprintf(stderr, "load failed\n"); return 2; }

    const char *only = argc > 2 ? argv[2] : "all";
    int ran = 0;
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        if (strcmp(only, "all") && strcmp(only, cases[i].name)) continue;
        int before = failures;
        cases[i].run();
        printf("%-12s %s\n", cases[i].name, failures == before ? "ok" : "FAILED");
        ran++;
    }
    action(kOfxActionUnload, NULL, NULL, NULL);
    if (!ran) { fprintf(stderr, "unknown case %s\n", only); return 2; }
    printf("%d case(s), %d failure(s)\n", ran, failures);
    return failures ? 1 : 0;
}
