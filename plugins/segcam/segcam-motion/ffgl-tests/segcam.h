/* segcam Motion's sequence and timing checks, using the host above. */
static void segcam_options(void *inst, int mode, int custom) {
    set(inst, "Output", (float)mode);
    if (custom) {
        set(inst, "Sensitivity", 90);
        set(inst, "Adaptation", 12.5f);
        set(inst, "Background", 4);
        set(inst, "Minimum size", 0.125f);
        set(inst, "Max segments", 8);
        set(inst, "Grid width", 96);
        set(inst, "Mirror", 1);
        set(inst, "Labels", 0);
        set(inst, "Trail frames", 5);
        set(inst, "Swarm max tiles", 12);
    }
}

/* Raw files are top-down straight ARGB in, top-down premultiplied ARGB out.
   Comparing the latter retains the host's alpha convention, including zero alpha. */
static int segcam_stream(int argc, char **argv) {
    if (argc != 10) {
        fprintf(stderr, "segcam-stream WIDTH HEIGHT MODE INPUT OUTPUT PAD VARIANT\n");
        return 2;
    }
    int w = atoi(argv[3]), h = atoi(argv[4]), mode = atoi(argv[5]), pad = atoi(argv[8]);
    if (w < 2 || h < 2 || w > 4096 || h > 4096 || mode < 0 || mode > 3) return 2;
    FILE *in = fopen(argv[6], "rb"), *out = fopen(argv[7], "wb");
    if (!in || !out) { fprintf(stderr, "cannot open sequence files\n"); return 2; }
    int custom = !strcmp(argv[9], "custom"), rewind = !strcmp(argv[9], "rewind");
    void *inst = instantiate(w, h);
    segcam_options(inst, mode, custom);
    if (!strcmp(argv[9], "resize")) {
        Frame old = frame(29, 17, 32, 32);
        for (int i = 0; i < 5; i++) CHECK(render(inst, &old), "resize prelude failed");
        free_frame(&old);
    }
    Frame f = frame(w, h, w + (pad ? 13 : 0), h + (pad ? 9 : 0));
    size_t bytes = (size_t)w * h * 4;
    unsigned char *raw = malloc(bytes);
    float *texels = malloc((size_t)f.hw * f.hh * 4 * sizeof(float));
    for (size_t i = 0; i < (size_t)f.hw * f.hh * 4; i++) texels[i] = 9.0f;
    int n = 0;
    double elapsed = 0;
    while (fread(raw, 1, bytes, in) == bytes) {
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
            unsigned char *s = raw + ((size_t)y * w + x) * 4;
            float *d = px(texels, f.hw, x, h - 1 - y);
            d[3] = s[0] / 255.0f;
            for (int c = 0; c < 3; c++) d[c] = (s[c + 1] / 255.0f) * d[3];
        }
        glBindTexture(GL_TEXTURE_2D, f.input);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, f.hw, f.hh, 0, GL_RGBA, GL_FLOAT, texels);
        glBindTexture(GL_TEXTURE_2D, 0);
        double time = (rewind && n >= 24 ? n - 24 : n) / 30.0;
        CHECK(call(SET_TIME, ptr(&time), inst).u == 0, "set time failed");
        double start = now();
        CHECK(render(inst, &f), "sequence frame %d failed", n);
        elapsed += now() - start;
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
            float *s = px(f.out, w, x, h - 1 - y);
            unsigned char *d = raw + ((size_t)y * w + x) * 4;
            d[0] = (unsigned char)roundf(fminf(1, fmaxf(0, s[3])) * 255);
            for (int c = 0; c < 3; c++)
                d[c + 1] = (unsigned char)roundf(fminf(1, fmaxf(0, s[c] * s[3])) * 255);
        }
        if (fwrite(raw, 1, bytes, out) != bytes) { CHECK(0, "output write failed"); break; }
        n++;
    }
    CHECK(n > 0 && !ferror(in), "empty or invalid input");
    printf("  %d frames %dx%d, mode %d, padded %d, %s: %.2f ms/frame (including host readback)\n",
           n, w, h, mode, pad, argv[9], elapsed * 1000 / (n ? n : 1));
    CHECK(call(DEINSTANTIATE_GL, u(0), inst).u == 0, "delete sequence instance");
    check_context();
    CHECK(glGetError() == GL_NO_ERROR, "GL error after sequence");
    free_frame(&f); free(raw); free(texels);
    fclose(in);
    CHECK(fclose(out) == 0, "close output failed");
    return failures ? 1 : 0;
}

static GLuint segcam_texture(int w, int h, int side) {
    float *pixels = calloc((size_t)w * h * 4, sizeof(float));
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
        float *p = px(pixels, w, x, y);
        int moving = x > (side ? w * 3 / 5 : w / 5) && x < (side ? w * 4 / 5 : w * 2 / 5)
                     && y > h / 4 && y < h / 2;
        p[0] = moving ? 0.9f : 0.05f;
        p[1] = moving ? 0.8f : 0.1f;
        p[2] = moving ? 0.7f : 0.15f;
        p[3] = 1;
    }
    GLuint tex = make_texture(w, h, pixels);
    free(pixels);
    return tex;
}

static void segcam_timing(void) {
    const char *names[] = {"Overlay", "Motion matte", "Box matte", "Swarm"};
    int sizes[][2] = {{1280, 720}, {1920, 1080}};
    for (int s = 0; s < 2; s++) {
        int w = sizes[s][0], h = sizes[s][1];
        Frame f = frame(w, h, w, h);
        GLuint textures[2] = {segcam_texture(w, h, 0), segcam_texture(w, h, 1)};
        for (int mode = 0; mode < 4; mode++) {
            void *inst = instantiate(w, h);
            segcam_options(inst, mode, 0);
            double elapsed = 0;
            /* Forty warm-ups fill even Swarm's default history; only process + finish
               is timed, excluding host uploads and output readback. */
            for (int i = 0; i < 70; i++) {
                Texture tex = {(uint32_t)w, (uint32_t)h, (uint32_t)w, (uint32_t)h, textures[i % 2]};
                Texture *list[1] = {&tex};
                Process p = {1, list, f.target.fbo};
                glBindFramebuffer(GL_FRAMEBUFFER, f.target.fbo);
                glViewport(0, 0, w, h);
                double time = i / 30.0;
                call(SET_TIME, ptr(&time), inst);
                double start = now();
                CHECK(call(PROCESS_OPENGL, ptr(&p), inst).u == 0, "timed segcam process failed");
                glFinish();
                if (i >= 40) elapsed += now() - start;
                check_context();
            }
            double ms = elapsed * 1000 / 30;
            printf("  %dx%d %-12s %.2f ms/frame (%.1f fps), 30 frames after 40 warm-ups\n", w, h, names[mode], ms, 1000 / ms);
            call(DEINSTANTIATE_GL, u(0), inst);
        }
#ifdef _WIN32
        typedef void (APIENTRY *DeleteTextures)(GLsizei, const GLuint *);
        ((DeleteTextures)gl_proc("glDeleteTextures"))(2, textures);
#else
        glDeleteTextures(2, textures);
#endif
        free_frame(&f);
    }
}
