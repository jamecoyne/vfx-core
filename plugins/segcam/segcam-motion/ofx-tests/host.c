/* Use the Preview host's suites, against the official OpenFX headers. */
#define main preview_main
#include "../../../../vfx-core/wrappers/ofx/minihost/minihost.c"
#undef main

static Img history[32], output;
static OfxPropertySetHandle history_props[32], output_props;
static int reads;
/* The plugin keeps what it worked out between renders, so it may read frames in any order and
   hold the current frame while it reads another. */
static OfxPropertySetHandle temporal(OfxImageClipHandle c, OfxTime t) {
    if (!strcmp(c->name, "Output")) return output_props;
    CHECK(t >= 0 && t < 32 && t == (int)t, "bad frame time %g", t);
    CHECK(images_out <= 1, "more than two source images held (%d)", images_out + 1);
    reads++;
    return t >= 0 && t < 32 ? history_props[(int)t] : NULL;
}
static OfxImageEffectHandle instance(int mode, int premult) {
    OfxImageEffectHandle e = make_instance(kOfxImageEffectContextFilter);
    set_param(e, "output", mode, 0);
    ClipRec *src = clip_find(e, "Source");
    setDouble(src->props, kOfxImageEffectPropFrameRange, 0, 0);
    setDouble(src->props, kOfxImageEffectPropFrameRange, 1, 31);
    setString(src->props, kOfxImageEffectPropPreMultiplication, 0, premult ? kOfxImagePreMultiplied : kOfxImageUnPreMultiplied);
    CHECK(find(e->props, kOfxImageEffectPropTemporalClipAccess, 0)->i[0] == 1, "effect temporal disabled");
    CHECK(find(src->props, kOfxImageEffectPropTemporalClipAccess, 0)->i[0] == 1, "clip temporal disabled");
    return e;
}
/* argv: plugin input output mode depth top premult order, where order is 0 forwards, 1 backwards,
   2 shuffled, 3 every frame cold on a fresh instance. Frames are written in frame order. */
int main(int argc, char **argv) {
    if (argc != 9) return 2;
    void *lib = open_plugin(argv[1]);
    if (!lib) { open_failed(); return 2; }
    OfxPlugin *(*getp)(int) = plugin_symbol(lib,"OfxGetPlugin");
    plugin = getp(0);
    CHECK(!strcmp(plugin->pluginIdentifier,"com.lofiscifi.SegcamMotionRust"),"wrong identity");
    static OfxHost host; host.host=props_new(); host.fetchSuite=fetchSuite; plugin->setHost(&host);
    CHECK(action(kOfxActionLoad,NULL,NULL,NULL)==kOfxStatOK,"load failed");
    int mode=atoi(argv[4]), top=atoi(argv[6]), premult=atoi(argv[7]), order=atoi(argv[8]);
    const char *depth=atoi(argv[5])==8?kOfxBitDepthByte:atoi(argv[5])==16?kOfxBitDepthShort:kOfxBitDepthFloat;
    FILE *in=fopen(argv[2],"rb"), *out=fopen(argv[3],"wb");
    if (!in || !out) return 2;
    for(int t=0;t<32;t++) {
        history[t]=img_new(0,0,320,180,depth); history[t].top_down=top;
        history[t].premultiplied=premult;
        for(int y=179;y>=0;y--) for(int x=0;x<320;x++) {
            unsigned char argb[4]; if(fread(argb,1,4,in)!=4) return 2;
            float *p=at(&history[t],x,y);
            p[0]=argb[1]/255.f; p[1]=argb[2]/255.f; p[2]=argb[3]/255.f; p[3]=argb[0]/255.f;
            /* Raw fixture already uses the desired alpha convention. */
        }
        int saved=history[t].premultiplied; history[t].premultiplied=0; store(&history[t]); history[t].premultiplied=saved;
        history_props[t]=image_props(&history[t]);
    }
    fclose(in);
    output=img_new(0,0,320,180,depth); output.top_down=top; output.premultiplied=premult; output_props=image_props(&output);
    temporal_image=temporal;
    int frames[32];
    for(int t=0;t<32;t++) frames[t]=order==1?31-t:t;
    if(order==2) {
        unsigned seed=12345;
        for(int i=31;i>0;i--) { seed=seed*1103515245u+12345u; int j=(int)((seed>>16)%(unsigned)(i+1)); int s=frames[i]; frames[i]=frames[j]; frames[j]=s; }
    }
    static unsigned char rendered[32][320*180*4];
    OfxImageEffectHandle e=instance(mode,premult);
    for(int i=0;i<32;i++) {
        int t=frames[i];
        if(order==3) e=instance(mode,premult);
        OfxPropertySetHandle args=props_new(), needed=props_new();
        setDouble(args,kOfxPropTime,0,t);
        int window[4]={0,0,320,180}; setIntN(args,kOfxImageEffectPropRenderWindow,4,window);
        CHECK(action(kOfxImageEffectActionGetFramesNeeded,e,args,needed)==kOfxStatOK,"frames needed failed");
        Prop *range=find(needed,"OfxImageClipPropFrameRange_Source",0);
        int back=mode==3?39:16;
        CHECK(range && range->dbl[0]==t-back && range->dbl[1]==t,"wrong frame range");
        reads=0;
        CHECK(action(kOfxImageEffectActionRender,e,args,NULL)==kOfxStatOK,"render failed at %d",t);
        CHECK(images_out==0,"image leak"); CHECK(reads<=(t<back?t:back)+1,"read %d frames at %d",reads,t);
        output.premultiplied=0; load(&output); output.premultiplied=premult;
        unsigned char *frame=rendered[t];
        for(int y=179;y>=0;y--) for(int x=0;x<320;x++) {
            float *p=at(&output,x,y); int order4[4]={3,0,1,2};
            for(int c=0;c<4;c++) *frame++=(unsigned char)(fminf(1,fmaxf(0,p[order4[c]]))*255+0.5f);
        }
    }
    fwrite(rendered,1,sizeof rendered,out);
    fclose(out); action(kOfxActionUnload,NULL,NULL,NULL);
    printf("mode %d depth %s top %d premult %d order %d: %d failures\n",mode,depth,top,premult,order,failures);
    return failures?1:0;
}
