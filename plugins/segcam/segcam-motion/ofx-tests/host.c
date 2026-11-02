/* Use the Preview host's suites, against the official OpenFX headers. */
#define main preview_main
#include "../../../../vfx-core/wrappers/ofx/minihost/minihost.c"
#undef main

static Img history[32], output;
static OfxPropertySetHandle history_props[32], output_props;
static int reads;
static OfxHost host;
/* Resolve turns temporal clip access off, and is the host earlier frames are trusted in. */
static int as_resolve;
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
    if (!as_resolve) {
        CHECK(find(e->props, kOfxImageEffectPropTemporalClipAccess, 0)->i[0] == 1, "effect temporal disabled");
        CHECK(find(src->props, kOfxImageEffectPropTemporalClipAccess, 0)->i[0] == 1, "clip temporal disabled");
    }
    return e;
}
static void render_frame(OfxImageEffectHandle e, int t) {
    OfxPropertySetHandle args=props_new();
    setDouble(args,kOfxPropTime,0,t);
    int window[4]={0,0,320,180}; setIntN(args,kOfxImageEffectPropRenderWindow,4,window);
    reads=0;
    CHECK(action(kOfxImageEffectActionRender,e,args,NULL)==kOfxStatOK,"render failed at %d",t);
    CHECK(images_out==0,"image leak");
}
/* Changes frames `from` to `to` or changes them back, as a grade upstream would. */
static void edit_frames(int from, int to) {
    for(int t=from;t<=to;t++) {
        unsigned char *data=history[t].data;
        for(size_t i=0;i<(size_t)320*180*4;i++) if(i%4!=3) data[i]^=0xff;
    }
}
/* Whether `pixels` is what a fresh instance renders of frame t. A kept instance's renders come
   first, and these after, so a cold render doesn't stretch the gap a stream depends on. */
static int same_as_fresh(int mode, int t, const unsigned char *pixels) {
    OfxImageEffectHandle fresh=instance(mode,0);
    render_frame(fresh,t);
    return !memcmp(pixels,output.data,(size_t)320*180*4);
}
static void begin_sequence(OfxImageEffectHandle e, int first, int last) {
    OfxPropertySetHandle sequence=props_new();
    setDouble(sequence,kOfxImageEffectPropFrameRange,0,first);
    setDouble(sequence,kOfxImageEffectPropFrameRange,1,last);
    CHECK(action(kOfxImageEffectActionBeginSequenceRender,e,sequence,NULL)==kOfxStatOK,"begin sequence failed");
}
/* Plays frames 6 to 23, plays again up to 13, edits frames `from` to `to`, and renders `order`
   straight after, a begin for one frame before each render, as Resolve does. A step back or a
   repeat checks earlier frames, and so does the render after it; any other step forward streams,
   reading the frame and the few a check dropped. Every frame's last render equals a fresh one. */
static void replay(int mode, const char *name, const int *order, int n, int from, int to) {
    OfxImageEffectHandle e=instance(mode,0);
    for(int t=6;t<=23;t++) render_frame(e,t);
    for(int t=10;t<=13;t++) render_frame(e,t);
    edit_frames(from,to);
    static unsigned char rendered[16][320*180*4];
    int before=13, jumped=0;
    for(int i=0;i<n;i++) {
        int t=order[i];
        begin_sequence(e,t,t);
        render_frame(e,t);
        int read=reads;
        if(t>before && !jumped) CHECK(read<=3,"%s, mode %d, edit %d to %d: a stream read %d frames at %d",name,mode,from,to,read,t);
        else CHECK(read>2,"%s, mode %d, edit %d to %d: a check read %d frames at %d",name,mode,from,to,read,t);
        jumped=t<=before;
        before=t;
        memcpy(rendered[i],output.data,sizeof rendered[i]);
    }
    for(int i=0;i<n;i++) {
        int t=order[i], again=0;
        for(int j=i+1;j<n;j++) again|=order[j]==t;
        if(!again) CHECK(same_as_fresh(mode,t,rendered[i]),"%s, mode %d, edit %d to %d: frame %d differs from a fresh render",name,mode,from,to,t);
    }
    edit_frames(from,to);
}
/* As Resolve: a stream of renders takes earlier frames as unchanged, and a sequence render
   straight after an edit checks them on its first frame, then streams again. A host's begin for
   one frame at a time leaves the stream as it was. */
static int edit_then_deliver(void) {
    as_resolve=1;
    setString(host.host,kOfxPropName,0,"DaVinci Resolve");
    for(int t=0;t<32;t++) {
        history[t]=img_new(0,0,320,180,kOfxBitDepthByte);
        for(int y=0;y<180;y++) for(int x=0;x<320;x++) {
            float v=20+t*6<=x && x<60+t*6 && 60<=y && y<105 ? .86f : .09f;
            float *p=at(&history[t],x,y); p[0]=p[1]=p[2]=v; p[3]=1;
        }
        store(&history[t]);
        history_props[t]=image_props(&history[t]);
    }
    output=img_new(0,0,320,180,kOfxBitDepthByte); output_props=image_props(&output);
    temporal_image=temporal;
    static unsigned char streamed[2][320*180*4], sequence[3][320*180*4];
    /* The orders Resolve 21 starts a Deliver in, from frame 14: on macOS the last frame first, and
       on Windows the first frame three times. */
    static const int mac[]={23,15,14,16,17,18,19,20,21,22,23,14,14,14};
    static const int win[]={14,14,14,16,15,17,18,19,20,21,22,23};
    int mac_n=sizeof mac/sizeof *mac, win_n=sizeof win/sizeof *win;
    for(int m=0;m<2;m++) {
        int mode=m?3:0;
        OfxImageEffectHandle e=instance(mode,0);
        render_frame(e,20); render_frame(e,21);
        edit_frames(0,21);
        render_frame(e,22);
        CHECK(reads==1,"mode %d: a stream read %d frames",mode,reads);
        memcpy(streamed[0],output.data,sizeof streamed[0]);
        render_frame(e,23);
        memcpy(streamed[1],output.data,sizeof streamed[1]);
        CHECK(!same_as_fresh(mode,22,streamed[0]),"mode %d: a stream checked earlier frames",mode);
        edit_frames(0,21);

        e=instance(mode,0);
        render_frame(e,20); render_frame(e,21);
        edit_frames(0,21);
        for(int t=22;t<24;t++) {
            begin_sequence(e,t,t);
            render_frame(e,t);
            CHECK(reads==1,"mode %d: a begin for frame %d alone read %d frames",mode,t,reads);
            CHECK(!memcmp(output.data,streamed[t-22],sizeof streamed[0]),"mode %d: a begin for frame %d alone changed the stream",mode,t);
        }
        edit_frames(0,21);

        e=instance(mode,0);
        render_frame(e,20); render_frame(e,21);
        edit_frames(0,21);
        begin_sequence(e,22,31);
        render_frame(e,22);
        CHECK(reads>1,"mode %d: the first frame of a sequence read %d frames",mode,reads);
        memcpy(sequence[0],output.data,sizeof sequence[0]);
        render_frame(e,23);
        CHECK(reads>1,"mode %d: the sequence's next frame read %d frames",mode,reads);
        memcpy(sequence[1],output.data,sizeof sequence[1]);
        render_frame(e,24);
        CHECK(reads==1,"mode %d: the frame after read %d frames",mode,reads);
        memcpy(sequence[2],output.data,sizeof sequence[2]);
        CHECK(same_as_fresh(mode,22,sequence[0]),"mode %d: the first frame of a sequence missed the edit",mode);
        CHECK(same_as_fresh(mode,23,sequence[1]),"mode %d: the sequence's next frame differs",mode);
        CHECK(same_as_fresh(mode,24,sequence[2]),"mode %d: the frame after differs",mode);
        edit_frames(0,21);

        replay(mode,"macOS order",mac,mac_n,0,13);
        replay(mode,"macOS order",mac,mac_n,15,15);
        replay(mode,"Windows order",win,win_n,0,13);
        replay(mode,"Windows order",win,win_n,15,15);
    }
    printf("edit then deliver: %d failures\n",failures);
    return failures?1:0;
}
/* argv: plugin input output mode depth top premult order, where order is 0 forwards, 1 backwards,
   2 shuffled, 3 every frame cold on a fresh instance. Frames are written in frame order.
   Or argv: plugin deliver, for edit_then_deliver. */
int main(int argc, char **argv) {
    if (argc != 9 && argc != 3) return 2;
    void *lib = open_plugin(argv[1]);
    if (!lib) { open_failed(); return 2; }
    OfxPlugin *(*getp)(int) = plugin_symbol(lib,"OfxGetPlugin");
    plugin = getp(0);
    CHECK(!strcmp(plugin->pluginIdentifier,"com.lofiscifi.SegcamMotionRust"),"wrong identity");
    host.host=props_new(); host.fetchSuite=fetchSuite; plugin->setHost(&host);
    CHECK(action(kOfxActionLoad,NULL,NULL,NULL)==kOfxStatOK,"load failed");
    if (argc == 3) return edit_then_deliver();
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
        int back=mode==3?75:19;
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
