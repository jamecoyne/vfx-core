/* Use the Preview host's suites, against the official OpenFX headers. */
#ifndef _WIN32
/* nanosleep, which -std=c11 hides on glibc without it. */
#define _POSIX_C_SOURCE 200809L
#endif
#define main preview_main
#include "../../../../vfx-core/wrappers/ofx/minihost/minihost.c"
#include <time.h>
#include "ofxMessage.h"
#undef main

static Img history[32], output;
static OfxPropertySetHandle history_props[32], output_props;
static int reads;
static OfxHost host;
/* Resolve turns temporal clip access off, and is the host earlier frames are trusted in. */
static int as_resolve;
/* The plugin keeps what it computed between renders, so it may read frames in any order and
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
    set_param(e, "detect", 0, 0);
    set_param(e, "eyes", 0, 0);
    set_param(e, "mouth", 0, 0);
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
/* Inverts frames `from` to `to`; a second call changes them back. */
static void edit_frames(int from, int to) {
    for(int t=from;t<=to;t++) {
        unsigned char *data=history[t].data;
        for(size_t i=0;i<(size_t)320*180*4;i++) if(i%4!=3) data[i]^=0xff;
    }
}
/* Call after a kept instance's renders: a cold render would stretch the gap a stream needs. */
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
/* A step back or a repeat checks earlier frames, and so does the render after it; other steps
   stream, reading the frames stepped over, since none are kept ahead, and the few a check dropped.
   Each frame's last render equals fresh. */
static void replay_order(int mode, const char *name, const int *order, int n, int from, int to) {
    OfxImageEffectHandle e=instance(mode,0);
    for(int t=6;t<=23;t++) render_frame(e,t);
    for(int t=10;t<=13;t++) render_frame(e,t);
    edit_frames(from,to);
    static unsigned char rendered[16][320*180*4];
    int previous=13, broke_stream=0;
    for(int i=0;i<n;i++) {
        int t=order[i];
        begin_sequence(e,t,t);
        render_frame(e,t);
        int read=reads;
        if(t>previous && !broke_stream) CHECK(read<=3 || read<=t-previous+1,"%s, mode %d, edit %d to %d: a stream read %d frames at %d",name,mode,from,to,read,t);
        else CHECK(read>2,"%s, mode %d, edit %d to %d: a check read %d frames at %d",name,mode,from,to,read,t);
        broke_stream=t<=previous;
        previous=t;
        memcpy(rendered[i],output.data,sizeof rendered[i]);
    }
    for(int i=0;i<n;i++) {
        int t=order[i], again=0;
        for(int j=i+1;j<n;j++) again|=order[j]==t;
        if(!again) CHECK(same_as_fresh(mode,t,rendered[i]),"%s, mode %d, edit %d to %d: frame %d differs from a fresh render",name,mode,from,to,t);
    }
    edit_frames(from,to);
}
/* 1 MB is four of these 320x180 8 bit frames. NULL takes the budget away again. */
#define BUDGET_FRAMES 4
static void set_budget(const char *mb) {
#ifdef _WIN32
    _putenv_s("VFX_OFX_FETCH_BUDGET_MB", mb ? mb : "");
#else
    if (mb) setenv("VFX_OFX_FETCH_BUDGET_MB", mb, 1); else unsetenv("VFX_OFX_FETCH_BUDGET_MB");
#endif
}
/* Under a fetch budget a stream stays exact. After a stop and an edit each render reads at most
   the budget, and the held frame matches a fresh render within frames / budget + 1 renders. The
   fresh renders go without a budget, which a cold render would spread over several renders. */
static void within_budget(int mode) {
    static unsigned char streamed[24][320*180*4];
    OfxImageEffectHandle e=instance(mode,0);
    set_budget("1");
    for(int t=0;t<24;t++) {
        render_frame(e,t);
        CHECK(reads<=BUDGET_FRAMES+1,"mode %d: a stream under a budget read %d frames at %d",mode,reads,t);
        memcpy(streamed[t],output.data,sizeof streamed[t]);
    }
    set_budget(NULL);
    for(int t=0;t<24;t++) CHECK(same_as_fresh(mode,t,streamed[t]),"mode %d: frame %d of a stream under a budget differs from a fresh render",mode,t);

    edit_frames(0,21);
    int renders=(23+BUDGET_FRAMES-1)/BUDGET_FRAMES+1, caught_up=0;
    for(int i=0;i<renders && !caught_up;i++) {
        set_budget("1");
        render_frame(e,23);
        CHECK(reads<=BUDGET_FRAMES+1,"mode %d: a check under a budget read %d frames",mode,reads);
        memcpy(streamed[0],output.data,sizeof streamed[0]);
        set_budget(NULL);
        caught_up=same_as_fresh(mode,23,streamed[0]);
    }
    CHECK(caught_up,"mode %d: frame 23 differs from a fresh render after %d renders under a budget",mode,renders);
    /* Changing the frames back is another edit. A jump to 12 and a stream from there, as an
       export from an In point: the stream checks what the jump left unread. */
    edit_frames(0,21);
    set_budget("1");
    for(int t=12;t<24;t++) {
        render_frame(e,t);
        /* A stream fetches the current frame twice around its checks: to see the source didn't
           start over, and to draw it. */
        CHECK(reads<=BUDGET_FRAMES+2,"mode %d: a stream after a jump read %d frames at %d",mode,reads,t);
        memcpy(streamed[t],output.data,sizeof streamed[t]);
    }
    set_budget(NULL);
    /* After an edit the stream reads what the edit left unchecked as well as the frames it has
       nothing kept for, and the frame before each render joins those. Here it matches within the
       held bound. The defaults read back to frame 0, so the history doesn't slide here;
       vfx_host::contract::check streams cold over more frames than a render reads. */
    for(int t=12+renders-1;t<24;t++) CHECK(same_as_fresh(mode,t,streamed[t]),"mode %d: frame %d, render %d of a stream after a jump, differs from a fresh render",mode,t,t-11);
}
static int export_after_edit(void) {
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
    /* The first two were seen on macOS: the frame at the playhead, then a step back to the range.
       `late_step_back` is from an older note, kept as the harder case. */
    static const int swapped_start[]={23,15,14,16,17,18,19,20,21,22,23,14,14,14};
    static const int forward_start[]={23,14,15,16,17,18,19,20,21,22,23,14,14,14};
    static const int late_step_back[]={14,14,14,16,15,17,18,19,20,21,22,23};
#define REPLAY(order,from,to) replay_order(mode,#order,order,sizeof order/sizeof *order,from,to)
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

        REPLAY(swapped_start,0,13);
        REPLAY(swapped_start,15,15);
        REPLAY(forward_start,0,13);
        REPLAY(forward_start,15,15);
        REPLAY(late_step_back,0,13);
        REPLAY(late_step_back,15,15);
        within_budget(mode);
    }
    printf("export after edit: %d failures\n",failures);
    return failures?1:0;
}
/* The message suite: how many errors failed renders showed, and the last. */
static char last_error[512];
static int errors_shown;
static OfxStatus record_error(const char *type, const char *format, va_list args) {
    CHECK(!strcmp(type, kOfxMessageError), "message type %s", type);
    vsnprintf(last_error, sizeof last_error, format, args);
    errors_shown++;
    return kOfxStatOK;
}
static OfxStatus message(void *handle, const char *type, const char *id, const char *format, ...) {
    (void)handle; (void)id;
    va_list args; va_start(args, format);
    OfxStatus status = record_error(type, format, args);
    va_end(args);
    return status;
}
static OfxMessageSuiteV1 message_suite = { message };
static const void *fetch_suite(OfxPropertySetHandle host_props, const char *name, int version) {
    if (!strcmp(name, kOfxMessageSuite) && version == 1) return &message_suite;
    return fetchSuite(host_props, name, version);
}
/* Waits `ms` milliseconds between a failed render and its retry. */
static void wait_ms(int ms) {
#ifdef _WIN32
    Sleep(ms);
#else
    struct timespec wait={ms/1000,(long)(ms%1000)*1000000L};
    nanosleep(&wait,NULL);
#endif
}
/* argv: plugin input output mode depth top premult order, where order is 0 forwards, 1 backwards,
   2 shuffled, 3 every frame cold on a fresh instance, and 4 forwards rendering a failed frame
   again for up to 10 s, as an editor does. With VFX_OFX_REMOVE_DEVICE_AFTER set, order 4 checks
   that no render failed and no error was shown, since the plugin waits for the next device.
   Frames are written in frame order.
   Or argv: plugin export, for export_after_edit. */
int main(int argc, char **argv) {
    if (argc != 9 && argc != 3) return 2;
    void *lib = open_plugin(argv[1]);
    if (!lib) { open_failed(); return 2; }
    OfxPlugin *(*getp)(int) = plugin_symbol(lib,"OfxGetPlugin");
    plugin = getp(0);
    CHECK(!strcmp(plugin->pluginIdentifier,"com.lofiscifi.segcam"),"wrong identity");
    host.host=props_new(); host.fetchSuite=fetch_suite; plugin->setHost(&host);
    CHECK(action(kOfxActionLoad,NULL,NULL,NULL)==kOfxStatOK,"load failed");
    if (argc == 3) return export_after_edit();
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
    int retries=0;
    const char *removal=getenv("VFX_OFX_REMOVE_DEVICE_AFTER");
    int removal_at=removal?atoi(removal):-1;
    OfxImageEffectHandle e=instance(mode,premult);
    for(int i=0;i<32;i++) {
        int t=frames[i];
        if(order==3) e=instance(mode,premult);
        OfxPropertySetHandle args=props_new(), needed=props_new();
        setDouble(args,kOfxPropTime,0,t);
        int window[4]={0,0,320,180}; setIntN(args,kOfxImageEffectPropRenderWindow,4,window);
        CHECK(action(kOfxImageEffectActionGetFramesNeeded,e,args,needed)==kOfxStatOK,"frames needed failed");
        Prop *range=find(needed,"OfxImageClipPropFrameRange_Source",0);
        int back=mode==3?123:64;
        CHECK(range && range->dbl[0]==t-back && range->dbl[1]==t,"wrong frame range");
        reads=0;
        /* Frames drawn while the face models load can differ by a level or two from run to
           run, so the removal runs start once they're up. */
        if(order>=4 && i==0) wait_ms(3000);
        /* Paced as playback is, so segcam's device opens before the removal. */
        if(order>=4) wait_ms(100);
        OfxStatus status=action(kOfxImageEffectActionRender,e,args,NULL);
        for(int tries=0;status!=kOfxStatOK && order==4 && tries<200;tries++) {
            retries++; wait_ms(50); reads=0;
            status=action(kOfxImageEffectActionRender,e,args,NULL);
        }
        CHECK(status==kOfxStatOK,"render failed at %d",t);
        /* The plugin renders the frame that meets a removal twice, reading its frames again. */
        int most=((t<back?t:back)+1)*(removal_at==t?2:1);
        CHECK(images_out==0,"image leak"); CHECK(reads<=most,"read %d frames at %d",reads,t);
        output.premultiplied=0; load(&output); output.premultiplied=premult;
        unsigned char *frame=rendered[t];
        for(int y=179;y>=0;y--) for(int x=0;x<320;x++) {
            float *p=at(&output,x,y); int order4[4]={3,0,1,2};
            for(int c=0;c<4;c++) *frame++=(unsigned char)(fminf(1,fmaxf(0,p[order4[c]]))*255+0.5f);
        }
    }
    if(order==4 && getenv("VFX_OFX_REMOVE_DEVICE_AFTER")) {
        CHECK(retries==0,"%d renders failed after the device was removed",retries);
        CHECK(errors_shown==0,"the removal showed an error: %s",last_error);
    }
    if(order==4) printf("renders tried again: %d\n",retries);
    fwrite(rendered,1,sizeof rendered,out);
    fclose(out); action(kOfxActionUnload,NULL,NULL,NULL);
    printf("mode %d depth %s top %d premult %d order %d: %d failures\n",mode,depth,top,premult,order,failures);
    return failures?1:0;
}
