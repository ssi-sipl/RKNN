#include <atomic>
#include <thread>
#include <vector>
#include <iostream>
#include <chrono>
#include <cstring>
#include <opencv2/dnn.hpp>
#include "rknn_api.h"
#include <opencv2/opencv.hpp>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <SDL2/SDL.h>

struct Detection{
    int x1,y1,x2,y2;
    float score;
    int cls;
};

static std::atomic<bool> running(true);
static std::atomic<bool> frame_ready(false);
static std::atomic<bool> infer_ready(false);

static SDL_mutex* frame_mutex=nullptr;
static SDL_mutex* infer_mutex=nullptr;
static SDL_mutex* det_mutex=nullptr;

static cv::Mat shared_bgr;
static cv::Mat infer_bgr;
static std::vector<Detection> detections;

static int video_w=0;
static int video_h=0;
rknn_context rknn_ctx=0;

void sleep_ms(int ms){
 std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

void load_model(){
 FILE* fp=fopen("yolov8s.rknn","rb");
 if(!fp){std::cout<<"model missing\n"; exit(-1);} 
 fseek(fp,0,SEEK_END);
 int size=ftell(fp);
 rewind(fp);
 auto* model=(unsigned char*)malloc(size);
 fread(model,1,size,fp);
 fclose(fp);
 if(rknn_init(&rknn_ctx,model,size,0,nullptr)<0){
   std::cout<<"rknn init fail\n";
   exit(-1);
 }
 free(model);
}

void decoder_loop(const char* url){
avformat_network_init();

RECONNECT:
AVFormatContext* fmt=nullptr;
AVDictionary* opts=nullptr;
av_dict_set(&opts,"rtsp_transport","tcp",0);
av_dict_set(&opts,"buffer_size","1024000",0);

if(avformat_open_input(&fmt,url,nullptr,&opts)<0){
 sleep_ms(1000);
 goto RECONNECT;
}

avformat_find_stream_info(fmt,nullptr);

int vs=-1;
for(unsigned i=0;i<fmt->nb_streams;i++){
 if(fmt->streams[i]->codecpar->codec_type==AVMEDIA_TYPE_VIDEO){
   vs=i;
   break;
 }
}

const AVCodec* codec=avcodec_find_decoder(fmt->streams[vs]->codecpar->codec_id);
AVCodecContext* ctx=avcodec_alloc_context3(codec);
avcodec_parameters_to_context(ctx,fmt->streams[vs]->codecpar);
ctx->thread_count=1;
avcodec_open2(ctx,codec,nullptr);

AVPacket* pkt=av_packet_alloc();
AVFrame* frame=av_frame_alloc();
SwsContext* sws=nullptr;
uint8_t* bgr_buf=nullptr;
AVFrame* bgr=av_frame_alloc();

while(running){

 if(av_read_frame(fmt,pkt)<0) break;

 if(pkt->stream_index!=vs){
   av_packet_unref(pkt);
   continue;
 }

 if(avcodec_send_packet(ctx,pkt)==0){

 while(avcodec_receive_frame(ctx,frame)==0){

 if(frame->width!=video_w || frame->height!=video_h){

 video_w=frame->width;
 video_h=frame->height;

 if(sws) sws_freeContext(sws);

 sws=sws_getContext(
 video_w,video_h,
 ctx->pix_fmt,
 video_w,video_h,
 AV_PIX_FMT_RGB24,
 SWS_BILINEAR,
 nullptr,nullptr,nullptr);

 int size=av_image_get_buffer_size(
 AV_PIX_FMT_RGB24,
 video_w,
 video_h,
 1);

 if(bgr_buf) av_free(bgr_buf);

 bgr_buf=(uint8_t*)av_malloc(size);

 av_image_fill_arrays(
 bgr->data,
 bgr->linesize,
 bgr_buf,
 AV_PIX_FMT_RGB24,
 video_w,
 video_h,
 1);
 }

 sws_scale(
 sws,
 frame->data,
 frame->linesize,
 0,
 video_h,
 bgr->data,
 bgr->linesize);

 cv::Mat img(
 video_h,
 video_w,
 CV_8UC3,
 bgr->data[0],
 bgr->linesize[0]);

 cv::Mat copy=img.clone();

 SDL_LockMutex(frame_mutex);
 shared_bgr=copy;
 frame_ready=true;
 SDL_UnlockMutex(frame_mutex);

 SDL_LockMutex(infer_mutex);
 infer_bgr=copy;
 infer_ready=true;
 SDL_UnlockMutex(infer_mutex);
 }
 }

 av_packet_unref(pkt);
}

goto RECONNECT;
}

void inference_loop(){

while(running){

    if(!infer_ready){
        sleep_ms(1);
        continue;
    }

    SDL_LockMutex(infer_mutex);

    if(infer_bgr.empty()){
        SDL_UnlockMutex(infer_mutex);
        continue;
    }

    cv::Mat frame=infer_bgr.clone();

    infer_ready=false;

    SDL_UnlockMutex(infer_mutex);

    //---------------------------------
    // MATCH PYTHON EXACTLY
    //---------------------------------

    cv::Mat infer_frame;

    cv::resize(
        frame,
        infer_frame,
        cv::Size(
            960,
            540
        )
    );

    float scale=
    std::min(
        640.0f/960.0f,
        640.0f/540.0f
    );

    int nw=
    int(960*scale);

    int nh=
    int(540*scale);

    cv::Mat resized;

    cv::resize(
        infer_frame,
        resized,
        cv::Size(
            nw,
            nh
        )
    );

    cv::Mat input(
        640,
        640,
        CV_8UC3,
        cv::Scalar(
            114,
            114,
            114
        )
    );

    int left=
    (640-nw)/2;

    int top=
    (640-nh)/2;

    resized.copyTo(
        input(
            cv::Rect(
                left,
                top,
                nw,
                nh
            )
        )
    );

    //---------------------------------
    // RKNN
    //---------------------------------

    rknn_input inputs[1];

    memset(
        inputs,
        0,
        sizeof(inputs)
    );

    inputs[0].index=0;
    inputs[0].type=RKNN_TENSOR_UINT8;
    inputs[0].fmt=RKNN_TENSOR_NHWC;
    inputs[0].size=640*640*3;
    inputs[0].buf=input.data;

    if(
        rknn_inputs_set(
            rknn_ctx,
            1,
            inputs
        )<0
    ) continue;

    if(
        rknn_run(
            rknn_ctx,
            nullptr
        )<0
    ) continue;

    rknn_output outputs[1];

    memset(
        outputs,
        0,
        sizeof(outputs)
    );

    outputs[0].want_float=1;

    if(
        rknn_outputs_get(
            rknn_ctx,
            1,
            outputs,
            nullptr
        )<0
    ) continue;

    float* out=
    (float*)outputs[0].buf;

    std::vector<Detection> local;

    for(int i=0;i<8400;i++)
    {
        float x=
        out[0*8400+i];

        float y=
        out[1*8400+i];

        float w=
        out[2*8400+i];

        float h=
        out[3*8400+i];

        float debugBest=0;

        for(int c=0;c<80;c++)
        {
            float s=
            out[(c+4)*8400+i];

            if(s>debugBest)
                debugBest=s;
        }

        if(i<5)
        {
            std::cout
            << "\n==== CPP DEBUG ====\n"
            << i
            << " x=" << x
            << " y=" << y
            << " w=" << w
            << " h=" << h
            << " score=" << debugBest
            << std::endl;
        }

        float best=0;

        int cls=-1;

        for(int c=0;c<80;c++)
        {
            float s=
            out[(c+4)*8400+i];

            if(s>best)
            {
                best=s;
                cls=c;
            }
        }

        if(best<0.7)
            continue;

        //---------------------
        // undo letterbox
        //---------------------

        x=(x-left)/scale;
        y=(y-top)/scale;

        w/=scale;
        h/=scale;

        float x1=
        x-w/2;

        float y1=
        y-h/2;

        float x2=
        x+w/2;

        float y2=
        y+h/2;

        //---------------------
        // match python
        //---------------------

        float sx=
        frame.cols/960.0f;

        float sy=
        frame.rows/540.0f;

        x1*=sx;
        y1*=sy;

        x2*=sx;
        y2*=sy;

        Detection d;

        d.x1=
        std::max(
        0,
        int(x1)
        );

        d.y1=
        std::max(
        0,
        int(y1)
        );

        d.x2=
        std::min(
        video_w-1,
        int(x2)
        );

        d.y2=
        std::min(
        video_h-1,
        int(y2)
        );

        if(
        d.x2-d.x1<15 ||
        d.y2-d.y1<15
        )
        continue;

        d.score=best;
        d.cls=cls;

        local.push_back(d);
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    for(auto& d:local)
    {
        boxes.push_back(
        cv::Rect(
        d.x1,
        d.y1,
        d.x2-d.x1,
        d.y2-d.y1
        ));

        scores.push_back(
        d.score
        );
    }

    std::vector<int> idx;

    cv::dnn::NMSBoxes(
        boxes,
        scores,
        .7,
        .45,
        idx
    );

    std::vector<Detection> final_det;

    for(int i:idx)
        final_det.push_back(
        local[i]
        );

    SDL_LockMutex(det_mutex);

    detections=
    final_det;

    SDL_UnlockMutex(det_mutex);

    rknn_outputs_release(
        rknn_ctx,
        1,
        outputs
    );
}
}

int main(int argc,char* argv[]){
if(argc<2){
std::cout<<"Usage: ./rknn_rtsp rtsp://...\n";
return -1;
}

load_model();
SDL_Init(SDL_INIT_VIDEO);
frame_mutex=SDL_CreateMutex();
infer_mutex=SDL_CreateMutex();
det_mutex=SDL_CreateMutex();

std::thread dec(decoder_loop,argv[1]);
std::thread infer(inference_loop);

SDL_Window* window=SDL_CreateWindow("RKNN",100,100,1280,720,0);
SDL_Renderer* renderer=SDL_CreateRenderer(window,-1,SDL_RENDERER_ACCELERATED);
SDL_Texture* texture=nullptr;
SDL_Event e;

while(running){
while(SDL_PollEvent(&e)){
if(e.type==SDL_QUIT) running=false;
}

if(frame_ready){
SDL_LockMutex(frame_mutex);

if(!shared_bgr.empty()){

if(!texture){
texture=SDL_CreateTexture(
renderer,
SDL_PIXELFORMAT_RGB24,
SDL_TEXTUREACCESS_STREAMING,
shared_bgr.cols,
shared_bgr.rows);
}

SDL_UpdateTexture(texture,nullptr,shared_bgr.data,shared_bgr.step);
SDL_RenderClear(renderer);
SDL_RenderCopy(renderer,texture,nullptr,nullptr);

SDL_LockMutex(det_mutex);
SDL_SetRenderDrawColor(renderer,0,255,0,255);
int winW,winH;

SDL_GetWindowSize(
window,
&winW,
&winH
);

float sx=
(float)winW/video_w;

float sy=
(float)winH/video_h;

for(auto &d:detections){

SDL_Rect r;

r.x=
(int)(d.x1*sx);

r.y=
(int)(d.y1*sy);

r.w=
(int)(
(d.x2-d.x1)
*sx
);

r.h=
(int)(
(d.y2-d.y1)
*sy
);

SDL_RenderDrawRect(
renderer,
&r
);
}
SDL_UnlockMutex(det_mutex);

SDL_RenderPresent(renderer);
}
frame_ready=false;
SDL_UnlockMutex(frame_mutex);
}
SDL_Delay(1);
}
return 0;
}
