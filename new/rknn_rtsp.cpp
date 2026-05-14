#include <atomic>
#include <thread>
#include <vector>
#include <iostream>
#include <chrono>
#include <cstring>

#include "rknn_api.h"

#include <opencv2/opencv.hpp>

extern "C"{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include <SDL2/SDL.h>

struct Detection{
    int x1;
    int y1;
    int x2;
    int y2;
    float score;
    int cls;
};

static std::atomic<bool> running{true};

static SDL_mutex* frame_mutex=nullptr;
static SDL_mutex* infer_mutex=nullptr;
static SDL_mutex* det_mutex=nullptr;

static AVFrame* shared_frame=nullptr;
static AVFrame* infer_frame=nullptr;

static std::vector<Detection> detections;

static int video_w=0;
static int video_h=0;

static std::atomic<bool> frame_ready{false};
static std::atomic<bool> infer_ready{false};

rknn_context rknn_ctx=0;

void sleep_ms(int ms){
    std::this_thread::sleep_for(
        std::chrono::milliseconds(ms)
    );
}

void load_model(){

    FILE* fp=fopen(
        "yolov8s.rknn",
        "rb"
    );

    if(!fp){

        std::cout<<"model missing\n";
        exit(-1);
    }

    fseek(fp,0,SEEK_END);

    int size=ftell(fp);

    rewind(fp);

    unsigned char* model=
    (unsigned char*)malloc(size);

    fread(
        model,
        1,
        size,
        fp
    );

    fclose(fp);

    if(
      rknn_init(
      &rknn_ctx,
      model,
      size,
      0,
      nullptr
      )<0
    ){

        std::cout<<"rknn init fail\n";

        exit(-1);
    }

    free(model);
}

void decoder_loop(
const char* url
){

avformat_network_init();

RECONNECT:

AVFormatContext* fmt=nullptr;

AVDictionary* opts=nullptr;

av_dict_set(
&opts,
"rtsp_transport",
"tcp",
0
);

av_dict_set(
&opts,
"buffer_size",
"1024000",
0
);

av_dict_set(
&opts,
"stimeout",
"3000000",
0
);

if(
avformat_open_input(
&fmt,
url,
nullptr,
&opts
)<0){

sleep_ms(1000);

goto RECONNECT;
}

avformat_find_stream_info(
fmt,
nullptr
);

int vs=-1;

for(
unsigned i=0;
i<fmt->nb_streams;
i++
){

if(
fmt->streams[i]
->codecpar
->codec_type
==
AVMEDIA_TYPE_VIDEO
){

vs=i;

break;
}
}

const AVCodec* codec=
avcodec_find_decoder(
fmt->streams[vs]
->codecpar
->codec_id
);

AVCodecContext* ctx=
avcodec_alloc_context3(
codec
);

avcodec_parameters_to_context(
ctx,
fmt->streams[vs]
->codecpar
);

ctx->flags|=
AV_CODEC_FLAG_LOW_DELAY;

ctx->thread_count=1;

avcodec_open2(
ctx,
codec,
nullptr
);

AVPacket* pkt=
av_packet_alloc();

AVFrame* frame=
av_frame_alloc();

AVFrame* yuv=
av_frame_alloc();

uint8_t* yuv_buf=nullptr;

SwsContext* sws=nullptr;

while(running){

if(
av_read_frame(
fmt,
pkt
)<0
){

break;
}

if(
pkt->stream_index
!=vs
){

av_packet_unref(pkt);

continue;
}

if(
avcodec_send_packet(
ctx,
pkt
)==0
){

while(
avcodec_receive_frame(
ctx,
frame
)==0
){

if(
frame->width
!=video_w
||
frame->height
!=video_h
){

video_w=
frame->width;

video_h=
frame->height;

if(sws)
sws_freeContext(
sws
);

sws=sws_getContext(

video_w,
video_h,

ctx->pix_fmt,

video_w,
video_h,

AV_PIX_FMT_YUV420P,

SWS_FAST_BILINEAR,

nullptr,
nullptr,
nullptr
);

if(yuv_buf)
av_free(
yuv_buf
);

int sz=
av_image_get_buffer_size(
AV_PIX_FMT_YUV420P,
video_w,
video_h,
1
);

yuv_buf=
(uint8_t*)
av_malloc(sz);

av_image_fill_arrays(

yuv->data,
yuv->linesize,

yuv_buf,

AV_PIX_FMT_YUV420P,

video_w,
video_h,

1
);
}

sws_scale(

sws,

frame->data,
frame->linesize,

0,
video_h,

yuv->data,
yuv->linesize
);

SDL_LockMutex(
frame_mutex
);

av_frame_unref(
shared_frame
);

av_frame_ref(
shared_frame,
yuv
);

frame_ready=true;

SDL_UnlockMutex(
frame_mutex
);

SDL_LockMutex(
infer_mutex
);

av_frame_unref(
infer_frame
);

av_frame_ref(
infer_frame,
yuv
);

infer_ready=true;

SDL_UnlockMutex(
infer_mutex
);

}
}

av_packet_unref(
pkt
);

}

goto RECONNECT;
}

void inference_loop(){

    AVFrame* local_frame=av_frame_alloc();

    while(running){

        if(!infer_ready){

            sleep_ms(1);
            continue;
        }

        SDL_LockMutex(infer_mutex);

        if(
            !infer_frame ||
            !infer_frame->data[0] ||
            video_w<=0 ||
            video_h<=0
        ){

            SDL_UnlockMutex(infer_mutex);

            sleep_ms(5);

            continue;
        }

        // Deep copy frame reference
        av_frame_unref(local_frame);

        if(
            av_frame_ref(
                local_frame,
                infer_frame
            )<0
        ){

            SDL_UnlockMutex(
                infer_mutex
            );

            continue;
        }

        SDL_UnlockMutex(
            infer_mutex
        );

        if(
            !local_frame->data[0]
        ){

            continue;
        }

        cv::Mat yuv(

            video_h+
            video_h/2,

            video_w,

            CV_8UC1,

            local_frame->data[0]

        );

        if(
            yuv.empty()
        ){

            continue;
        }

        cv::Mat bgr;

        try{

            cv::cvtColor(

                yuv,

                bgr,

                cv::COLOR_YUV2BGR_I420

            );

        }catch(...){

            continue;
        }

        if(
            bgr.empty()
        ){

            continue;
        }

        cv::Mat input;

        cv::resize(

            bgr,

            input,

            cv::Size(
                640,
                640
            )

        );

        cv::cvtColor(

            input,

            input,

            cv::COLOR_BGR2RGB
        );

        rknn_input inputs[1];

        memset(
            inputs,
            0,
            sizeof(inputs)
        );

        inputs[0].index=0;
        inputs[0].type=
            RKNN_TENSOR_UINT8;

        inputs[0].fmt=
            RKNN_TENSOR_NHWC;

        inputs[0].size=
            640*640*3;

        inputs[0].buf=
            input.data;

        if(
            rknn_inputs_set(
                rknn_ctx,
                1,
                inputs
            )<0
        ){

            continue;
        }

        if(
            rknn_run(
                rknn_ctx,
                nullptr
            )<0
        ){

            continue;
        }

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
        ){

            continue;
        }

        float* out=
            (float*)outputs[0].buf;

        std::vector<Detection> local;

        for(
            int i=0;
            i<8400;
            i++
        ){

            float x=
                out[i];

            float y=
                out[8400+i];

            float w=
                out[16800+i];

            float h=
                out[25200+i];

            float best=0;
            int cls=-1;

            for(
                int c=4;
                c<84;
                c++
            ){

                float s=
                out[
                c*8400+i
                ];

                if(
                    s>best
                ){

                    best=s;
                    cls=c-4;
                }
            }

            if(
                best<0.45
            )
            continue;

            Detection d;

            d.x1=
            (x-w/2)
            *video_w/640;

            d.y1=
            (y-h/2)
            *video_h/640;

            d.x2=
            (x+w/2)
            *video_w/640;

            d.y2=
            (y+h/2)
            *video_h/640;

            d.score=best;

            d.cls=cls;

            local.push_back(d);
        }

        SDL_LockMutex(
            det_mutex
        );

        detections=
            local;

        SDL_UnlockMutex(
            det_mutex
        );

        rknn_outputs_release(
            rknn_ctx,
            1,
            outputs
        );

        infer_ready=false;
    }

    av_frame_free(
        &local_frame
    );
}

int main(
int argc,
char* argv[]
){

if(argc<2){

std::cout
<<"usage\n";

return -1;
}

load_model();

SDL_Init(
SDL_INIT_VIDEO
);

frame_mutex=
SDL_CreateMutex();

infer_mutex=
SDL_CreateMutex();

det_mutex=
SDL_CreateMutex();

shared_frame=
av_frame_alloc();

infer_frame=
av_frame_alloc();

std::thread dec(
decoder_loop,
argv[1]
);

std::thread infer(
inference_loop
);

SDL_Window* window=
SDL_CreateWindow(

"RKNN",

SDL_WINDOWPOS_CENTERED,

SDL_WINDOWPOS_CENTERED,

1280,
720,

SDL_WINDOW_RESIZABLE
);

SDL_Renderer* renderer=
SDL_CreateRenderer(

window,

-1,

SDL_RENDERER_ACCELERATED
);

SDL_Texture*
texture=nullptr;

SDL_Event e;

while(running){

while(
SDL_PollEvent(
&e
)
){

if(
e.type==
SDL_QUIT
)
running=false;
}

if(
frame_ready
){

SDL_LockMutex(
frame_mutex
);

if(
!texture
){

texture=
SDL_CreateTexture(

renderer,

SDL_PIXELFORMAT_IYUV,

SDL_TEXTUREACCESS_STREAMING,

video_w,

video_h
);
}

SDL_UpdateYUVTexture(

texture,

nullptr,

shared_frame->data[0],
shared_frame->linesize[0],

shared_frame->data[1],
shared_frame->linesize[1],

shared_frame->data[2],
shared_frame->linesize[2]
);

SDL_UnlockMutex(
frame_mutex
);

SDL_RenderClear(
renderer
);

SDL_RenderCopy(
renderer,
texture,
nullptr,
nullptr
);

SDL_LockMutex(
det_mutex
);

SDL_SetRenderDrawColor(
renderer,
0,
255,
0,
255
);

for(
auto& d:
detections
){

SDL_Rect r;

r.x=d.x1;
r.y=d.y1;

r.w=d.x2-d.x1;
r.h=d.y2-d.y1;

SDL_RenderDrawRect(
renderer,
&r
);
}

SDL_UnlockMutex(
det_mutex
);

SDL_RenderPresent(
renderer
);

frame_ready=false;
}

SDL_Delay(1);
}

dec.join();

infer.join();

return 0;
}