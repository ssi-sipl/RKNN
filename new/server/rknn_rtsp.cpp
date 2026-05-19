#include <atomic>
#include <thread>
#include <vector>
#include <iostream>
#include <chrono>
#include <cstring>

#include <opencv2/opencv.hpp>

extern "C"{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

struct Detection{
    int x1,y1,x2,y2;
    float score;
    int cls;
    char label[64];
};

static std::atomic<bool> running(true);
static std::atomic<bool> frame_ready(false);
static std::atomic<bool> infer_ready(false);

SDL_mutex* frame_mutex=nullptr;
SDL_mutex* infer_mutex=nullptr;
SDL_mutex* det_mutex=nullptr;

cv::Mat shared_bgr;
cv::Mat infer_bgr;

std::vector<Detection> detections;

int video_w=0;
int video_h=0;

int sockfd=-1;

void sleep_ms(int ms){
    std::this_thread::sleep_for(
        std::chrono::milliseconds(ms)
    );
}

void connect_server(){

    sockfd=socket(
        AF_INET,
        SOCK_STREAM,
        0
    );

    sockaddr_in server;

    server.sin_family=AF_INET;
    server.sin_port=htons(9000);

    inet_pton(
        AF_INET,
        "127.0.0.1",
        &server.sin_addr
    );

    while(
        connect(
            sockfd,
            (sockaddr*)&server,
            sizeof(server)
        )<0
    ){
        std::cout
        <<"waiting server...\n";

        sleep_ms(1000);
    }

    std::cout
    <<"connected\n";
}

void decoder_loop(const char* url){

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

if(
avformat_open_input(
&fmt,
url,
nullptr,
&opts
)<0
){
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

AVFrame* bgr=
av_frame_alloc();

uint8_t* bgr_buf=nullptr;

SwsContext* sws=nullptr;

while(running){

if(
av_read_frame(
fmt,
pkt
)<0
) break;

if(
pkt->stream_index!=vs
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
frame->width!=video_w
||
frame->height!=video_h
){

video_w=frame->width;
video_h=frame->height;

sws=
sws_getContext(
video_w,
video_h,
ctx->pix_fmt,
video_w,
video_h,
AV_PIX_FMT_RGB24,
SWS_FAST_BILINEAR,
0,0,0
);

int size=
av_image_get_buffer_size(
AV_PIX_FMT_RGB24,
video_w,
video_h,
1
);

bgr_buf=
(uint8_t*)
av_malloc(size);

av_image_fill_arrays(
bgr->data,
bgr->linesize,
bgr_buf,
AV_PIX_FMT_RGB24,
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
bgr->data,
bgr->linesize
);

cv::Mat img(
video_h,
video_w,
CV_8UC3,
bgr->data[0]
);

cv::Mat copy=
img.clone();

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

void client_inference_loop(){

while(running){

if(!infer_ready){

sleep_ms(1);

continue;
}

SDL_LockMutex(infer_mutex);

cv::Mat frame=
infer_bgr.clone();

infer_ready=false;

SDL_UnlockMutex(infer_mutex);

std::vector<uchar> buf;

cv::imencode(
".jpg",
frame,
buf
);

int sz=buf.size();

send(
sockfd,
&sz,
sizeof(int),
0
);

send(
sockfd,
buf.data(),
sz,
0
);

int n;

recv(
sockfd,
&n,
sizeof(int),
MSG_WAITALL
);

std::vector<Detection>
det(n);

recv(
sockfd,
det.data(),
n*sizeof(Detection),
MSG_WAITALL
);

SDL_LockMutex(
det_mutex
);

detections=det;

SDL_UnlockMutex(
det_mutex
);
}
}

int main(int argc,char* argv[]){

if(argc<2){

std::cout
<<"./rknn_rtsp rtsp_url\n";

return -1;
}

connect_server();

SDL_Init(
SDL_INIT_VIDEO
);

TTF_Init();

TTF_Font* font=
TTF_OpenFont(
"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
16
);

frame_mutex=
SDL_CreateMutex();

infer_mutex=
SDL_CreateMutex();

det_mutex=
SDL_CreateMutex();

std::thread dec(
decoder_loop,
argv[1]
);

std::thread infer(
client_inference_loop
);

SDL_Window* window=
SDL_CreateWindow(
"RKNN",
100,
100,
1280,
720,
0
);

SDL_Renderer* renderer=
SDL_CreateRenderer(
window,
-1,
SDL_RENDERER_ACCELERATED
);

SDL_Texture* texture=nullptr;

while(running){

SDL_Event e;

while(SDL_PollEvent(&e))
if(e.type==SDL_QUIT)
running=false;

if(frame_ready){

SDL_LockMutex(
frame_mutex
);

if(!texture){

texture=
SDL_CreateTexture(
renderer,
SDL_PIXELFORMAT_RGB24,
SDL_TEXTUREACCESS_STREAMING,
shared_bgr.cols,
shared_bgr.rows
);
}

SDL_UpdateTexture(
texture,
0,
shared_bgr.data,
shared_bgr.step
);

SDL_RenderClear(
renderer
);

SDL_RenderCopy(
renderer,
texture,
0,
0
);

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

SDL_LockMutex(
det_mutex
);

for(auto& d:detections){

SDL_Rect r;

r.x=d.x1*sx;
r.y=d.y1*sy;

r.w=(d.x2-d.x1)*sx;
r.h=(d.y2-d.y1)*sy;

SDL_SetRenderDrawColor(
renderer,
0,
255,
0,
255
);

SDL_RenderDrawRect(
renderer,
&r
);

std::string text=
std::string(d.label)
+" "
+std::to_string(
d.score
).substr(0,4);

SDL_Color c={0,255,0};

SDL_Surface* s=
TTF_RenderText_Blended(
font,
text.c_str(),
c
);

SDL_Texture* t=
SDL_CreateTextureFromSurface(
renderer,
s
);

SDL_Rect tr={
r.x,
r.y-20,
s->w,
s->h
};

SDL_RenderCopy(
renderer,
t,
0,
&tr
);

SDL_FreeSurface(s);

SDL_DestroyTexture(t);
}

SDL_UnlockMutex(
det_mutex
);

SDL_RenderPresent(
renderer
);

frame_ready=false;

SDL_UnlockMutex(
frame_mutex
);
}

SDL_Delay(1);
}

close(sockfd);

}