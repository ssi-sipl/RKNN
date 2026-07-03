#include <iostream>
#include <thread>
#include <vector>
#include <fstream>
#include <cstring>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <future>
#include <memory>
#include <atomic>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "rknn_api.h"

#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

struct Detection{
    int x1,y1,x2,y2;
    float score;
    int cls;
    char label[64];
};

// ---------------------------------------------------------------------
// Globals shared across all workers (read-only after startup, so no
// locking needed for these)
// ---------------------------------------------------------------------

std::vector<std::string> classes;

float conf=.7;

static std::atomic<bool> running(true);

void load_classes(const char* file){

    std::ifstream f(file);

    std::string s;

    while(std::getline(f,s)){

        if(!s.empty())
            classes.push_back(s);
    }
}

// Reads the model file into a freshly malloc'd buffer. Caller frees it
// (and only needs to keep it alive for the *first* rknn_init call --
// duplicated contexts via rknn_dup_context do not need the raw bytes).
unsigned char* read_model_file(const char* p, int& size){

    FILE* fp=fopen(p,"rb");

    if(!fp){
        std::cout<<"model missing\n";
        exit(-1);
    }

    fseek(fp,0,SEEK_END);
    size=ftell(fp);
    rewind(fp);

    auto* model=(unsigned char*)malloc(size);

    fread(model,1,size,fp);

    fclose(fp);

    return model;
}

// ---------------------------------------------------------------------
// Preprocessing (unchanged letterbox logic from the original file)
// ---------------------------------------------------------------------

struct PreprocResult{
    cv::Mat input;      // 640x640x3 uint8, letterboxed
    int orig_w, orig_h;
    float scale;
    int left, top;
};

PreprocResult preprocess(const cv::Mat& frame){

    PreprocResult r;

    r.orig_w=frame.cols;
    r.orig_h=frame.rows;

    r.scale=std::min(640.f/r.orig_w,640.f/r.orig_h);

    int nw=int(r.orig_w*r.scale);
    int nh=int(r.orig_h*r.scale);

    cv::Mat resized;

    cv::resize(frame,resized,cv::Size(nw,nh));

    r.input=cv::Mat(640,640,CV_8UC3,cv::Scalar(114,114,114));

    r.left=(640-nw)/2;
    r.top=(640-nh)/2;

    resized.copyTo(r.input(cv::Rect(r.left,r.top,nw,nh)));

    return r;
}

// ---------------------------------------------------------------------
// Postprocessing (unchanged YOLO decode + NMS logic from the original
// file). Operates on the raw float* output buffer from rknn_outputs_get.
// ---------------------------------------------------------------------

std::vector<Detection> postprocess(float* pred,int orig_w,int orig_h,float scale,int left,int top){

    std::vector<Detection> local;

    for(int i=0;i<8400;i++){

        float x=pred[0*8400+i];
        float y=pred[1*8400+i];
        float w=pred[2*8400+i];
        float h=pred[3*8400+i];

        float best=0;
        int cls=-1;

        for(int c=0;c<(int)classes.size();c++){

            float s=pred[(c+4)*8400+i];

            if(s>best){
                best=s;
                cls=c;
            }
        }

        if(best<conf)
            continue;

        x=(x-left)/scale;
        y=(y-top)/scale;
        w/=scale;
        h/=scale;

        Detection d;

        d.x1=std::max(0,int(x-w/2));
        d.y1=std::max(0,int(y-h/2));
        d.x2=std::min(orig_w-1,int(x+w/2));
        d.y2=std::min(orig_h-1,int(y+h/2));

        if(d.x2-d.x1<15 || d.y2-d.y1<15)
            continue;

        d.score=best;
        d.cls=cls;

        memset(d.label,0,sizeof(d.label));
        strncpy(d.label,classes[cls].c_str(),63);

        local.push_back(d);
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    for(auto& d:local){

        boxes.push_back(cv::Rect(d.x1,d.y1,d.x2-d.x1,d.y2-d.y1));
        scores.push_back(d.score);
    }

    std::vector<int> idx;

    cv::dnn::NMSBoxes(boxes,scores,conf,.45,idx);

    std::vector<Detection> final_det;

    for(int i:idx)
        final_det.push_back(local[i]);

    return final_det;
}

// ---------------------------------------------------------------------
// Worker: owns one rknn_context pinned to one NPU core, one mutex
// guarding that context, its own job queue, and its own dedicated
// inference thread.
// ---------------------------------------------------------------------

struct InferenceJob{
    cv::Mat input;
    int orig_w, orig_h;
    float scale;
    int left, top;
    std::promise<std::vector<Detection>> result;
};

struct Worker{
    rknn_context ctx=0;
    std::mutex ctx_mutex;

    std::queue<std::shared_ptr<InferenceJob>> jobs;
    std::mutex queue_mutex;
    std::condition_variable cv;

    std::thread thread;
    int core_id=0;
};

std::vector<std::unique_ptr<Worker>> workers;

void worker_loop(Worker* w){

    while(true){

        std::shared_ptr<InferenceJob> job;

        {
            std::unique_lock<std::mutex> lk(w->queue_mutex);

            w->cv.wait(lk,[w]{ return !w->jobs.empty() || !running; });

            if(!running && w->jobs.empty())
                return;

            job=w->jobs.front();
            w->jobs.pop();
        }

        std::vector<Detection> result;

        {
            std::lock_guard<std::mutex> lock(w->ctx_mutex);

            rknn_input inputs[1];
            memset(inputs,0,sizeof(inputs));

            inputs[0].index=0;
            inputs[0].type=RKNN_TENSOR_UINT8;
            inputs[0].fmt=RKNN_TENSOR_NHWC;
            inputs[0].size=640*640*3;
            inputs[0].buf=job->input.data;

            rknn_output outputs[1];
            memset(outputs,0,sizeof(outputs));

            outputs[0].want_float=1;

            int rc;

            rc=rknn_inputs_set(w->ctx,1,inputs);

            if(rc<0){
                std::cout<<"[worker "<<w->core_id<<"] rknn_inputs_set failed rc="<<rc<<"\n";
            } else {

                rc=rknn_run(w->ctx,nullptr);

                if(rc<0){
                    std::cout<<"[worker "<<w->core_id<<"] rknn_run failed rc="<<rc<<"\n";
                } else {

                    rc=rknn_outputs_get(w->ctx,1,outputs,nullptr);

                    if(rc<0){
                        std::cout<<"[worker "<<w->core_id<<"] rknn_outputs_get failed rc="<<rc<<"\n";
                    } else {

                        float* pred=(float*)outputs[0].buf;

                        result=postprocess(pred,job->orig_w,job->orig_h,job->scale,job->left,job->top);

                        rknn_outputs_release(w->ctx,1,outputs);
                    }
                }
            }
        }

        job->result.set_value(std::move(result));
    }
}

// Maps a worker index to a physical NPU core mask. RK3588 has 3 NPU
// cores; if more than 3 workers are requested they wrap around and
// share a core (still better than 1 shared context, but diminishing
// returns past 3 workers on this SoC).
rknn_core_mask core_mask_for(int worker_idx){

    switch(worker_idx%3){
        case 0: return RKNN_NPU_CORE_0;
        case 1: return RKNN_NPU_CORE_1;
        default: return RKNN_NPU_CORE_2;
    }
}

void init_workers(const char* model_path,int num_workers){

    int size=0;

    unsigned char* model=read_model_file(model_path,size);

    rknn_context master=0;

    if(rknn_init(&master,model,size,0,nullptr)<0){
        std::cout<<"rknn init fail\n";
        exit(-1);
    }

    // The raw model buffer is only needed for the initial load; dup'd
    // contexts share the already-loaded weights internally.
    free(model);

    for(int i=0;i<num_workers;i++){

        auto w=std::make_unique<Worker>();

        w->core_id=i%3;

        if(i==0){
            w->ctx=master;
        } else {
            if(rknn_dup_context(&master,&w->ctx)<0){
                std::cout<<"rknn_dup_context fail for worker "<<i<<"\n";
                exit(-1);
            }
        }

        if(rknn_set_core_mask(w->ctx,core_mask_for(i))<0){
            std::cout<<"rknn_set_core_mask fail for worker "<<i<<"\n";
            exit(-1);
        }

        workers.push_back(std::move(w));
    }

    for(auto& w:workers){
        Worker* raw=w.get();
        raw->thread=std::thread(worker_loop,raw);
    }

    std::cout<<num_workers<<" workers ready across NPU cores\n";
}

// ---------------------------------------------------------------------
// Client handling: unchanged wire protocol. Each client thread does
// socket I/O + JPEG decode + letterbox preprocessing (CPU work), then
// hands off to its assigned worker's queue and blocks on the result.
// ---------------------------------------------------------------------

void handle(int client,int worker_idx){

    Worker* w=workers[worker_idx].get();

    while(true){

        int sz;

        int r=recv(client,&sz,sizeof(int),MSG_WAITALL);

        if(r<=0)
            break;

        if(sz<=0 || sz>20*1024*1024)
            continue;

        std::vector<unsigned char> data(sz);

        int received=0;

        while(received<sz){

            r=recv(client,data.data()+received,sz-received,MSG_WAITALL);

            if(r<=0)
                break;

            received+=r;
        }

        if(received!=sz)
            continue;

        cv::Mat frame=cv::imdecode(data,1);

        if(frame.empty())
            continue;

        PreprocResult pp=preprocess(frame);

        auto job=std::make_shared<InferenceJob>();

        job->input=pp.input;
        job->orig_w=pp.orig_w;
        job->orig_h=pp.orig_h;
        job->scale=pp.scale;
        job->left=pp.left;
        job->top=pp.top;

        std::future<std::vector<Detection>> fut=job->result.get_future();

        {
            std::lock_guard<std::mutex> lk(w->queue_mutex);
            w->jobs.push(job);
        }

        w->cv.notify_one();

        std::vector<Detection> final_det=fut.get();

        int n=final_det.size();

        send(client,&n,sizeof(int),0);

        if(n){
            send(client,final_det.data(),n*sizeof(Detection),0);
        }
    }

    close(client);
}

int main(int argc,char** argv){

    if(argc<4){

        std::cout
        <<"Usage:\n"
        <<"./rknn_server model.rknn classes.txt .7 [num_workers]\n";

        return -1;
    }

    load_classes(argv[2]);

    conf=atof(argv[3]);

    int num_workers=3;

    if(argc>=5)
        num_workers=std::max(1,atoi(argv[4]));

    // Prevent OpenCV from spawning its own internal thread pool for
    // imdecode/resize/NMSBoxes -- we already have explicit per-client
    // and per-worker threads, and letting OpenCV parallelize on top of
    // that oversubscribes the CPU cores as camera count grows.
    cv::setNumThreads(1);

    init_workers(argv[1],num_workers);

    int server=socket(AF_INET,SOCK_STREAM,0);

    int opt=1;

    setsockopt(server,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt));

    sockaddr_in addr;

    addr.sin_family=AF_INET;
    addr.sin_port=htons(9000);
    addr.sin_addr.s_addr=INADDR_ANY;

    bind(server,(sockaddr*)&addr,sizeof(addr));

    listen(server,20);

    std::cout<<"server ready\n";

    std::atomic<int> client_counter(0);

    while(true){

        int client=accept(server,0,0);

        int worker_idx=client_counter.fetch_add(1)%num_workers;

        std::cout<<"client connected -> worker "<<worker_idx
                  <<" (NPU core "<<workers[worker_idx]->core_id<<")\n";

        std::thread(handle,client,worker_idx).detach();
    }
}
