#include <iostream>
#include <thread>
#include <vector>
#include <fstream>
#include <cstring>

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

rknn_context ctx=0;

std::vector<std::string> classes;

float conf=.7;

void load_classes(const char* file){

    std::ifstream f(file);

    std::string s;

    while(std::getline(f,s)){

        if(!s.empty())
            classes.push_back(s);
    }
}

void load_model(const char* p){

    FILE* fp=fopen(p,"rb");

    if(!fp){

        std::cout<<"model missing\n";
        exit(-1);
    }

    fseek(fp,0,SEEK_END);

    int size=ftell(fp);

    rewind(fp);

    auto* model=
    (unsigned char*)
    malloc(size);

    fread(
        model,
        1,
        size,
        fp
    );

    fclose(fp);

    if(
        rknn_init(
        &ctx,
        model,
        size,
        0,
        nullptr
        )<0
    ){

        std::cout
        <<"rknn init failed\n";

        exit(-1);
    }

    free(model);

    std::cout
    <<"model loaded\n";
}

void handle(int client){

while(true){

    int sz;

    int r=
    recv(
        client,
        &sz,
        sizeof(int),
        MSG_WAITALL
    );

    if(r<=0)
        break;

    std::vector<unsigned char>
    data(sz);

    recv(
        client,
        data.data(),
        sz,
        MSG_WAITALL
    );

    cv::Mat frame=
    cv::imdecode(
        data,
        1
    );

    if(frame.empty())
        continue;

    //--------------------------------
    // exact python preprocessing
    //--------------------------------

    int orig_w=
    frame.cols;

    int orig_h=
    frame.rows;

    float scale=
    std::min(
        640.f/orig_w,
        640.f/orig_h
    );

    int nw=
    int(orig_w*scale);

    int nh=
    int(orig_h*scale);

    cv::Mat resized;

    cv::resize(
        frame,
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

    //--------------------------------
    // RKNN
    //--------------------------------

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
        ctx,
        1,
        inputs
        )<0
    ) continue;

    if(
        rknn_run(
        ctx,
        nullptr
        )<0
    ) continue;

    rknn_output outputs[1];

    memset(
        outputs,
        0,
        sizeof(outputs)
    );

    outputs[0]
    .want_float=1;

    if(
        rknn_outputs_get(
        ctx,
        1,
        outputs,
        nullptr
        )<0
    ) continue;

    float* pred=
    (float*)
    outputs[0].buf;

    std::vector<Detection>
    local;

    for(
    int i=0;
    i<8400;
    i++
    ){

        float x=
        pred[0*8400+i];

        float y=
        pred[1*8400+i];

        float w=
        pred[2*8400+i];

        float h=
        pred[3*8400+i];

        float best=0;

        int cls=-1;

        for(
        int c=0;
        c<classes.size();
        c++
        ){

            float s=
            pred[
            (c+4)
            *8400+i
            ];

            if(s>best){

                best=s;

                cls=c;
            }
        }

        if(
        best<conf
        )
        continue;

        //--------------------------------
        // undo letterbox
        //--------------------------------

        x=
        (x-left)
        /scale;

        y=
        (y-top)
        /scale;

        w/=
        scale;

        h/=
        scale;

        Detection d;

        d.x1=
        std::max(
        0,
        int(
        x-w/2
        ));

        d.y1=
        std::max(
        0,
        int(
        y-h/2
        ));

        d.x2=
        std::min(
        orig_w-1,
        int(
        x+w/2
        ));

        d.y2=
        std::min(
        orig_h-1,
        int(
        y+h/2
        ));

        if(
        d.x2-d.x1<15
        ||
        d.y2-d.y1<15
        )
        continue;

        d.score=
        best;

        d.cls=
        cls;

        memset(
        d.label,
        0,
        sizeof(d.label)
        );

        strncpy(
        d.label,
        classes[cls].c_str(),
        63
        );

        local.push_back(d);
    }

    //--------------------------------
    // NMS
    //--------------------------------

    std::vector<cv::Rect>
    boxes;

    std::vector<float>
    scores;

    for(auto& d:local){

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

    std::vector<int>
    idx;

    cv::dnn::NMSBoxes(

        boxes,

        scores,

        conf,

        .45,

        idx
    );

    std::vector<Detection>
    final_det;

    for(
    int i:idx
    ){

        final_det.push_back(
        local[i]
        );
    }

    int n=
    final_det.size();

    send(
        client,
        &n,
        sizeof(int),
        0
    );

    if(n>0){

    send(
        client,
        final_det.data(),
        n*sizeof(Detection),
        0
    );
    }

    rknn_outputs_release(
        ctx,
        1,
        outputs
    );
}

close(client);
}

int main(
int argc,
char** argv
){

if(argc<4){

std::cout
<<"Usage:\n"
<<"./rknn_server model.rknn classes.txt .7\n";

return -1;
}

load_model(
argv[1]
);

load_classes(
argv[2]
);

conf=
atof(
argv[3]
);

int server=
socket(
AF_INET,
SOCK_STREAM,
0
);

sockaddr_in addr;

addr.sin_family=
AF_INET;

addr.sin_port=
htons(
9000
);

addr.sin_addr.s_addr=
INADDR_ANY;

bind(
server,
(sockaddr*)&addr,
sizeof(addr)
);

listen(
server,
10
);

std::cout
<<"server ready\n";

while(true){

int client=
accept(
server,
0,
0
);

std::cout
<<"client connected\n";

std::thread(
handle,
client
).detach();
}
}