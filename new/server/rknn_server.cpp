#include <iostream>
#include <thread>
#include <vector>
#include <fstream>
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
};

rknn_context ctx=0;

std::vector<std::string> classes;

float conf=.7;

void load_classes(const char* f){

 std::ifstream file(f);

 std::string s;

 while(std::getline(file,s))
   classes.push_back(s);
}

void load_model(const char* p){

 FILE* fp=fopen(p,"rb");

 fseek(fp,0,SEEK_END);

 int size=ftell(fp);

 rewind(fp);

 unsigned char* model=
 (unsigned char*)malloc(size);

 fread(model,1,size,fp);

 fclose(fp);

 rknn_init(
 &ctx,
 model,
 size,
 0,
 nullptr
 );

 free(model);
}

void handle(int client){

 while(true){

   int sz;

   if(recv(
      client,
      &sz,
      sizeof(int),
      MSG_WAITALL
   )<=0)
      break;

   std::vector<unsigned char> data(sz);

   recv(
      client,
      data.data(),
      sz,
      MSG_WAITALL
   );

   cv::Mat img=
   cv::imdecode(
      data,
      1
   );

   cv::resize(
      img,
      img,
      cv::Size(
         640,
         640
      )
   );

   rknn_input input[1]={0};

   input[0].index=0;

   input[0].type=
   RKNN_TENSOR_UINT8;

   input[0].fmt=
   RKNN_TENSOR_NHWC;

   input[0].size=
   640*640*3;

   input[0].buf=
   img.data;

   rknn_inputs_set(
      ctx,
      1,
      input
   );

   rknn_run(
      ctx,
      nullptr
   );

   rknn_output out[1]={0};

   out[0].want_float=1;

   rknn_outputs_get(
      ctx,
      1,
      out,
      nullptr
   );

   float* pred=
   (float*)
   out[0].buf;

   std::vector<Detection> det;

   for(int i=0;i<8400;i++){

      float best=0;

      int cls=-1;

      for(int c=0;c<classes.size();c++){

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

      if(best<conf)
         continue;

      Detection d;

      d.x1=
      pred[0*8400+i];

      d.y1=
      pred[1*8400+i];

      d.x2=
      pred[2*8400+i];

      d.y2=
      pred[3*8400+i];

      d.score=
      best;

      d.cls=
      cls;

      det.push_back(d);
   }

   int n=
   det.size();

   send(
      client,
      &n,
      sizeof(int),
      0
   );

   send(
      client,
      det.data(),
      n*sizeof(Detection),
      0
   );

   rknn_outputs_release(
      ctx,
      1,
      out
   );
 }

 close(client);
}

int main(
int argc,
char** argv
){

load_model(argv[1]);

load_classes(argv[2]);

conf=
atof(argv[3]);

int s=
socket(
AF_INET,
SOCK_STREAM,
0
);

sockaddr_in addr;

addr.sin_family=
AF_INET;

addr.sin_port=
htons(9000);

addr.sin_addr.s_addr=
INADDR_ANY;

bind(
s,
(sockaddr*)&addr,
sizeof(addr)
);

listen(
s,
10
);

while(true){

int client=
accept(
s,
0,
0
);

std::thread(
handle,
client
).detach();
}
}