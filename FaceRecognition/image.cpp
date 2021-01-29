#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>


using namespace std;

int main(){

    ofstream outfile;
    outfile.open("image.csv",ios::out);

    // for(int i = 0;i < 10; i++){
    //     outfile <<"E:/41.opencv_pro/Pro/face_recog_basic/FaceDetected/outimage/face_"<<i+1<<"0.jpg"<<";"<<19<<endl;

    // }
for(int i = 0; i< 41;i++){
    for(int j = 0;j < 10; j++){
        outfile <<"F:/opencv/others/orl_faces/s"<<i+1<<"/"<<j+1<<".pgm"<<";"<<i+1<<endl;

    }
}

    outfile.close();
    return 0;
}