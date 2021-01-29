#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

string haar_face_datapath = "F:/opencv/download/opencv/build/etc/haarcascades/haarcascade_frontalface_alt2.xml";
string lbp_face_datapath = "F:/opencv/download/opencv/build/etc/lbpcascades/lbpcascade_frontalface.xml";

int main(int argc, char** argv) {
	VideoCapture capture(0); // open camera
	if (!capture.isOpened()) {
		printf("could not open camera...\n");
		return -1;
	}

	Size S = Size((int)capture.get(CAP_PROP_FRAME_WIDTH), (int)capture.get(CAP_PROP_FRAME_HEIGHT));
	int fps = capture.get(CAP_PROP_FPS);

	CascadeClassifier faceDetector;
	//级联分类器加载
	faceDetector.load(haar_face_datapath);

	Mat frame;
	namedWindow("camera-demo", WINDOW_AUTOSIZE);
	//vector存储检测到的人脸矩形
	vector<Rect> faces;
	int count = 0;
	while (capture.read(frame)) {
		
		flip(frame, frame, 1);
		/*
		参数：1、frame:待检测图片，一般为灰度图像加快检测速度
			  2、faces:被检测物体的矩形框向量组
			  3、scaleFactor:表示在前后两次相继的扫描中，搜索窗口的比例系数，默认为1.1即每次搜索窗口依次扩大10%;
			  4、minNeighbors:表示构成检测目标的相邻矩形的最小个数（默认为3）
			    如果组成检测目标的小矩形的个数和小于 min_neighbors - 1 都会被排除。
				如果min_neighbors 为 0, 则函数不做任何操作就返回所有的被检候选矩形框，
				这种设定值一般用在用户自定义对检测结果的组合程序上
			  5、flags--要么使用默认值，要么使用CV_HAAR_DO_CANNY_PRUNING，如果设置为
				 CV_HAAR_DO_CANNY_PRUNING，那么函数将会使用Canny边缘检测来排除边缘过多或过少的区域，
				 因此这些区域通常不会是人脸所在区域；
			  6、7:minSize和maxSize用来限制得到的目标区域的范围。
		*/
		faceDetector.detectMultiScale(frame, faces, 1.1, 3, 0, Size(100, 120), Size(380, 400));
		for (int i = 0; i < faces.size(); i++) {
			if (count % 10 == 0) {
				Mat dst;
				resize(frame(faces[i]), dst, Size(92, 112));
				cvtColor(dst, dst, COLOR_BGR2GRAY);
				imwrite(format("E:/41.opencv_pro/Pro/face_recog_basic/FaceDetected/outimage/face_%d.jpg", count), dst);
			}
			rectangle(frame, faces[i], Scalar(0, 0, 255), 2, 8, 0);
		}
		imshow("camera-demo", frame);
		char c = waitKey(50);
		if (c == 27) {
			break;
		}
		count++;
	}

	capture.release();

	waitKey(0);
	return 0;
}