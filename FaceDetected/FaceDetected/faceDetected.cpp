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
	//��������������
	faceDetector.load(haar_face_datapath);

	Mat frame;
	namedWindow("camera-demo", WINDOW_AUTOSIZE);
	//vector�洢��⵽����������
	vector<Rect> faces;
	int count = 0;
	while (capture.read(frame)) {
		
		flip(frame, frame, 1);
		/*
		������1��frame:�����ͼƬ��һ��Ϊ�Ҷ�ͼ��ӿ����ٶ�
			  2��faces:���������ľ��ο�������
			  3��scaleFactor:��ʾ��ǰ��������̵�ɨ���У��������ڵı���ϵ����Ĭ��Ϊ1.1��ÿ������������������10%;
			  4��minNeighbors:��ʾ���ɼ��Ŀ������ھ��ε���С������Ĭ��Ϊ3��
			    �����ɼ��Ŀ���С���εĸ�����С�� min_neighbors - 1 ���ᱻ�ų���
				���min_neighbors Ϊ 0, ���������κβ����ͷ������еı����ѡ���ο�
				�����趨ֵһ�������û��Զ���Լ��������ϳ�����
			  5��flags--Ҫôʹ��Ĭ��ֵ��Ҫôʹ��CV_HAAR_DO_CANNY_PRUNING���������Ϊ
				 CV_HAAR_DO_CANNY_PRUNING����ô��������ʹ��Canny��Ե������ų���Ե�������ٵ�����
				 �����Щ����ͨ��������������������
			  6��7:minSize��maxSize�������Ƶõ���Ŀ������ķ�Χ��
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