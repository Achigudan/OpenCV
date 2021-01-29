#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::face;
using namespace std;

string haar_face_datapath = "F:/opencv/download/opencv/build/etc/haarcascades/haarcascade_frontalface_alt2.xml";
int main(int argc, char** argv) {
	string filename = string("E:/41.opencv_pro/Pro/face_recog_basic/FaceRecognition/image.csv");
	ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		printf("could not load file correctly...\n");
		return -1;
	}

	string line, path, classlabel;
	vector<Mat> images;
	vector<int> labels;
	char separator = ';';
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			printf("path : %s\n", path.c_str());
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}

	if (images.size() < 1 || labels.size() < 1) {
		printf("invalid image path...\n");
		return -1;
	}

	int height = images[0].rows;
	int width = images[0].cols;
	printf("height : %d, width : %d\n", height, width);

	Mat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];
	images.pop_back();
	labels.pop_back();

	// train it
	Ptr<BasicFaceRecognizer> model = EigenFaceRecognizer::create();
	model->train(images, labels);
	

	// recognition face
	int predictedLabel = model->predict(testSample);
	printf("actual label : %d, predict label :  %d\n", testLabel, predictedLabel);
	model->write("E:/41.opencv_pro/Pro/face_recog_basic/FaceRecognition/face_recog.xml");

	//加载一个人脸识别器
	Ptr<BasicFaceRecognizer> model_test = EigenFaceRecognizer::create();
	//opencv3.3要用read，要不然会出错
	model_test->read("E:/41.opencv_pro/Pro/face_recog_basic/FaceRecognition/face_recog.xml");
	CascadeClassifier faceDetector;
	faceDetector.load(haar_face_datapath);

	VideoCapture capture(0);
	if (!capture.isOpened()) {
		printf("could not open camera...\n");
		return -1;
	}

	Mat frame;
	string name;
	namedWindow("face-recognition", WINDOW_AUTOSIZE);
	vector<Rect> faces;
	Mat dst;
	while (capture.read(frame)) {
		flip(frame, frame, 1);
		faceDetector.detectMultiScale(frame, faces, 1.1, 1, 0, Size(80, 100), Size(380, 400));
		for (int i = 0; i < faces.size(); i++) {
			Mat roi = frame(faces[i]);
			cvtColor(roi, dst, COLOR_BGR2GRAY);
			resize(dst, testSample, Size(92,112));
			int label = model_test->predict(testSample);
			cout << label << endl;
			rectangle(frame, faces[i], Scalar(255, 0, 0), 2, 8, 0);
			switch (label)
			{
			case 41:
				name = "I am csl";
				break;
			case 42:
				name = "Ashida Mana";
				break;
			default:
				name = "Unknown";
				break;
			}
			putText(frame, name, faces[i].tl(), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 2, 8);
			//putText(frame, format("i'm %s", (label == 19 ? "csl" : "Unknow")), faces[i].tl(), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 2, 8);
		}


		imshow("face-recognition", frame);
		char c = waitKey(15);
		if (c == 27) {
			break;
		}
	}

	waitKey(0);
	return 0;
}