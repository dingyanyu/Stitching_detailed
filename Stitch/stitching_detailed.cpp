#include <iostream>
#include <fstream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/stitcher.hpp"

using namespace std;
using namespace cv;

bool try_use_gpu = false;
vector<Mat> imgs;
string result_name = "result.jpg";


int main(int argc, char* argv[])
{
	Mat img1 = imread("boat1.jpg");
    Mat img2 = imread("boat2.jpg");

     if (img1.empty() || img2.empty())
	{
       cout << "Can't read image" << endl;
         return -1;
    }

	 imgs.push_back(img1);
	 imgs.push_back(img2);

	Mat pano;
	Stitcher stitcher = Stitcher::createDefault(try_use_gpu);
	Stitcher::Status status = stitcher.stitch(imgs, pano);

	if (status != Stitcher::OK)
	{
		cout << "Can't stitch images, error code = " << int(status) << endl;
		return -1;
	}

	imwrite(result_name, pano);

	imshow("È«¾°Í¼Ïñ", pano);
	waitKey(0);

	return 0;
}


