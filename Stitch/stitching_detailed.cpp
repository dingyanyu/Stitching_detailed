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
	
	// 表示使用默认参数创建Stitcher类对象stitcher
	// try_use_gpu表示是否打开GPU，默认不打开，即try_use_gpu=false
	Stitcher stitcher = Stitcher::createDefault(try_use_gpu);

	//为了加速，我选0.1,默认是0.6,最大值1最慢，此方法用于特征点检测阶段，如果找不到特征点，调高吧
	stitcher.setRegistrationResol(0.6);

	//默认是0.1
	stitcher.setSeamEstimationResol(0.1);

	//默认是-1，用于特征点检测阶段，找不到特征点的话，改-1
	stitcher.setCompositingResol(-1);

	//默认是1,见过有设0.6和0.4的
	stitcher.setPanoConfidenceThresh(1);

	//默认是true，为加速选false，表示跳过WaveCorrection步骤
	stitcher.setWaveCorrection(true);

	//还可以选detail::WAVE_CORRECT_VERT ,波段修正(wave correction)功能
	//（水平方向/垂直方向修正）。因为setWaveCorrection设的false，此语句没用
	stitcher.setWaveCorrectKind(detail::WAVE_CORRECT_HORIZ);


	stitcher.setFeaturesMatcher(new detail::BestOf2NearestMatcher(try_use_gpu));
	stitcher.setBundleAdjuster(new detail::BundleAdjusterRay());


	stitcher.setFeaturesFinder(new detail::SurfFeaturesFinder());
	stitcher.setWarper(new SphericalWarper());
	stitcher.setSeamFinder(new detail::GraphCutSeamFinder(detail::GraphCutSeamFinderBase::COST_COLOR));

	//默认的就是不要曝光补偿
	stitcher.setExposureCompensator(new detail::BlocksGainCompensator());
	stitcher.setBlender(new detail::MultiBandBlender(try_use_gpu));


	Stitcher::Status status = stitcher.stitch(imgs, pano);

	if (status != Stitcher::OK)
	{
		cout << "Can't stitch images, error code = " << int(status) << endl;
		return -1;
	}

	imwrite(result_name, pano);

	imshow("全景图像", pano);
	waitKey(0);

	return 0;
}


