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
	
	// ��ʾʹ��Ĭ�ϲ�������Stitcher�����stitcher
	// try_use_gpu��ʾ�Ƿ��GPU��Ĭ�ϲ��򿪣���try_use_gpu=false
	Stitcher stitcher = Stitcher::createDefault(try_use_gpu);

	//Ϊ�˼��٣���ѡ0.1,Ĭ����0.6,���ֵ1�������˷���������������׶Σ�����Ҳ��������㣬���߰�
	stitcher.setRegistrationResol(0.6);

	//Ĭ����0.1
	stitcher.setSeamEstimationResol(0.1);

	//Ĭ����-1��������������׶Σ��Ҳ���������Ļ�����-1
	stitcher.setCompositingResol(-1);

	//Ĭ����1,��������0.6��0.4��
	stitcher.setPanoConfidenceThresh(1);

	//Ĭ����true��Ϊ����ѡfalse����ʾ����WaveCorrection����
	stitcher.setWaveCorrection(true);

	//������ѡdetail::WAVE_CORRECT_VERT ,��������(wave correction)����
	//��ˮƽ����/��ֱ��������������ΪsetWaveCorrection���false�������û��
	stitcher.setWaveCorrectKind(detail::WAVE_CORRECT_HORIZ);


	stitcher.setFeaturesMatcher(new detail::BestOf2NearestMatcher(try_use_gpu));
	stitcher.setBundleAdjuster(new detail::BundleAdjusterRay());


	stitcher.setFeaturesFinder(new detail::SurfFeaturesFinder());
	stitcher.setWarper(new SphericalWarper());
	stitcher.setSeamFinder(new detail::GraphCutSeamFinder(detail::GraphCutSeamFinderBase::COST_COLOR));

	//Ĭ�ϵľ��ǲ�Ҫ�عⲹ��
	stitcher.setExposureCompensator(new detail::BlocksGainCompensator());
	stitcher.setBlender(new detail::MultiBandBlender(try_use_gpu));


	Stitcher::Status status = stitcher.stitch(imgs, pano);

	if (status != Stitcher::OK)
	{
		cout << "Can't stitch images, error code = " << int(status) << endl;
		return -1;
	}

	imwrite(result_name, pano);

	imshow("ȫ��ͼ��", pano);
	waitKey(0);

	return 0;
}


