#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stats.h>
#include <utils.h>
#include <view.h>
#include <many_images_matching.h>
#include <sfm.h>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

int main() //int argc, char** argv
{
    //Global variables
    string descriptorType = defaultDescriptorType;
    string matcherType = defaultMatcherType;
    string fileWithTrainImages = defaultFileWithTrainImages;
    //string dirToSaveResImages = defaultDirToSaveResImages;

    //feature variables
    Ptr<Feature2D> featureDetector;
    Ptr<DescriptorMatcher> descriptorMatcher;

    //Images Variables
    vector<Mat> trainImages;
    vector<string> trainImagesNames;
    vector<Point2f>imgpts1,imgpts2;

    //KeyPoing Variables
    vector<vector<KeyPoint> > trainKeypoints;

    //Descriptor Variables
    vector<Mat> trainDescriptors;

    //Matches Variables.
    //vector<DMatch> matches;
    vector<vector<DMatch> > Mega_matches;

    //views
    vector<view> views;

    //Point cloud
    vector<Point3d> pointcloud;

    //default Calibrated camera matrix
    Mat K = cv::Mat::zeros(3,3,CV_64FC1);
    K.at<double>(0,0) = 3.7930862128394227e+02;
    K.at<double>(1,1) = 3.7930862128394227e+02;
    K.at<double>(0,2) = 320.;
    K.at<double>(1,2) = 240.;
    K.at<double>(2,2) = 1;

    //camera matrices
    Matx34d P1;
    Matx34d P( 1,0, 0, 0,
    0,1, 0, 0,
    0,0, 1, 0);

    //flags
    bool work_from_saved_flag;

    string Chosen_Type;
    cout << "Starting SfM program" << endl;
    cout << "WORK FROM SAVED FILES?: (Y/N) ";
    cin >> Chosen_Type;

    work_from_saved_flag = (Chosen_Type == "Y")? true:false;

    /*if(argc != 7 && argc != 1)
    {
        help();
    }
    */

    if (work_from_saved_flag == false){
        cout << "no saved values..." << endl
             << "starting Matching process" << endl;
        //create detector
        if(!createDetectorDescriptorMatcher(descriptorType, matcherType, featureDetector, descriptorMatcher))
        {
            //printPrompt( argv[0] );
            return -1;
        }

        //Read images:
        if(!readImages(fileWithTrainImages,trainImages,trainImagesNames,views))
        {
            //printPrompt(argv[0]);
            return -1;
        }

        //detect keypoints
        detectKeypoints(trainImages, trainKeypoints, featureDetector,views);

        computeDescriptors(trainImages, trainKeypoints, trainDescriptors,featureDetector);

        matchDescriptors(Mega_matches, trainDescriptors,descriptorMatcher);

        //saveResultImages(Mega_matches, trainImages, trainKeypoints,trainImagesNames, dirToSaveResImages);
    }

    else {
        //reads from saved files:
        read_from_files(trainKeypoints,views,Mega_matches);
    }

    //continuew with sfm.

    //First two baseline.
    int par = 0;
    P1 = Find_camera_matrix(K,trainKeypoints,imgpts1,imgpts2,Mega_matches,par);

    TriangulatePoints(imgpts1,imgpts2, K, P, P1, pointcloud);

    return 0;
}



