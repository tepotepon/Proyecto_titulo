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

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

int main(int argc, char** argv)
{
    //Global variables
    string descriptorType = defaultDescriptorType;
    string matcherType = defaultMatcherType;
    string fileWithTrainImages = defaultFileWithTrainImages;
    string dirToSaveResImages = defaultDirToSaveResImages;

    //feature variables
    Ptr<Feature2D> featureDetector;
    Ptr<DescriptorMatcher> descriptorMatcher;

    //Images Variables
    vector<Mat> trainImages;
    vector<string> trainImagesNames;

    //KeyPoing Variables
    vector<vector<KeyPoint> > trainKeypoints;

    //Descriptor Variables
    vector<Mat> trainDescriptors;

    //Matches Variables.
    //vector<DMatch> matches;
    vector<vector<DMatch> > Mega_matches;

    //views
    vector<view> views;

    //default camera matrix
    Mat K = cv::Mat::zeros(3,3,CV_64FC1);
    K.at<double>(0,0) = 3.7930862128394227e+02;
    K.at<double>(1,1) = 3.7930862128394227e+02;
    K.at<double>(0,2) = 320.;
    K.at<double>(1,2) = 240.;
    K.at<double>(2,2) = 1;

    //flags
    bool flag = true;

    //flag = (argv[1] == "true")? true:false;

    /*if(argc != 7 && argc != 1)
    {
        help();
    }
    */

    if (flag == false){
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
        read_from_files(trainKeypoints,views,Mega_matches);
    }

    bool first_two_flag = true;
    vector<Point2f>imgpts1,imgpts2;

    if (first_two_flag == true){
        vector<KeyPoint> kpts1, kpts2;
        kpts1 = trainKeypoints[0];
        kpts2 = trainKeypoints[1];
        vector<DMatch> maches = Mega_matches[0];
        for( unsigned int i = 0; i<maches.size(); i++ ){
            // queryIdx is the "left" image
            imgpts1.push_back(kpts1[maches[i].queryIdx].pt);
            // trainIdx is the "right" image
            imgpts2.push_back(kpts2[maches[i].trainIdx].pt);
        }
        cout << "done with de imgpts..." << endl;
    }

    return 0;
}
