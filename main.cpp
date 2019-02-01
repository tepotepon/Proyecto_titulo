#include <iostream>
#include <fstream>

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
    string detectorType = defaultDetectorType;
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
    vector<DMatch> matches;
    vector<KeyPoint> matched1, matched2;
    vector<vector<DMatch> > Mega_matches_aux;

    //views
    vector<view> views;


    /*if(argc != 7 && argc != 1)
    {
        help();
    }
    */

    if(!createDetectorDescriptorMatcher(detectorType, descriptorType, matcherType, featureDetector, descriptorMatcher))
    {
        //printPrompt( argv[0] );
        return -1;
    }

    //Mathing Process:
    if(!readImages(fileWithTrainImages,trainImages,trainImagesNames,views))
    {
        //printPrompt(argv[0]);
        return -1;
    }

    detectKeypoints(trainImages, trainKeypoints, featureDetector);

    computeDescriptors(trainImages, trainKeypoints, trainDescriptors,
                       featureDetector);

    matchDescriptors(Mega_matches_aux, trainDescriptors,trainKeypoints, matches, descriptorMatcher,matched1,matched2);

    saveResultImages(Mega_matches_aux, trainImages, trainKeypoints,
                     trainImagesNames, dirToSaveResImages);


    return 0;
}
