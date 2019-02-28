#include <iostream>
#include <fstream>
#include <string>

#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "stats.h"
#include "utils.h"
#include "view.h"
#include "many_images_matching.h"
#include "sfm.h"
#include "cloudpoint.h"
#include "cloud_to_pcldatastruct.h"


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
    vector<vector<DMatch> > Mega_matches;

    //views
    vector<view> views;

    //Point cloud
    typedef map<int,Matx34d> MyMap;
    MyMap Pmats;


    //global 3D point cloud
    vector<CloudPoint> pcloud;

    //default Calibrated camera matrix
    Mat K = Mat::zeros(3,3,CV_64FC1);
    K.at<double>(0,0) = 3.7930862128394227e+02;
    K.at<double>(1,1) = 3.7930862128394227e+02;
    K.at<double>(0,2) = 320.;
    K.at<double>(1,2) = 240.;
    K.at<double>(2,2) = 1;

    //default distortion_coefficients camera matrix
    Mat distcoeff = Mat::zeros(5,1,CV_64FC1);
    distcoeff.at<double>(0,0) = 3.1472442785745963e-01;
    distcoeff.at<double>(1,0) = 1.2068665737088996e-01;
    distcoeff.at<double>(4,0) = -2.3079325801519321e-02;

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
        cout << endl << "total views: " << views.size() << endl;
        cout << "Mega_matches size :" << Mega_matches.size() << endl;
    }

    //First two baseline.
    cout << endl << "< first two baseline: " << endl ;
    P1 = Find_camera_matrix(K,trainKeypoints,imgpts1,imgpts2,Mega_matches,1);
    TriangulatePoints(0,Mega_matches,imgpts1,imgpts2, K, P, P1, pcloud);
    Pmats.insert({0,P1});
    cout << ">" << endl;

    cout << P1 << endl;

    for (unsigned int i = 2 ; i < views.size()-1 ; i++){
        imgpts1.clear();
        imgpts2.clear();
        cout << endl << "pair of images: " << i << " - " << i-1 << endl;
        cout << "actual pcloud size: " << pcloud.size() << endl;
        P1 = P1_from_correspondence(pcloud,trainKeypoints,Mega_matches,K,distcoeff,i,i-1);
        Pmats.insert({int(i-1),P1});
        sort_imgpts(i,trainKeypoints,Mega_matches,imgpts1,imgpts2);
        TriangulatePoints(i-1,Mega_matches,imgpts1,imgpts2, K, P, P1, pcloud);
    }

    // prints P elements of map
    cout << endl << "KEY\tELEMENT\n";
    for (MyMap::iterator itr = Pmats.begin(); itr != Pmats.end(); ++itr) {
        cout << itr->first
             << '\t' << itr->second << '\n';
    }

    vector<Point3d> mccloud;
    return 0;
}



