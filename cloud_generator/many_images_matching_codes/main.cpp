#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <pcl/features/normal_3d.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

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

const string defaultDescriptorType = "SURF";
const string defaultMatcherType = "FlannBased";
const string defaultFileWithTrainImages = "../../train/trainImages.txt";
const string defaultDirPCD = "../../Clouds/"


int main() //int argc, char** argv
{
    //Global variables
    string descriptorType = defaultDescriptorType;
    string matcherType = defaultMatcherType;
    string fileWithTrainImages = defaultFileWithTrainImages;
    string PCDdir = defaultDirPCD;

    //feature variables
    Ptr<Feature2D> featureDetector;
    Ptr<DescriptorMatcher> descriptorMatcher;

    //Images Variables
    vector<Mat> trainImages;
    vector<string> trainImagesNames;
    vector<Point2f> imgpts1,imgpts2;
    vector<vector<Vec3b> > all_colors;
    vector<Vec3b> actual_colors;

    //KeyPoing Variables
    vector<vector<KeyPoint> > trainKeypoints;

    //Descriptor Variables
    vector<Mat> trainDescriptors;

    //Matches Variables.
    vector<vector<DMatch> > Mega_matches;

    //views
    vector<view> views;

    //Pmat camera matrices
    typedef map<int,Matx34d> MyMap;
    MyMap Pmats;

    //global 3D point cloud
    vector<CloudPoint> pcloud;

    //PCL 3D point cloud with color
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr final_cloud;

    //default Calibrated camera matrix
    Mat K = Mat::zeros(3,3,CV_64FC1);
    K.at<double>(0,0) = 361.01260941343548;
    K.at<double>(1,1) = 360.47481889255971;
    K.at<double>(0,2) = 320.;
    K.at<double>(1,2) = 240.;
    K.at<double>(2,2) = 1;

    //default distortion_coefficients camera matrix
    Mat distcoeff = Mat::zeros(5,1,CV_64FC1);
    distcoeff.at<double>(0,0) = -2.9421997795578470e-01;
    distcoeff.at<double>(1,0) = 9.8304132222971852e-02;
    distcoeff.at<double>(2,0) = -1.2893816224428640e-03;
    distcoeff.at<double>(3,0) = 6.6629639415434483e-04;
    distcoeff.at<double>(4,0) = -1.4792082836362129e-02;

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
        detectKeypoints(trainImages, trainKeypoints, featureDetector,views,all_colors);
        computeDescriptors(trainImages, trainKeypoints, trainDescriptors,featureDetector);
        matchDescriptors(trainKeypoints,Mega_matches, trainDescriptors,descriptorMatcher);

        cout << "wish to save matching results? (Y/N): ";
        string save_aux; 
        cin >> save_aux
        bool save_images_flag = (save_aux == "Y")? true:false;
        if (save_images_flag == true){
            string path = "../../matching_results/";
            string sa; 
            cout << "save as: "; 
            cin >> sa; 
            string dirToSaveResImages = path + sa; 
            saveResultImages(Mega_matches, trainImages, trainKeypoints,trainImagesNames, dirToSaveResImages);
        }
    }

    else {
        //reads from saved files:
        read_from_files(all_colors, trainKeypoints,Mega_matches);
        cout << endl << "total views: " << views.size() << endl;
        cout << "Mega_matches size :" << Mega_matches.size() << endl;
    }

    string NameToSavePCD; 
    string incremental_res;
    bool incremental_flag;
    cout << "enter pcd file name: "; 
    cin >> NameToSavePCD;
    cout << "wish to save incrementally? (Y/N): "; 
    cin << incremental_res; 
    incremental_flag = (incremental_res == "Y")? true:false;

    //First two baseline.
    Pmats.insert({0,P});
    cout << endl << "< first two baseline: " << endl ;
    sort_imgpts(trainKeypoints[0],trainKeypoints[1],Mega_matches[0],all_colors[0],imgpts1,imgpts2, actual_colors);
    P1 = Find_camera_matrix(K,imgpts1,imgpts2,P);
    TriangulatePoints(0,Mega_matches,imgpts1,imgpts2, K, P, P1, pcloud);
    Pmats.insert({1,P1});
    cout << ">" << endl;
    PopulatePCLPointCloud(pcloud,actual_colors,final_cloud);

    if(incremental_flag == true) {
        string pcd_path; 
        pcd_path = PCDdir + NameToSavePCD;
        pcl::io::savePCDFile(pcd_path,*final_cloud,false);
    }

    for (size_t i = 1 ; i < Mega_matches.size() ; i++){
        imgpts1.clear();
        imgpts2.clear();
        cout << "actual pcloud size: " << pcloud.size() << endl;
        cout << endl << "adding pair of images: " << i << " - " << i+1 << endl;
        P1 = P1_from_correspondence(pcloud,trainKeypoints,Mega_matches,K,distcoeff,i);
        Pmats.insert({int(i+1),P1});
        sort_imgpts(trainKeypoints[i],trainKeypoints[i+1],Mega_matches[i],all_colors[i],imgpts1,imgpts2, actual_colors);
        TriangulatePoints(i,Mega_matches,imgpts1,imgpts2, K, Pmats[i], P1, pcloud);
        PopulatePCLPointCloud(pcloud,actual_colors,final_cloud);
        // AGREGAR INCREMENTAL SAVE
        if (incremental_flag == true) {
            string next_save = pcd_path + string(i); 
            pcl::io::savePCDFile(next_save,*final_cloud,false);
        }
    }

    if(incremental_flag == false) {
        string pcd_path; 
        pcd_path = PCDdir + NameToSavePCD;
        pcl::io::savePCDFile(pcd_path,*final_cloud,false);
    }

    // prints P elements of map

    /*
    cout << endl << "KEY\tELEMENT\n";
    for (MyMap::iterator itr = Pmats.begin(); itr != Pmats.end(); ++itr) {
        cout << itr->first
             << '\t' << itr->second << '\n';
    }
    */

    cout << "listo" << endl;

    //SORFilter(cloud);

    return 0;
}
