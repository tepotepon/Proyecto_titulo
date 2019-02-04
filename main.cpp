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
    //vector<KeyPoint> matched1, matched2;
    vector<vector<DMatch> > Mega_matches_aux;

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
        detectKeypoints(trainImages, trainKeypoints, featureDetector);

        //store keypoints on view class
        for (uint i = 0; i<trainKeypoints.size(); i++){
            views[i] = trainKeypoints[i];
        }

        //save keypoints on file
        cout << "Saving keypoints no file..." << endl;
        string filename = "keypoints";
        FileStorage fs(filename, FileStorage::WRITE);
        for (uint i = 0; i<trainKeypoints.size(); i++){
            stringstream ss;
            ss << i;
            fs << "keypoints " + ss.str();
            fs << trainKeypoints[i];
        }

        // explicit close
        fs.release();

        cout << "Write Done." << endl;

        computeDescriptors(trainImages, trainKeypoints, trainDescriptors,
                           featureDetector);

        matchDescriptors(Mega_matches_aux, trainDescriptors,descriptorMatcher);

        //save machets into file
        string filename2 = "matches";
        FileStorage fs2(filename2, FileStorage::WRITE);

        for (uint i = 0; i<Mega_matches_aux.size(); i++){
            fs2 << "Matches" ;
            fs2 << Mega_matches_aux[i];
        }

        // explicit close
        fs2.release();
        cout << "Write Done." << endl;




        /*
            // -- Align matches into arrays (same indexes)
            cout << "trainKeypoints.size()" << trainKeypoints.size() << endl;
            for(int j=0; j < trainKeypoints.size(); j++)
            {
                cout << "j: " << j << endl;
                train_points = trainKeypoints[j];
                for(int i = 0; i < good_matches.size(); i++)
                {
                    matched1.push_back(queryKeypoints[good_matches[i].queryIdx]);
                    matched2.push_back(train_points[good_matches[i].trainIdx]);
                    cout << "i :" << i << endl;
                }
            }*/



        saveResultImages(Mega_matches_aux, trainImages, trainKeypoints,
                         trainImagesNames, dirToSaveResImages);
    }

    else {
        cout << "extracting values from files..." << endl;
        cout << endl << "Reading... " << endl;
        FileStorage fs3;
        string filename3 = "keypoints";
        fs3.open(filename3, FileStorage::READ);

        if (!fs3.isOpened())
        {
            cerr << "Failed to open " << filename3 << endl;
            return 1;
        }

        int i = 0;

        while(true){
            vector<KeyPoint> Keypoints;
            stringstream ss;
            ss << i;
            fs3["keypoints " + ss.str()] >> Keypoints;
            if (Keypoints.size() != 0){
                view vista(Keypoints);
                views.push_back(vista);
                i++;
            }
            else {
                cout << "i :" << i-1 << endl;
                break;
            }
        }

         fs3.release();
         cout << "done reading" << endl;
         cout << "total views: " << views.size()<< endl;
    }

    return 0;
}
