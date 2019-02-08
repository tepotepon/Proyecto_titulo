#ifndef SFM
#define SFM

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <stats.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

bool CheckCoherentRotation(Mat_<double>& R);

Mat_<double> LinearLSTriangulation(
    Point3d u,//homogenous image point (u,v,1)
    Matx34d P,//camera 1 matrix
    Point3d u1,//homogenous image point in 2nd camera
    Matx34d P1//camera 2 matrix
    );

double TriangulatePoints(
    const vector<Point2f>& pt_set1,
    const vector<Point2f>& pt_set2,
    const Mat&K,
    const Matx34d& P,
    const Matx34d& P1,
    vector<Point3d>& pointcloud);

Matx34d Find_camera_matrix(
    Mat& K,
    const vector<vector<KeyPoint> >& trainKeypoints,
    vector<Point2f>& imgpts1,
    vector<Point2f>& imgpts2,
    vector<vector<DMatch> >& Mega_matches,
    int par);

bool CheckCoherentRotation(Mat_<double>& R){
    if(fabsf(determinant(R))-1.0 > 1e-07) {
        cerr<<"det(R) != +-1.0, this is not a rotation matrix" << endl;
        return false;
    }
    else {
        cout << "det(R) -> valid rotation" << endl;
        return true;
    }
}

Mat_<double> LinearLSTriangulation(
    Point3d u,//homogenous image point (u,v,1)
    Matx34d P,//camera 1 matrix
    Point3d u1,//homogenous image point in 2nd camera
    Matx34d P1//camera 2 matrix
    ){
    //build A matrix // esto sale de (7) triangulation .pdf
    Matx43d A(u.x*P(2,0)-P(0,0),u.x*P(2,1)-P(0,1),u.x*P(2,2)-P(0,2),
    u.y*P(2,0)-P(1,0),u.y*P(2,1)-P(1,1),u.y*P(2,2)-P(1,2),
    u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),u1.x*P1(2,2)-P1(0,2),
    u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),u1.y*P1(2,2)-P1(1,2));

    //build B vector // cómo asegurarse de que esto está bueno ??
    Matx41d B(-(u.x*P(2,3)-P(0,3)),
    -(u.y*P(2,3)-P(1,3)),
    -(u1.x*P1(2,3)-P1(0,3)),
    -(u1.y*P1(2,3)-P1(1,3)));

    //solve for X
    Mat_<double> X;
    solve(A,B,X,DECOMP_SVD);
    return X;
}

double TriangulatePoints(
    const vector<Point2f>& pt_set1,
    const vector<Point2f>& pt_set2,
    const Mat&K,
    const Matx34d& P,
    const Matx34d& P1,
    vector<Point3d>& pointcloud){

    vector<double> reproj_error;

    Mat Kinv = K.inv();

    for (unsigned int i=0; i<pt_set1.size(); i++) {
        //convert to normalized homogeneous coordinates

        Point3d u(pt_set1[i].x,pt_set1[i].y,1.0);
        Mat_<double> um = Kinv * Mat_<double>(u);
        u = um.at<Point3d>(0);

        Point3d u1(pt_set2[i].x,pt_set2[i].y,1.0);
        Mat_<double> um1 = Kinv * Mat_<double>(u1);
        u1 = um1.at<Point3d>(0);

        //triangulate

        Mat_<double> X = LinearLSTriangulation(u,P,u1,P1);

        //calculate reprojection error
        Mat X_ = cv::Mat::zeros(4,1,CV_64FC1);
        X_.at<double>(0,0) = X(0);
        X_.at<double>(1,0) = X(1);
        X_.at<double>(2,0) = X(2);
        X_.at<double>(3,0) = 1;
        Mat_<double> xPt_img = K * Mat(P1) * X_;
        Point2f xPt_img_(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));

        reproj_error.push_back(norm(xPt_img_ - pt_set2[i]));

        //store 3D point
        pointcloud.push_back(Point3d(X(0),X(1),X(2)));
    }

    //return mean reprojection error
    Scalar me = mean(reproj_error);
    cout <<endl << "mean :" << me[0] << endl;
    return me[0];
}

Matx34d Find_camera_matrix(
    Mat& K,
    vector<vector<KeyPoint> >& trainKeypoints,
    vector<Point2f>& imgpts1,
    vector<Point2f>& imgpts2,
    vector<vector<DMatch> >& Mega_matches,
    int par){

    vector<KeyPoint> kpts1, kpts2;
    kpts1 = trainKeypoints[par];
    kpts2 = trainKeypoints[par+1];
    vector<DMatch> maches = Mega_matches[par];
    for( unsigned int i = 0; i<maches.size(); i++ ){
        // queryIdx is the "left" image
        imgpts1.push_back(kpts1[maches[i].queryIdx].pt);
        // trainIdx is the "right" image
        imgpts2.push_back(kpts2[maches[i].trainIdx].pt);
    }

    Mat status;
    Mat F = findFundamentalMat(imgpts1, imgpts2, FM_RANSAC, 0.1, 0.99, status);
    Mat_<double> E = K.t() * F * K; //according to HZ (9.12)

    /*
    cout << status.rows << endl << endl; // equal to matches size!!
    cout << F << endl << endl;
    cout << E << endl << endl;*/

    Matx33d W(0,-1,0,//HZ 9.13
    1,0,0,
    0,0,1);

    SVD svd(E);// single value decomposition (SVD)
    Mat_<double> R = svd.u * Mat(W) * svd.vt; //HZ 9.19
    Mat_<double> t = svd.u.col(2); //u3
    Matx34d P1( R(0,0),R(0,1), R(0,2), t(0),
    R(1,0),R(1,1), R(1,2), t(1),
    R(2,0),R(2,1), R(2,2), t(2)); //POSE ESTIMATION !!!

    /*
    cout << R << endl << endl;
    cout << t << endl << endl;*/

    CheckCoherentRotation(R);

    return (P1);
}

#endif // SFM

