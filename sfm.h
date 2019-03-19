#ifndef SFM
#define SFM

#include <opencv2/core.hpp>
#include "cloudpoint.h"

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
    int par,
    vector<vector<DMatch> >& Mega_matches,
    vector<Point2f>& pt_set1,
    vector<Point2f>& pt_set2,
    const Mat&K,
    const Matx34d& P,
    const Matx34d& P1,
    vector<CloudPoint>& pcloud);

Matx34d Find_camera_matrix(
    Mat& K,
    const vector<vector<KeyPoint> >& trainKeypoints,
    vector<Point2f>& imgpts1,
    vector<Point2f>& imgpts2,
    vector<vector<DMatch> >& Mega_matches,
    int par); //solo para los primeros 2 ... podria borrar "par"

Matx34d P1_from_correspondence(
        vector<CloudPoint>& pcloud,
        vector<vector<KeyPoint> >&trainKeypoints,
        vector<vector<DMatch> >& Mega_matches,
        Mat& K,
        Mat& distcoeff,
        int wview,
        int oview);

void sort_imgpts(
        int par,
        vector<vector<KeyPoint> >& trainKeypoints,
        vector<vector<DMatch> >& Mega_matches,
        vector<Point2f>& imgpts1,
        vector<Point2f>& imgpts2);


bool CheckCoherentRotation(Mat_<double>& R){
    if(fabsf(determinant(R))-1.0 > 1e-07) {
        cerr<<"det(R) != +-1.0, this is not a rotation matrix" << endl;
        return false;
    }
    else {
        cout << "det(R) ok: valid rotation" << endl;
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
    int par,
    vector<vector<DMatch> >& Mega_matches,
    vector<Point2f>& pt_set1,
    vector<Point2f>& pt_set2,
    const Mat&K,
    const Matx34d& P,
    const Matx34d& P1,
    vector<CloudPoint>& pcloud){

    vector<double> reproj_error;

    Mat Kinv = K.inv();
    vector<DMatch> maches = Mega_matches[par];

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
        Mat X_ = Mat::zeros(4,1,CV_64FC1);
        X_.at<double>(0,0) = X(0);
        X_.at<double>(1,0) = X(1);
        X_.at<double>(2,0) = X(2);
        X_.at<double>(3,0) = 1;
        Mat_<double> xPt_img = K * Mat(P) * X_;
        Point2f xPt_img_(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));

        reproj_error.push_back(norm(xPt_img_ - pt_set1[i]));

        //store 3D point
        CloudPoint Point;
        Point.pt = Point3d(X(0),X(1),X(2));
        Point.index_of_2d_origin = vector<int>(par+1,maches[i].trainIdx);
        pcloud.push_back(Point);
    }

    //return mean reprojection error
    Scalar me = mean(reproj_error);
    cout << "mean :" << me[0] << endl;
    return me[0];
}

Matx34d Find_camera_matrix(
    Mat& K,
    vector<Point2f>& imgpts1,
    vector<Point2f>& imgpts2
        ){


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

Matx34d P1_from_correspondence(
        vector<CloudPoint>& pcloud,
        vector<vector<KeyPoint> >&trainKeypoints,
        vector<vector<DMatch> >& Mega_matches,
        Mat& K,
        Mat& distcoeff,
        int wview,
        int oview){

    //check for matches between i'th frame and 0'th frame (and thus the current cloud)
    vector<Point3f> correspondence_cloud;
    vector<Point2f> imgPoints;
    vector<int> pcloud_status(pcloud.size(),0);
    vector<KeyPoint> kpts1;
    int working_view = wview;
    int old_view = oview;

    kpts1 = trainKeypoints[working_view];

    //scan all previews views.
    //for (set<int>::iterator viewwewe = good_views.begin(); viewwewe != good_views.end(); ++viewwewe){
    vector<DMatch> NaOmatches = Mega_matches[old_view];

    //scan the 2D-2D matched-points
    for (unsigned int iterator = 0; iterator<NaOmatches.size(); iterator++) {
        // the index of the matching 2D point in <old_view>
        int old_view_keypointidx = NaOmatches[iterator].queryIdx;
        //scan existing cloud to see if this point from <old_view> exists
        for (unsigned int some_point = 0; some_point<pcloud.size(); some_point++) {
            // see if this 2D point from <old_view> contributed to this 3D point in the cloud
            if (old_view_keypointidx == pcloud[some_point].index_of_2d_origin[old_view-1] && pcloud_status[some_point] == 0) //prevent duplicates
            {
                //3d point in cloud
                correspondence_cloud.push_back(pcloud[some_point].pt);
                //2d point in image <working_view>
                Point2d pt_ = kpts1[NaOmatches[iterator].trainIdx].pt;
                imgPoints.push_back(pt_);
                pcloud_status[some_point] = 1;
                break;
            }
        }
    }
    //}

    cout<< "3d-2d points correspondences: "<< correspondence_cloud.size() <<endl;
    Mat_<double> t,rvec,R;
    solvePnPRansac(correspondence_cloud, imgPoints, K, distcoeff, rvec, t, false);
    //get rotation in 3x3 matrix form
    Rodrigues(rvec, R);
    Matx34d P1 = Matx34d(R(0,0),R(0,1),R(0,2),t(0),
    R(1,0),R(1,1),R(1,2),t(1),
    R(2,0),R(2,1),R(2,2),t(2));

    CheckCoherentRotation(R);

    return(P1);
}

void sort_imgpts(
        vector<vector<KeyPoint> >& trainKeypoints,
        vector<vector<DMatch> >& Mega_matches,
        vector<vector<Point2f> >& all_imgpts1,
        vector<vector<Point2f> >& all_imgpts2){

    vector<KeyPoint> kpts1, kpts2;
    vector<Point2f> imgpts1;
    vector<Point2f> imgpts2;
    for(uint j=1; j<=Mega_matches.size() ; j++){
        kpts1 = trainKeypoints[j-1];
        kpts2 = trainKeypoints[j];
        vector<DMatch> maches = Mega_matches[j-1];
        imgpts1.clear();
        imgpts2.clear();
        for( unsigned int i = 0; i<maches.size(); i++ ){
            // queryIdx is the "left" image
            imgpts1.push_back(kpts1[maches[i].queryIdx].pt);
            // trainIdx is the "right" image
            imgpts2.push_back(kpts2[maches[i].trainIdx].pt);
        }
        all_imgpts1.push_back(imgpts1);
        all_imgpts2.push_back(imgpts2);
    }
}

#endif // SFM

