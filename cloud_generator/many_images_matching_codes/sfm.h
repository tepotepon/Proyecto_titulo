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

Matx34d Find_camera_matrix(
    Mat& K,
    vector<Point2f>& imgpts1,
    vector<Point2f>& imgpts2
    ){

    Mat status;
    Mat F = findFundamentalMat(imgpts1, imgpts2, FM_RANSAC, 0.1, 0.99, status);
    Mat_<double> E = K.t() * F * K; //according to HZ (9.12)
    cout << "status: " << status << endl;

    Matx33d W(0,-1,0,//HZ 9.13
    1,0,0,
    0,0,1);

    SVD svd(E);// single value decomposition (SVD)
    Mat_<double> R = svd.u * Mat(W) * svd.vt; //HZ 9.19
    Mat_<double> t = svd.u.col(2); //u3
    Matx34d P1( R(0,0),R(0,1), R(0,2), t(0),
    R(1,0),R(1,1), R(1,2), t(1),
    R(2,0),R(2,1), R(2,2), t(2)); //POSE ESTIMATION !!!

    CheckCoherentRotation(R);

    return(P1);
}


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

    for (size_t i=0; i<pt_set1.size(); i++) {
        //convert to normalized homogeneous coordinates

        Point3d u(pt_set1[i].x,pt_set1[i].y,1.0);
        Mat_<double> um = Kinv * Mat_<double>(u);
        u.x = um(0);
        u.y = um(1);
        u.z = um(2);

        Point3d u1(pt_set2[i].x,pt_set2[i].y,1.0);
        Mat_<double> um1 = Kinv * Mat_<double>(u1);
        u1.x = um1(0);
        u1.y = um1(1);
        u1.z = um1(2);

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
        Point.origin.push_back(par+1);
        Point.index.push_back(maches[i].trainIdx);
        pcloud.push_back(Point);
    }

    //return mean reprojection error
    Scalar me = mean(reproj_error);
    cout << "reproj_error mean:" << me[0] << endl;
    return me[0];
}

Matx34d P1_from_correspondence(
        vector<CloudPoint>& pcloud,
        vector<vector<KeyPoint> >& trainKeypoints,
        vector<vector<DMatch> >& Mega_matches,
        Mat& K,
        Mat& distcoeff,
        int old_view){

    
    vector<Point3f> correspondence_cloud;
    vector<Point2f> imgPoints;
    int working_view;
    vector<KeyPoint> new_kpts;
    vector<DMatch> new_matches;
    vector<int> pcloud_status(pcloud.size(),0);

    working_view = old_view+1;
    new_kpts = trainKeypoints[working_view];    
    new_matches = Mega_matches[old_view]; // NOTA MENTAL: match new + old ... no confundir old + old_old
    
    // Implementar ciclo FOR si hago más matches (scan all previews views).
    //scan the 2D-2D matched-points
    for (size_t i = 0; i<new_matches.size(); i++) {
        // the index of the matching 2D point in <new> + <old_view>
        int old_view_keypointidx = new_matches[i].queryIdx;
        //scan existing cloud to see if this point from <old_view> exists
        for (size_t some_point = 0; some_point<pcloud.size(); some_point++) {
            // see if this 2D point from <old_view> contributed to this 3D point in the cloud
                vector<int> opciones = pcloud[some_point].index;
                for(size_t j=0 ; j < opciones.size() ; j++) {
                    if (old_view_keypointidx == opciones[j] && pcloud_status[some_point] == 0) //prevent duplicates
                    {
                        //3d point in cloud
                        correspondence_cloud.push_back(pcloud[some_point].pt);
                        //2d point in image <working_view>
                        Point2f pt = new_kpts[new_matches[i].trainIdx].pt;
                        imgPoints.push_back(pt);
                        pcloud_status[some_point] = 1;
                        break;
                    }
                }
        }
    }

    cout<< "3d-2d points correspondences: "<< correspondence_cloud.size() << endl;
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
        vector<KeyPoint>& kpts1,
        vector<KeyPoint>& kpts2,
        vector<DMatch>& matches,
        vector<Vec3b>& colors1,
        vector<Point2f>& imgpts1,
        vector<Point2f>& imgpts2,
        vector<Vec3b>& true_colors){

    for(size_t i = 0; i<matches.size(); i++ ){
        // queryIdx is the "left" image
        int q = matches[i].queryIdx; 
        int t = matches[i].trainIdx;
        Vec3b color1 = colors1[q];
        imgpts1.push_back(kpts1[q].pt);
        // trainIdx is the "right" image
        imgpts2.push_back(kpts2[t].pt);
        true_colors.push_back(color1);
    }
}

#endif // SFM

