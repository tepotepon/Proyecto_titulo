#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;


const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio
const float focal = 6.6925901405703587e+02;

bool CheckCoherentRotation(cv::Mat_<double>& R) {
    if(fabsf(determinant(R))-1.0 > 1e-07) {
        cerr<<"det(R) != +-1.0, this is not a rotation matrix"<<endl;
        return false;
    }
    return true;
}

struct CloudPoint {
    Point3d pt;
    vector<int>index_of_2d_origin;
};

Mat_<double> LinearLSTriangulation(
    Point3d u,//homogenous image point (u,v,1)
    Matx34d P,//camera 1 matrix
    Point3d u1,//homogenous image point in 2nd camera
    Matx34d P1//camera 2 matrix
){
    //build A matrix
    Matx43d A(u.x*P(2,0)-P(0,0),u.x*P(2,1)-P(0,1),u.x*P(2,2)-P(0,2),
    u.y*P(2,0)-P(1,0),u.y*P(2,1)-P(1,1),u.y*P(2,2)-P(1,2),
    u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),u1.x*P1(2,2)-P1(0,2),
    u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),u1.y*P1(2,2)-P1(1,2));

    //build B vector
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
    const vector<KeyPoint>& pt_set1,
    const vector<KeyPoint>& pt_set2,
    const Mat&K,
    const Matx34d& P,
    const Matx34d& P1,
    vector<Point3d>& pointcloud)
{
    vector<double> reproj_error;
    Mat Kinv = K.inv();

    for (unsigned int i=0; i<pt_set1.size(); i++) {
        //cout << "-------------------------" << endl;
        //convert to normalized homogeneous coordinates
        Point2f kp = pt_set1[i].pt;
        Point3d u(kp.x,kp.y,1.0);
        Mat_<double> um = Kinv * Mat_<double>(u);
        u = um.at<Point3d>(0);

        //convert to normalized homogeneous coordinates
        Point2f kp1 = pt_set2[i].pt;
        //cout << "kp1: " << endl;
        //cout << kp1 << endl;
        Point3d u1(kp1.x,kp1.y,1.0);
        Mat_<double> um1 = Kinv * Mat_<double>(u1);
        u1 = um1.at<Point3d>(0);

        //triangulate
        Mat_<double> X = LinearLSTriangulation(u,P,u1,P1);

        //store 3D point
        pointcloud.push_back(Point3d(X(0),X(1),X(2)));

        //calculate reprojection error
        Mat X_ = cv::Mat::zeros(4,1,CV_64FC1);
        X_.at<double>(0,0) = X(0);
        X_.at<double>(1,0) = X(1);
        X_.at<double>(2,0) = X(2);
        X_.at<double>(3,0) = 1;
        Mat_<double> xPt_img = K * Mat(P1) * X_;
        Point2f xPt_img_(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));
        /*cout << "xPt_img_: " << endl;
        cout << xPt_img_ << endl;
        cout << "xPt_img_-kp1" << endl;
        cout << xPt_img_-kp1 << endl;*/
        reproj_error.push_back(norm(xPt_img_-kp1));
    }
    //return mean reprojection error
    Scalar me = mean(reproj_error);
    cout << "me: " << endl;
    cout << me << endl;
    return me[0];
}

int main(){

    Mat img1, img2;
    Mat desc1, desc2;
    Mat_<double> R;
    Mat_<double> t;
    Mat mask;

    vector<KeyPoint> kpts1, kpts2;
    vector<KeyPoint> matched1, matched2;
    vector<Point2f> imgpts1,imgpts2;

    vector<DMatch> matches;
    vector<Point3d> pointcloud;

    Mat K = cv::Mat::zeros(3,3,CV_64FC1);
    K.at<double>(0,0) = focal;
    K.at<double>(1,1) = focal;
    K.at<double>(0,2) = 640.;
    K.at<double>(1,2) = 360.;
    K.at<double>(2,2) = 1;

    img1 = imread("data/fotos_usm/foto1.jpg",IMREAD_GRAYSCALE);
    img2 = imread("data/fotos_usm/foto2.jpg", IMREAD_GRAYSCALE);

    if(img1.empty()) {
        cerr << "Error reading image " << endl;
        return 1;
    }

    Ptr<cv::xfeatures2d::SiftFeatureDetector> bdesc = cv::xfeatures2d::SiftFeatureDetector::create();
    bdesc->detectAndCompute(img1, noArray(), kpts1, desc1);
    bdesc->detectAndCompute(img2, noArray(), kpts2, desc2);

    BFMatcher matcher;
    matcher.match(desc1, desc2, matches);

    double max_dist = 0, min_dist = DBL_MAX;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < matches.size(); i++ ) {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

   // -- Align matches into arrays (same indexes)
    for( int i = 0; i < matches.size(); i++ ){
        if( matches[i].distance < nn_match_ratio*max_dist ) {
            matched1.push_back(kpts1[matches[i].queryIdx]);
            matched2.push_back(kpts2[matches[i].trainIdx]);
        }
    }

    //keypoint structure to Point2f structure:
    for( int i = 0; i < matches.size(); i++ ){
        if( matches[i].distance < nn_match_ratio*max_dist ) {
            imgpts1.push_back(kpts1[matches[i].queryIdx].pt);
            imgpts2.push_back(kpts2[matches[i].trainIdx].pt);
        }
    }

    cout << "Matching Results" << endl;
    cout << "*******************************" << endl;
    cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
    cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
    cout << "# Matches:                            \t" << matched1.size() << endl;
    cout << endl;

    Mat F = findFundamentalMat(imgpts1, imgpts2, FM_RANSAC, 0.1, 0.99);
    Mat E = findEssentialMat(imgpts1, imgpts2, focal,Point2d(0,0),RANSAC,0.999,1.0);

    recoverPose(E,imgpts1,imgpts2,R,t,focal,Point2d(0,0),mask);

    if (!CheckCoherentRotation(R))
        cout<<"resulting rotation is not coherent\n";

    Matx34d Pleft = Matx34d::eye();
    Matx34d Pright = Matx34d(R(0,0), R(0,1), R(0,2), t(0),

                         R(1,0), R(1,1), R(1,2), t(1),

                         R(2,0), R(2,1), R(2,2), t(2));

    //mean_reprojection_error
    double mre = TriangulatePoints(matched1,matched2,K,Pleft,Pright,pointcloud);

    vector<CloudPoint> pcloud; //our global 3D point cloud

    //check for matches between i'th frame and 0'th frame (and thus the current cloud)
    vector<Point3f> ppcloud;
    vector<Point2f> imgPoints;
    vector<int> pcloud_status(pcloud.size(),0);

    //scan the views we already used (good_views)
    for (set<int>::iterator done_view = good_views.begin(); done_view != good_views.end(); ++done_view){
       int old_view = *done_view; //a view we already used for reconstrcution
       //check for matches_from_old_to_working between <working_view>'thframe and <old_view>'th frame (and thus the current cloud)
       vector<DMatch> matches_from_old_to_working = matches_matrix[make_pair(old_view,working_view)];
       //scan the 2D-2D matched-points
       for (unsigned int match_from_old_view=0; match_from_old_view<matches_from_old_to_working.size(); match_from_old_view++){
           // the index of the matching 2D point in <old_view>
           int idx_in_old_view = matches_from_old_to_working[match_from_old_view].queryIdx;
           //scan the existing cloud to see if this point from <old_view> exists
           for (unsigned int pcldp=0; pcldp<pcloud.size(); pcldp++){
                // see if this 2D point from <old_view> contributed to this 3D point in the cloud
                if (idx_in_old_view == pcloud[pcldp].index_of_2d_origin[old_view]&& pcloud_status[pcldp] == 0){
                    //3d point in cloud
                    ppcloud.push_back(pcloud[pcldp].pt);
                    //2d point in image <working_view>
                    Point2d pt_ = imgpts[working_view][matches_from_old_to_working[match_from_old_view].trainIdx].pt;
                    imgPoints.push_back(pt_);
                    pcloud_status[pcldp] = 1;
                    break;
                }
            }
        }
    }

    cout<<"found "<<ppcloud.size() <<" 3d-2d point correspondences"<<endl;

    //cout << pointcloud << endl;
    return 0;
}
