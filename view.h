#ifndef VIEW_H
#define VIEW_H

#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

class view
{
private:
    static int numOfviews;
    string File_name;
    Mat image;
    vector<KeyPoint> Keypoints;
    /*Matx34d pose;
    int n_features;*/


public:
    string GetName(){return File_name;}
    void SetFile_name(string file_name){this->File_name = file_name;}

    Mat GetMat(){return image;}
    void SetMat(Mat img){this->image = img;}

    vector<KeyPoint> GetKeypoints(){return Keypoints;}
    void SetKeypoints(vector<KeyPoint> Keypoints){this->Keypoints = Keypoints;}

    int getNumOfviews(){return numOfviews;}

    void SetAll(string, Mat);
    view(string, Mat);
    view(vector<KeyPoint>);
    view();
    ~view();

};

#endif // VIEW_H
