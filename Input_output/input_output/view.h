#ifndef VIEW_H
#define VIEW_H

#include <iostream>
#include <sstream>
#include <string>

#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;
class view
{
private:
    static int numOfviews;
    string File_name;
    Mat image;
    /*Matx34d pose;
    int n_features;
    Mat features;*/


public:
    string GetName(){return File_name;}
    void SetFile_name(string file_name){this->File_name = file_name;}
    Mat GetMat(){return image;}
    void SetMat(Mat img){this->image = img;}

    void SetAll(string, Mat);
    view(string, Mat);
    view();
    ~view();

};

#endif // VIEW_H
