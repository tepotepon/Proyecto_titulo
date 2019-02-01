#include "view.h"

#include <iostream>
#include <sstream>
#include <string>

#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

int view::numOfviews = 0;

view::view()
{
    this->File_name = "";
    //this->image = ;
    //view::numOfviews++;
}

view::view(string filename, Mat img)
{
    this->File_name = filename;
    this->image = img;
    view::numOfviews++;
}

void view::SetAll(string filename,Mat img){
    this->File_name = filename;
    this->image = img;
    view::numOfviews++;
}

