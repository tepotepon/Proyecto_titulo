#ifndef CLOUDPOINT
#define CLOUDPOINT

#include <opencv2/core.hpp>
#include <string>

using namespace cv;
using namespace std;

struct CloudPoint {
    Point3d pt;
    vector<int> index;
    vector<int> origin;
};

#endif // CLOUDPOINT

