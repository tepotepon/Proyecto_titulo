#ifndef UTILS_H
#define UTILS_H


#include "stats.h"

using namespace std;
using namespace cv;


void drawStatistics(Mat image, const Stats& stats);
void printStatistics(string name, Stats stats);
void save_on_stats(int matches,int queryKeypoints,int trainKeypoints,Stats stat);


void drawStatistics(Mat image, const Stats& stats)
{
    static const int font = FONT_HERSHEY_PLAIN;
    stringstream str0,str1, str2, str3, str4;

    str0 << "foto id: " << stats.id;
    str1 << "Matches: " << stats.matches;
    str2 << "Keypoints1: " << stats.keypoints;
    str3 << "keypoints2: " << stats.trainKeypoints;

    putText(image, str0.str(), Point(0, image.rows - 140), font, 1, Scalar::all(255), 1);
    putText(image, str1.str(), Point(0, image.rows - 120), font, 1, Scalar::all(255), 1);
    putText(image, str2.str(), Point(0, image.rows - 90), font, 1, Scalar::all(255), 1);
    putText(image, str3.str(), Point(0, image.rows - 60), font, 1, Scalar::all(255), 1);
}

//Falta terminar función, agregar mas datos.
void save_on_stats(int matches,int queryKeypoints,int trainKeypoints,Stats& stat)
{
    /*
    cout << "matches: " << matches << endl;
    cout << "queryKeypoints: " << queryKeypoints << endl;
    cout << "trainKeypoints: " << trainKeypoints << endl;*/

    stat.matches = matches;
    //stat.ratio = ;
    stat.keypoints = queryKeypoints;
    stat.trainKeypoints = trainKeypoints;

    cout << stat.matches << endl;
}

#endif // UTILS_H

