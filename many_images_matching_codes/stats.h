#ifndef STATS_H
#define STATS_H

#include <iostream>
#include <fstream>


struct Stats
{
    int id;
    int matches;
    double ratio;
    int keypoints;
    int trainKeypoints;

    Stats() :
        matches(0),
        ratio(0),
        keypoints(0),
        trainKeypoints(0)
    {}

    Stats& operator+=(const Stats& op) {
        matches += op.matches;
        ratio += op.ratio;
        keypoints += op.keypoints;
        trainKeypoints += trainKeypoints;
        return *this;
    }
    Stats& operator/=(int num)
    {
        matches /= num;
        ratio /= num;
        keypoints /= num;
        trainKeypoints /= num;
        return *this;
    }
};

#endif // STATS_H
