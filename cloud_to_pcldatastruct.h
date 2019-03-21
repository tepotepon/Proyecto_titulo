#ifndef CLOUD_TO_PCLDATASTRUCT
#define CLOUD_TO_PCLDATASTRUCT

#include <string>
#include <iostream>
#include <cstdint>

#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <opencv2/core.hpp>

#include "cloudpoint.h"


using namespace std;

void PopulatePCLPointCloud(
        vector<CloudPoint>& pointcloud,
        vector<cv::Vec3b>& pointcloud_RGB,
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud)//Populate point cloud
{
    cout<<"Creating PCL point cloud...";
    cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (unsigned int i=0; i<pointcloud.size(); i++) {
        // get the RGB color value for the point
        cv::Vec3b rgbv= pointcloud_RGB[i];
        cv::Point3d point = pointcloud[i].pt;

        /* NO ENTIENDO ESTA WEA
        if (pointcloud_RGB.size() >= i)
            rgbv = pointcloud_RGB[i];*/

        // check for erroneous coordinates (NaN, Inf, etc.)
        /*
        if (pointcloud[i].x != pointcloud[i].x || isnan(pointcloud[i].x) ||
        pointcloud[i].y != pointcloud[i].y || isnan(pointcloud[i].y) ||
        pointcloud[i].z != pointcloud[i].z || isnan(pointcloud[i].z) ||
        fabsf(pointcloud[i].x) > 10.0 ||
        fabsf(pointcloud[i].y) > 10.0 ||
        fabsf(pointcloud[i].z) > 10.0) {
            continue;
        }*/

        pcl::PointXYZRGB pclp;
        // 3D coordinates
        pclp.x = point.x;
        pclp.y = point.y;
        pclp.z = point.z;
        // RGB color, needs to be represented as an integer

        uint32_t rgb = ((uint32_t)rgbv[2] << 16 | (uint32_t)rgbv[1] << 8 | (uint32_t)rgbv[0]);
        pclp.rgb = *reinterpret_cast<int*>(&rgb);
        cloud->push_back(pclp);
    }

    cloud->width = (uint32_t) cloud->points.size(); // number of points
    cloud->height = 1; // a list of points, one row of data
    cout << "done writting pcl rgb cloud" << endl ;
    cout <<"cloud->width: " << cloud->width << endl;


}

/*
void SORFilter(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
    
    // Create the filtering object

    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;

    sor.setInputCloud(cloud);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    cout << "todo ok" << endl;

    sor.filter(*cloud_filtered);
    cerr<<"Cloud after SOR filtering: "<<cloud_filtered->width *
    cloud_filtered->height <<" data points "<< endl;
    copyPointCloud(*cloud_filtered,*cloud);
}*/

#endif // CLOUD_TO_PCLDATASTRUCT

