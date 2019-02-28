#ifndef CLOUD_TO_PCLDATASTRUCT
#define CLOUD_TO_PCLDATASTRUCT

#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

#include <opencv2/imgproc/imgproc.hpp>

using namespace pcl;
using namespace std;

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;

void PopulatePCLPointCloud(
        const vector<Point3d>& pointcloud,
        const vector<cv::Vec3b>& pointcloud_RGB)//Populate point cloud
{
    cout<<"Creating PCL point cloud...";
    cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (unsigned int i=0; i<pointcloud.size(); i++) {
        // get the RGB color value for the point
        Vec3b rgbv(255,255,255);
        if (pointcloud_RGB.size() >= i)
            rgbv = pointcloud_RGB[i];

        // check for erroneous coordinates (NaN, Inf, etc.)
        if (pointcloud[i].x != pointcloud[i].x || isnan(pointcloud[i].x) ||
        pointcloud[i].y != pointcloud[i].y || isnan(pointcloud[i].y) ||
        pointcloud[i].z != pointcloud[i].z || isnan(pointcloud[i].z) ||
        fabsf(pointcloud[i].x) > 10.0 ||
        fabsf(pointcloud[i].y) > 10.0 ||
        fabsf(pointcloud[i].z) > 10.0) {
            continue;
        }

        pcl::PointXYZRGB pclp;
        // 3D coordinates
        pclp.x = pointcloud[i].x;
        pclp.y = pointcloud[i].y;
        pclp.z = pointcloud[i].z;
        // RGB color, needs to be represented as an integer
        uint32_t rgb = ((uint32_t)rgbv[2] << 16 | (uint32_t)rgbv[1] << 8 |
        (uint32_t)rgbv[0]);
        pclp.rgb = *reinterpret_cast<float*>(&rgb);
        cloud->push_back(pclp);
    }

    cloud->width = (uint32_t) cloud->points.size(); // number of points
cloud->height = 1; // a list of points, one row of data
}

#endif // CLOUD_TO_PCLDATASTRUCT

