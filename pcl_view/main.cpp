#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>

/*
#include <boost/thread/thread.hpp>
#include <pcl/console/parse.h>*/

void printUsage (const char* progName)
{
  std::cout << "\n\nUsage: "<<progName<<" path_to_cloud.pcd\n\n" ; 
}

int main(int argc, char** argv)
{
  if(argc != 2)
    printUsage(argv[0]); 

  else {

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    std::string s("/home/javier/Documents/pcl_codes/Clouds/");
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (s+argv[1], *cloud) == -1) //* load the file
    {
      PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
      return (-1);
    }
  
    std::cout << "Loaded "
            << cloud->width * cloud->height
            << " data points from test_pcd.pcd with the following fields: "
            << std::endl;
  
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();

    while (!viewer->wasStopped ())
    {
      viewer->spinOnce (100);
        //boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
  }

  return (0);

}


