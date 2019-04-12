#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

int main (int argc, char **argv)
{
  if (argc !=2)
  {
    std::cout << "USE: " << argv[0] << " <video-filename>" << std::endl;
    return 1;
  }

  //Open the video that you pass from the command line
  cv::VideoCapture cap(argv[1]);
  if (!cap.isOpened())
  {
    std::cerr << "ERROR: Could not open video " << argv[1] << std::endl;
    return 1;
  }

  int frame_count = 0;
  int image_count = 0;
  bool should_stop = false;

  while(!should_stop)
  {
    cv::Mat frame;
    cap >> frame; //get a new frame from the video

    if (frame.empty())
    {
      should_stop = true; //we arrived to the end of the video
      continue;
    }

    if (frame_count == 60){
        char filename[128];
        sprintf(filename, "image_%04d.jpg", image_count);
        cv::imwrite(filename, frame);
        image_count++;
        frame_count = 0;
    }
    frame_count++;
  }

  return 0;
}
