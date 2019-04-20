#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <stdlib.h>
#include <stdio.h>

using namespace std;
using namespace cv;

int main (int argc, char **argv)
{
  if (argc !=2)
  {
    cout << "USE: " << argv[0] << " <video-filename>" << endl;
    return 1;
  }

  //Open the video that you pass from the command line
  VideoCapture cap(argv[1]);
  if (!cap.isOpened())
  {
    cerr << "ERROR: Could not open video " << argv[1] << endl;
    return 1;
  }

  int frame_count = 0;
  int image_count = 0;
  bool should_stop = false;

  while(!should_stop)
  {
    Mat frame;
    cap >> frame; //get a new frame from the video

    if (frame.empty())
    {
      should_stop = true; //we arrived to the end of the video
      continue;
    }

    if (frame_count == 60){
        char filename[128];
        sprintf(filename, "image_%04d.jpg", image_count);
        imwrite(filename, frame);
        image_count++;
        frame_count = 0;
    }
    frame_count++;
  }

  return 0;
}
