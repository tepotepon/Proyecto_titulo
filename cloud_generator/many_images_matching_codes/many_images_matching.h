#ifndef MANY_IMAGES_MATCHING_H
#define MANY_IMAGES_MATCHING_H

#include "stats.h"
#include "utils.h"
#include "view.h"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

static void readTrainFilenames( const string& filename, string& dirName, vector<string>& trainFilenames );

static bool createDetectorDescriptorMatcher( const string& descriptorType,const string& matcherType,
                                             Ptr<Feature2D>& featureDetector,Ptr<DescriptorMatcher>& descriptorMatcher );

static bool readImages( const string& trainFilename,vector <Mat>& trainImages,
                        vector<string>& trainImageNames,vector<view>& views);

static void detectKeypoints( const vector<Mat>& trainImages,vector<vector<KeyPoint> >& trainKeypoints,Ptr<Feature2D>& featureDetector,vector<view> views);

static void computeDescriptors( const vector<Mat>& trainImages, vector<vector<KeyPoint> >& trainKeypoints, vector<Mat>& trainDescriptors,
                                Ptr<Feature2D> featureDetector);

static void matchDescriptors(vector<vector<DMatch> >& Mega_matches_aux,const vector<Mat>& trainDescriptors,Ptr<DescriptorMatcher>& descriptorMatcher);

void saveResultImages( vector<vector<DMatch> >& Mega_matches_aux, vector<Mat>& trainImages, vector<vector<KeyPoint> >& trainKeypoints,
                              vector<string>& trainImagesNames, string& resultDir );

void read_from_files(const vector<vector<KeyPoint> >& trainKeypoints,vector<view>& views,vector<vector<DMatch> >& Mega_matches);

static void readTrainFilenames( const string& filename, string& dirName, vector<string>& trainFilenames )
{
    trainFilenames.clear();

    ifstream file( filename.c_str() );
    if ( !file.is_open() )
        return;

    size_t pos = filename.rfind('\\');
    char dlmtr = '\\';
    if (pos == String::npos)
    {
        pos = filename.rfind('/');
        dlmtr = '/';
    }
    dirName = pos == string::npos ? "" : filename.substr(0, pos) + dlmtr;

    while( !file.eof() )
    {
        string str; getline( file, str );
        if( str.empty() ) break;
        trainFilenames.push_back(str);
    }
    file.close();
}

static bool readImages(const string& trainFilename,
                 vector <Mat>& trainImages, vector<string>& trainImageNames,vector<view>& views)
{
    string trainDirName;
    readTrainFilenames( trainFilename, trainDirName, trainImageNames );

    if( trainImageNames.empty() )
    {
        cout << "Train image filenames can not be read." << endl << ">" << endl;
        return false;
    }

    int readImageCount = 0;

    views.clear();

    for(size_t i = 0; i < trainImageNames.size(); i++)
    {
        string filename = trainDirName + trainImageNames[i];
        Mat img = imread(filename, 1);         

        if(img.empty()){
            cout << "Train image " << filename << " can not be read." << endl;
        }
        else {
            readImageCount++;
            trainImages.push_back( img );
            //GOTTA FIX THIS SHIT 
            view vista(filename,img);
            views.push_back(vista);
        }
     }

    if(!readImageCount)
    {
        cout << "All train images can not be read." << endl << ">" << endl;
        return false;
    }
    else
        cout << readImageCount << " train images were read." << endl;
    return true;
}

static bool createDetectorDescriptorMatcher(const string& descriptorType, const string& matcherType,
                                      Ptr<Feature2D>& featureDetector,
                                      Ptr<DescriptorMatcher>& descriptorMatcher )
{
    cout << "< Creating feature detector, descriptor extractor and descriptor matcher ..." << endl;

    string Chosen_Type;
    cout << endl << "choose Descriptor Type: (SIFT/SURF) ";
    cin >> Chosen_Type;

    if (Chosen_Type == "SURF")
        featureDetector = SURF::create();
    else if(Chosen_Type == "SIFT")
            featureDetector = SIFT::create();
        else 
            cout << "Not valid: default selected: SURF" << endl; 

    descriptorMatcher = DescriptorMatcher::create(matcherType);
    cout << ">" << endl;

    bool isCreated = !( featureDetector.empty() ||descriptorMatcher.empty() );
    if( !isCreated )
        cout << "Can not create feature detector or descriptor extractor or descriptor matcher of given types." << endl << ">" << endl;

    return isCreated;
}

static void detectKeypoints(const vector<Mat>& trainImages, vector<vector<KeyPoint> >& trainKeypoints,
                      Ptr<Feature2D>& featureDetector,vector<view> views,vector<vector<Vec3b> >& colors)
{
    cout << "converting images to gray scale for latter keypoint extraction..." << endl; 
    vector<Mat> gray_train; 
    TickMeter tm;
    tm.start();
    for (size_t i =0 ; i < trainImages.size() ; i++){
        Mat img; 
        cvtColor(trainImages[i],img,COLOR_BGR2GRAY);
        gray_train.push_back(img); 
    }

    tm.stop();
    double converting_time = tm.getTimeMilli();

    tm.start();
    cout << endl << "< Extracting keypoints from images..." << endl;
    featureDetector->detect( gray_train, trainKeypoints );
    cout << ">" << endl;
    tm.stop(); 
    double extraction_time = tm.getTimeMilli();

    cout << "convertion time: " << converting_time << " ms; extraction time: " << extraction_time << " ms" << endl;

    /////////////// store keypoints on view class ////////////////////


    //save keypoints on file
    cout << "Saving keypoints into file..." ;
    FileStorage fs("keypoints_colors", FileStorage::WRITE);
    for (size_t i = 0; i<trainKeypoints.size(); i++){
        Mat pic = trainImages[i];
        vector<KeyPoint> Keypoints = trainKeypoints[i];
        vector<Vec3b> colores;
        for (size_t j =0 ; j< Keypoints.size() ; j++){
            Point2f point = Keypoints[j].pt;
            Vec3b bgr = pic.at<Vec3b>(point.y,point.x);
            colores.push_back(bgr);
        }
        stringstream ss;
        ss << i;
        fs << "keypoints " + ss.str();
        fs << trainKeypoints[i];
        fs << "colors " + ss.str();
        fs << colores;
        colors.push_back(colores);
    }

    // explicit close
    fs.release();
    cout << "Done." << endl << endl;

}

static void computeDescriptors(const vector<Mat>& trainImages, vector<vector<KeyPoint> >& trainKeypoints,
                               vector<Mat>& trainDescriptors, Ptr<Feature2D> featureDetector)
{
    cout << "< Computing descriptors for keypoints..." << endl;
    featureDetector->compute(trainImages, trainKeypoints, trainDescriptors );

    int totalTrainDesc = 0;
    for( vector<Mat>::const_iterator tdIter = trainDescriptors.begin(); tdIter != trainDescriptors.end(); tdIter++ )
        totalTrainDesc += tdIter->rows;

    cout << "Total train descriptors count: " << totalTrainDesc << endl;
    cout << ">" << endl;
}

static void matchDescriptors( vector<vector<KeyPoint> >& trainKeypoints, vector<vector<DMatch> >& Mega_matches_aux,const vector<Mat>& trainDescriptors,
                              Ptr<DescriptorMatcher>& descriptorMatcher)
{
    cout << "matching process: " << endl;
    TickMeter tm;

    tm.start();
    descriptorMatcher->add( trainDescriptors );
    descriptorMatcher->train();
    tm.stop();
    double buildTime = tm.getTimeMilli();

    tm.start();

    for (size_t i =0; i < trainDescriptors.size()-1; i++){
        vector<DMatch> matches_aux;
        vector<DMatch> good_matches;
        vector<KeyPoint> kps1 = trainKeypoints[i]; 
        vector<KeyPoint> kps2 = trainKeypoints[i+1]; 
        double min_dist = 100;

        descriptorMatcher->match( trainDescriptors[i],trainDescriptors[i+1], matches_aux);
        cout << "-----------------------------------------------" << endl;
      
        cout << "matches size: " << matches_aux.size() << endl; 

        for(int j = 0; j < matches_aux.size(); j++)
            {
                double dist = matches_aux[j].distance;
                if( dist < min_dist ) min_dist = dist;
            }
        cout << "min dist: "<< min_dist << endl;

        for(size_t a = 0; a < matches_aux.size(); a++)
        {
            if(matches_aux[a].distance <= 10*min_dist){
                matches_aux[a].imgIdx = i+1;
                good_matches.push_back(matches_aux[a]);
            }
        }
        cout << good_matches.size() << " good matches (DISTANCE) out of "<< matches_aux.size() << endl;
        matches_aux.clear(); 

        vector<Point2f> imgpts1;
        vector<Point2f> imgpts2;

        for(size_t i = 0; i<good_matches.size(); i++ ){
            // queryIdx is the "left" image
            imgpts1.push_back(kps1[good_matches[i].queryIdx].pt);
            // trainIdx is the "right" image
            imgpts2.push_back(kps2[good_matches[i].trainIdx].pt);
        }

        Mat mask;
        Mat H = findHomography(imgpts1, imgpts2, RANSAC,7.0, mask);
        imgpts1.clear(); 
        imgpts2.clear(); 

        vector<uchar> array;
        if (mask.isContinuous()) {
          array.assign((uchar*)mask.datastart, (uchar*)mask.dataend);
        } else {
          for (int i = 0; i < mask.rows; ++i) {
            array.insert(array.end(), mask.ptr<uchar>(i), mask.ptr<uchar>(i)+mask.cols);
            cout << i << "NOT continuos" << endl;
          }
        }

        for(size_t i =0; i<array.size() ; i++){
            if(array[i] !=0){
                matches_aux.push_back(good_matches[i]);
            }
        }

        Mega_matches_aux.push_back(matches_aux);

    }

    tm.stop();
    double matchTime = tm.getTimeMilli();

    cout << "Build time: " << buildTime << " ms; Match time: " << matchTime << " ms" << endl;
    cout << ">" << endl << endl;

    //save machets into file
    cout << "saving matches into file...";
    FileStorage fs2("matches", FileStorage::WRITE);

    for (size_t i = 0; i<Mega_matches_aux.size(); i++){
        stringstream ss;
        ss << i;
        fs2 << "Matches " + ss.str();
        fs2 << Mega_matches_aux[i];
    }
    // explicit close
    fs2.release();
    cout << "done" << endl << endl;
}

void saveResultImages( vector<vector<DMatch> >& Mega_matches_aux, vector<Mat>& trainImages, vector<vector<KeyPoint> >& trainKeypoints,
                       vector<string>& trainImagesNames, string& resultDir )
{
    cout << "< Save results..." << endl;
    Mat drawImg;
    static Stats stat;
    int foto_id = 1;
    for( size_t i = 0; i < trainImages.size()-1; i++ )
    {
        if( !trainImages[i].empty() )
        {
            drawMatches(trainImages[i], trainKeypoints[i], trainImages[i+1], trainKeypoints[i+1],
                         Mega_matches_aux[i], drawImg, Scalar::all(-1), Scalar(0, 255, 255)); // 4
            stat.keypoints = trainKeypoints[i].size();
            stat.trainKeypoints = trainKeypoints[i+1].size();
            stat.matches = Mega_matches_aux[i].size();
            stat.id = foto_id;
            foto_id++;
            drawStatistics(drawImg, stat);
            string filename = resultDir + "/res_" + trainImagesNames[i];
            if( !imwrite( filename, drawImg ) )
                cout << "Image " << filename << " can not be saved (may be because directory " << resultDir << " does not exist)." << endl;
        }
    }
    cout << ">" << endl;
}

static void read_from_files(vector<vector<Vec3b> >& colors, vector<vector<KeyPoint> >& trainKeypoints, vector<vector<DMatch> >& Mega_matches){
    cout << endl << "extracting features from file..." << endl;
    cout << "Reading... " << endl;
    FileStorage fs3;
    fs3.open("keypoints_colors", FileStorage::READ);

    if (!fs3.isOpened())
    {
        cerr << "Failed to open " << "keypoints" << endl;
        return ;
    }

    int i = 0;
    while(true){
        vector<KeyPoint> Keypoints;
        vector<Vec3b> new_colors;
        stringstream ss;
        ss << i;
        fs3["keypoints " + ss.str()] >> Keypoints;
        if (Keypoints.size() != 0){
            trainKeypoints.push_back(Keypoints);
            Keypoints.clear();
        }
        else {
            i = 0;
            break;
        }
        fs3["colors " + ss.str()] >> new_colors;
        colors.push_back(new_colors);
        new_colors.clear();
        i++;

    }
    fs3.release();
    cout << "done reading" << endl;

    // READ MATCHES FILE
    cout << endl << "extracting matches from file..." << endl;
    cout << "Reading... " << endl;
    FileStorage fs4;
    fs4.open("matches", FileStorage::READ);

    if (!fs4.isOpened())
    {
        cerr << "Failed to open " << "matches" << endl;
        return ;
    }

    while(true){
        vector<DMatch> maches;
        stringstream ss;
        ss << i;
        fs4["Matches " + ss.str()] >> maches;
        if (maches.size() != 0){
            Mega_matches.push_back(maches);
            i++;
            maches.clear();
        }
        else
            break;
    }

    fs4.release();
    cout << "done reading" << endl;
}

#endif // MANY_IMAGES_MATCHING_H

