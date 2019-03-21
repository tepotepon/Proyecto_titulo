#ifndef MANY_IMAGES_MATCHING_H
#define MANY_IMAGES_MATCHING_H

#include "stats.h"
#include "utils.h"
#include "view.h"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

const string defaultDescriptorType = "SIFT";
const string defaultMatcherType = "FlannBased";
const string defaultFileWithTrainImages = "/home/javier/Documents/cv_codes/build-many_images_matching-Desktop-Release/train/trainImages.txt";
const string defaultDirToSaveResImages = "/home/javier/Documents/cv_codes/build-many_images_matching-Desktop-Release/results";

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

static bool createDetectorDescriptorMatcher(const string& descriptorType, const string& matcherType,
                                      Ptr<Feature2D>& featureDetector,
                                      Ptr<DescriptorMatcher>& descriptorMatcher )
{
    cout << "< Creating feature detector, descriptor extractor and descriptor matcher ..." << endl;

    string Chosen_Type;
    cout << endl << "choose Descriptor Type: (SIFT/SURF) ";
    cin >> Chosen_Type;

    if (Chosen_Type == descriptorType)
        featureDetector = SIFT::create();
    else
        featureDetector = SURF::create();

    descriptorMatcher = DescriptorMatcher::create(matcherType);
    cout << ">" << endl;

    bool isCreated = !( featureDetector.empty() ||descriptorMatcher.empty() );
    if( !isCreated )
        cout << "Can not create feature detector or descriptor extractor or descriptor matcher of given types." << endl << ">" << endl;

    return isCreated;
}


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

//funcion debe ser modificada para leer pares consecutivos de imagenes.
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

static void detectKeypoints(const vector<Mat>& trainImages, vector<vector<KeyPoint> >& trainKeypoints,
                      Ptr<Feature2D>& featureDetector,vector<view> views)
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
    FileStorage fs("keypoints", FileStorage::WRITE);
    for (size_t i = 0; i<trainKeypoints.size(); i++){
        views[i] = trainKeypoints[i];
        stringstream ss;
        ss << i;
        fs << "keypoints " + ss.str();
        fs << trainKeypoints[i];
    }

    // explicit close
    fs.release();
    cout << "Done." << endl << endl;

}

void save_colors(const vector<Mat>& trainImages, vector<vector<Point2f> >& all_imgpts1)
{
    FileStorage fs("colors", FileStorage::WRITE);
    vector<Vec3b> colores;
    for(size_t i=0; i<trainImages.size()-1; i++){
        Mat img = trainImages[i];
        vector<Point2f> kps = all_imgpts1[i];
        colores.clear();
        for(size_t j=0; j<kps.size(); j++){
            Point2f point = kps[j];
            Vec3b bgr = img.at<Vec3b>(point.y,point.x);
            colores.push_back(bgr);
        }
        stringstream ss;
        ss << i;
        fs << "colors " + ss.str();
        fs << colores;
    }
    // explicit close
    fs.release();
    cout << "Don with colors." << endl << endl;
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

static void matchDescriptors( vector<vector<DMatch> >& Mega_matches_aux,const vector<Mat>& trainDescriptors,
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
    double max_dist = 0;
    double min_dist = 100;

    for (size_t i =0; i < trainDescriptors.size()-1; i++){
        vector<DMatch> matches_aux;
        vector<DMatch> good_matches;
        descriptorMatcher->match( trainDescriptors[i],trainDescriptors[i+1], matches_aux);
        cout << "-----------------------------------------------" << endl;

        for(int i = 0; i < trainDescriptors[i].rows; i++)
            {
                double dist = matches_aux[i].distance;
                if( dist < min_dist ) min_dist = dist;
                if( dist > max_dist ) max_dist = dist;
            }
        float prom = (max_dist + min_dist)/2.;
        //cout << "min dist: "<< min_dist << endl;
        //cout << "prom dist: " << prom << endl;
        for(size_t i = 0; i < matches_aux.size(); i++)
        {
            if(matches_aux[i].distance <= prom*0.66 )
                good_matches.push_back(matches_aux[i]);
        }
        cout << good_matches.size() << " good matches out of "<< matches_aux.size() << endl;
        Mega_matches_aux.push_back(good_matches);
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
    vector<char> mask;
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

static void read_from_files(vector<vector<KeyPoint> >& trainKeypoints,vector<view>& views,vector<vector<DMatch> >& Mega_matches){
    cout << endl << "extracting features from file..." << endl;
    cout << "Reading... " << endl;
    FileStorage fs3;
    fs3.open("keypoints", FileStorage::READ);

    if (!fs3.isOpened())
    {
        cerr << "Failed to open " << "keypoints" << endl;
        return ;
    }

    int i = 0;
    while(true){
        vector<KeyPoint> Keypoints;
        stringstream ss;
        ss << i;
        fs3["keypoints " + ss.str()] >> Keypoints;
        if (Keypoints.size() != 0){
            view vista(Keypoints);
            views.push_back(vista);
            trainKeypoints.push_back(Keypoints);
            i++;
            Keypoints.clear();
        }
        else {
            i = 0;
            break;
        }
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

static void read_from_colors(vector<Vec3b>& color){
    cout << endl << "extracting colors from file..." << endl;
    cout << "Reading... " << endl;
    FileStorage fs3;
    fs3.open("colors", FileStorage::READ);

    if (!fs3.isOpened())
    {
        cerr << "Failed to open " << "colors" << endl;
        return ;
    }

    int i = 0;

    while(true){
        vector<Vec3b> new_colors;
        stringstream ss;
        ss << i;
        fs3["colors " + ss.str()] >> new_colors;
        if (color.size() == 0)
            color = new_colors;
        if (new_colors.size() != 0){
            //view vista(color);
            //views.push_back(vista);
            color.insert( color.end(), new_colors.begin(), new_colors.end() );
            i++;
            new_colors.clear();
        }
        else {
            break;
        }
    }
    fs3.release();
    cout << "done reading colors" << endl;
}

#endif // MANY_IMAGES_MATCHING_H

