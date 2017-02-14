#include <iostream>
#include <opencv2/opencv.hpp>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <time.h>
#include <sstream>
#include <fstream>
#include <string>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define MAX_FRAME 1500
#define CLOCK_PER_SEC 1000
#define STAR_MAXSIZE 30
#define STAR_RESPONSE_TH 20
#define STAR_LINEBACKPROJ 10
#define STAR_LINEBACKBIN 8
#define STAR_NONMAXSUPP 3
#define MATCHING_TH 0.2

Ptr<StarDetector> star = StarDetector::create(STAR_MAXSIZE,STAR_RESPONSE_TH,STAR_LINEBACKPROJ,STAR_LINEBACKBIN,STAR_NONMAXSUPP);
Ptr<BRISK> brisk = BRISK::create();
Ptr<SIFT> sift = SIFT::create();
Ptr<SURF> surf = SURF::create(1500,4,3,false,true);
Ptr<AKAZE> akaze = AKAZE::create();

double focal = 718.856; //focal lenght
Point2d pp(607.1928,185.2157); //principle point

Mat img1, img2, img3;
Mat R_f = Mat::eye(3,3,CV_64F);
Mat t_f = Mat::zeros(3,1,CV_64F);
Mat K = (Mat_<double>(3,3) << 718.856, 0, 607.1928, 0, 718.856, 185.2157, 0, 0, 1);

clock_t time_star, begin;

char text[100];
int fontFace = FONT_HERSHEY_PLAIN;
double fontScale = 1;
int thickness = 1;
cv::Point textOrg(10, 50);
Mat traj = Mat::zeros(600, 600, CV_8UC3);

Mat proj_mat_1, proj_mat_2, proj_mat_3;

vector<KeyPoint> keypoint_1, keypoint_2, keypoint_3;
Mat descriptor_1, descriptor_2, descriptor_3;
vector<Point2f> point1, point2, match_1, match_2, match_3;
vector<DMatch> good_matches_2;

vector<Point3d> pnp_3d_point;
Mat pnp_4d_point; //homogenous coordinat

Mat R_pnp;
Mat R_vec = Mat::zeros(3,1,CV_64FC1);
Mat t_vec = Mat::zeros(3,1,CV_64FC1);
Mat mask1;
double _dc[] = {0,0,0,0};

Point3f getGroundTruth(int frame_id)
{
  string line;
  int i = 0;
  ifstream myfile ("/media/dikysepta/DATA/Final Project/Datasets/pose/poses/00.txt");
  double x =0, y=0, z = 0;
  if (myfile.is_open())
  {
    while (( getline (myfile,line) ) && (i<=frame_id))
    {
      std::istringstream in(line);
      for (int j=0; j<12; j++)  {
        in >> z ;
        if (j==7) y=z;
        if (j==3)  x=z;
      }
      i++;
    }
    myfile.close();
  }

  else {
    cout << "Unable to open file";
  }
  Point3f output(x,y,z);
  return output;
}

double getAbsoluteScale(int frame_id)
{
  string line;
  int i = 0;
  ifstream myfile ("/media/dikysepta/DATA/Final Project/Datasets/pose/poses/00.txt");
  double x =0, y=0, z = 0;
  double x_prev, y_prev, z_prev;
  if (myfile.is_open())
  {
    while (( getline (myfile,line) ) && (i<=frame_id))
    {
      z_prev = z;
      x_prev = x;
      y_prev = y;
      std::istringstream in(line);
      //cout << line << '\n';
      for (int j=0; j<12; j++)  {
        in >> z ;
        if (j==7) y=z;
        if (j==3)  x=z;
      }

      i++;
    }
    myfile.close();
  }

  else {
    cout << "Unable to open file";
    return 0;
  }

  return sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;
}

Mat captureImage(int frame)
{
    Size ukuran(640,480);
    Mat image, imresize;
    char filename[256];

    sprintf(filename, "/media/dikysepta/DATA/Final Project/Datasets/dataset/sequences/00/image_0/%06d.png", frame);

    imresize = imread(filename, IMREAD_GRAYSCALE);

    resize(imresize, image, ukuran);

    return imresize;
}

/*descriptor_1 is train image, descriptor_2 is query image*/
vector<DMatch> matchFeature(Mat descriptor_1, Mat descriptor_2)
{
    vector<DMatch> matches;//, good_matches;

    clock_t timing = clock();
    BFMatcher matcher(NORM_L2);
    cout << "matching init :" << (clock()-timing)/CLOCK_PER_SEC << endl;
    timing = clock();
    matcher.match(descriptor_2, descriptor_1, matches); //paling lama

    cout << "matching match :" << (clock()-timing)/CLOCK_PER_SEC << endl;
    timing = clock();
    double max_dist = 0; double min_dist = 10000;
    //-- Quick calculation of max and min distances between keypoints
    for(unsigned int i = 0; i < matches.size(); i++ )
    {
        double dist = matches.at(i).distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    cout << "matching min max :" << (clock()-timing)/CLOCK_PER_SEC << endl;
    timing = clock();
    int count = 0;
    int bound = matches.size();
    for( int i = 0; i < bound; i++ )
    {
        if( matches.at(count).distance > MATCHING_TH*(max_dist-min_dist) )
        {
            matches.erase(matches.begin() + count);
        }
        else
            count++;
    }
    cout << "matching remove :" << (clock()-timing)/CLOCK_PER_SEC << endl;

    return matches;
}

/*match 2 array of keypoint, output 2 array of 2d point*/
void matchTwoPoints(vector<DMatch> good_matches, vector<KeyPoint> keypoint_1, vector<KeyPoint> keypoint_2, vector<Point2f> &point_1, vector<Point2f> &point_2)
{
    KeyPoint::convert(keypoint_1, point_1);
    KeyPoint::convert(keypoint_2, point_2);

    vector<Point2f> temp_point_1, temp_point_2;
    for(unsigned int i = 0; i < good_matches.size(); i++)
    {
       temp_point_1.push_back(point_1.at(good_matches.at(i).trainIdx));
       temp_point_2.push_back(point_2.at(good_matches.at(i).queryIdx));
    }
    for(unsigned int i = 0; i < good_matches.size(); i++)
    {
        point_1.at(i).x = temp_point_1.at(i).x;
        point_1.at(i).y = temp_point_1.at(i).y;
        point_2.at(i).x = temp_point_2.at(i).x;
        point_2.at(i).y = temp_point_2.at(i).y;
    }
    point_1.erase(point_1.begin()+good_matches.size(), point_1.end());
    point_2.erase(point_2.begin()+good_matches.size(), point_2.end());
}

int main()
{
    /*Inisialisasi Mode*/
    img1 = captureImage(0);

    /*extract feature and compute descriptors*/
    star->detect(img1,keypoint_1);
    surf->compute(img1, keypoint_1, descriptor_1);

    for(int frame = 1; frame < MAX_FRAME; frame++)
    {
        cout << "Frame " << frame << endl;
        time_star = clock();
        /*Capture new frame*/
        img2 = captureImage(frame);
        cout << "capture image :" << (clock()-time_star)/CLOCK_PER_SEC << endl;
        time_star = clock();

        /*Detect keypoint and compute descriptors*/
        star->detect(img2,keypoint_2);
        cout << "Feature detection :" << (clock()-time_star)/CLOCK_PER_SEC << endl;
        time_star = clock();
        surf->compute(img2, keypoint_2, descriptor_2);
        cout << "descriptor :" << (clock()-time_star)/CLOCK_PER_SEC << endl;
        time_star = clock();

        /*Matching feature from two images*/
        vector<DMatch> good_matches_1 = matchFeature(descriptor_1, descriptor_2);
        cout << "matching :" << (clock()-time_star)/CLOCK_PER_SEC << endl;
        time_star = clock();

        /*Membuat vector point sesuai feature yang cocok agar bisa digunakan findessentialmat()*/
        matchTwoPoints(good_matches_1, keypoint_1, keypoint_2, point1, point2);
        cout << "Sorting :" << (clock()-time_star)/CLOCK_PER_SEC << endl;
        time_star = clock();

        /*Hitung Essential matrix using ransac*/
        Mat E = findEssentialMat(point1, point2, focal, pp, RANSAC, 0.9999, 1.0);
        cout << "Essential Mat :" << (clock()-time_star)/CLOCK_PER_SEC << endl;
        time_star = clock();

        /*Hitung extrinsic matrix*/
        Mat R, t, mask;
        recoverPose(E, point1, point2, R, t, focal, pp, mask);
        cout << "RecoverPose :" << (clock()-time_star)/CLOCK_PER_SEC << endl;

        Mat imgout;
        drawKeypoints(img2, keypoint_2, imgout, Scalar(0,255,255),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        imshow("matches", imgout);

        /*Hitung matrix rotasi dan translasi*/
        double scale = getAbsoluteScale(frame);
        if(frame == 1)
        {
            t_f = -scale*R.t()*t;
            R_f = R.t();
        }
        else
        {
            R_f = R.t()*R_f;
            t_f = t_f - scale*R_f*t;
        }

        //time_star = clock() - time_star;
        //Mat R_vec;
        //Rodrigues(R_f, R_vec);
        /*tampilkan variabel untuk diamati*/
        cout << "Frame " << frame << endl;
//        cout << "Key1:" << keypoint_1.size() << " Key2:" << keypoint_2.size();
//        cout << " Match1:" << good_matches_1.size();
//        cout << " Inlier:" << countNonZero(mask) << "  " << "time: " << (clock() - time_star)/CLOCK_PER_SEC;
//        cout << " X:" << t_f.at<double>(0,0) << " Y:" << t_f.at<double>(1,0) << " Z:" << t_f.at<double>(2,0) << endl;
        //cout << " Yaw:" << 180/3.14*R_vec.at<double>(0,0) << " Pitch:" << 180/3.14*R_vec.at<double>(1,0) << " Roll:" << 180/3.14*R_vec.at<double>(2,0) << endl;
        cout << " ----------------------------------------------------------------------------" << endl;

        namedWindow( "Trajectory", WINDOW_AUTOSIZE );// Create a window for display.

        Point3f pose_true = getGroundTruth(frame);
        int x_true = int(pose_true.x) + 300;
        int y_true = -int(pose_true.z) + 500;
        circle(traj, Point(x_true, y_true) ,1, CV_RGB(0,0,255), 2);

        int x = int(t_f.at<double>(0,0)) + 300;
        int y = -int(t_f.at<double>(2,0)) + 500;
        circle(traj, Point(x, y) ,1, CV_RGB(255,0,0), 2);

        rectangle(traj, Point(10, 30), Point(550, 50), CV_RGB(0,0,0), CV_FILLED);
        sprintf(text, "Koordinat: x = %.3fm y = %.3fm z = %.3fm", t_f.at<double>(0,0), t_f.at<double>(1,0), t_f.at<double>(2,0));
        putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
        imshow( "Trajectory", traj );

        /*update variable*/
        keypoint_1 = keypoint_2;
        descriptor_1 = descriptor_2.clone();


        waitKey(1);
    }
    imwrite("/media/dikysepta/DATA/Final Project/trajectory.png", traj);
    return 0;
}
