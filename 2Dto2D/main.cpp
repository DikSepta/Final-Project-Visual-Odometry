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
#include <cvsba.h>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define MAX_FRAME 15
#define CLOCK_PER_SEC 1000
#define STAR_MAXSIZE 30
#define STAR_RESPONSE_TH 20
#define STAR_LINEBACKPROJ 10
#define STAR_LINEBACKBIN 8
#define STAR_NONMAXSUPP 3
#define MATCHING_TH 0.2
#define MAX_POINT_THRESHOLD 500

Ptr<StarDetector> star = StarDetector::create(STAR_MAXSIZE,STAR_RESPONSE_TH,STAR_LINEBACKPROJ,STAR_LINEBACKBIN,STAR_NONMAXSUPP);
Ptr<BRISK> brisk = BRISK::create();
Ptr<SIFT> sift = SIFT::create();
Ptr<SURF> surf = SURF::create(1500,4,3,false,true);
Ptr<AKAZE> akaze = AKAZE::create();

double focal = 718.856; //focal lenght
Point2d pp(607.1928,185.2157); //principle point

Mat first_img, curr_img;
Mat R_w = Mat::eye(3,3,CV_64F), R_c = Mat::eye(3,3,CV_64F);
Mat t_w = Mat::zeros(3,1,CV_64F), t_c = Mat::zeros(3,1,CV_64F);
Mat K = (Mat_<double>(3,3) << 718.856, 0, 607.1928, 0, 718.856, 185.2157, 0, 0, 1);

Mat proj_mat_1 = (Mat_<double>(3,4) << 718.856, 0, 607.1928, 0, 0, 718.856, 185.2157, 0, 0, 0, 1, 0);
Mat proj_mat_2 = Mat::zeros(3,4, CV_32FC1);
Mat proj_mat_3 = Mat::zeros(3,4, CV_32FC1);

clock_t time_per_frame;

char text[100];
int fontFace = FONT_HERSHEY_PLAIN;
double fontScale = 1;
int thickness = 1;
cv::Point textOrg(10, 50);
Mat traj = Mat::zeros(600, 600, CV_8UC3);

vector<KeyPoint> prev_keypoint, curr_keypoint, keypoint_0;
Mat prev_descriptor, curr_descriptor;
vector<Point2f> point1, point2, match_1, match_2, match_3;
vector<DMatch> prev_good_matches;

vector<Point3d> point3d;
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

    BFMatcher matcher(NORM_L2);
    matcher.match(descriptor_2, descriptor_1, matches); //paling lama

    double max_dist = 0; double min_dist = 10000;
    //-- Quick calculation of max and min distances between keypoints
    for(unsigned int i = 0; i < matches.size(); i++ )
    {
        double dist = matches.at(i).distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
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

/*match 3 array of keypoint, output 3 array of point2d*/
void matchThreePoints(vector<DMatch> good_matches_1, vector<DMatch> good_matches_2,
                      vector<KeyPoint> keypoint_1, vector<KeyPoint> keypoint_2, vector<KeyPoint> keypoint_3,
                      vector<Point2f> &point_1, vector<Point2f> &point_2, vector<Point2f> &point_3)
{
    KeyPoint::convert(keypoint_1, point_1);
    KeyPoint::convert(keypoint_2, point_2);
    KeyPoint::convert(keypoint_3, point_3);
    unsigned int indeks = 0;

    for(unsigned int i = 0; i < good_matches_2.size(); i++)
    {
        for(unsigned int j = 0; j < good_matches_1.size(); j++)
        {
            if(good_matches_2.at(i).trainIdx == good_matches_1.at(j).queryIdx)
            {
                indeks++;
            }
        }
    }
    int temp_point_1[2][indeks], temp_point_2[2][indeks], temp_point_3[2][indeks];
    indeks = 0;
    for(unsigned int i = 0; i < good_matches_2.size(); i++)
    {
        for(unsigned int j = 0; j < good_matches_1.size(); j++)
        {
            if(good_matches_2.at(i).trainIdx == good_matches_1.at(j).queryIdx)
            {
                temp_point_1[0][indeks] = point_1.at(good_matches_1.at(j).trainIdx).x;
                temp_point_1[1][indeks] = point_1.at(good_matches_1.at(j).trainIdx).y;
                temp_point_2[0][indeks] = point_2.at(good_matches_2.at(i).trainIdx).x;
                temp_point_2[1][indeks] = point_2.at(good_matches_2.at(i).trainIdx).y;
                temp_point_3[0][indeks] = point_3.at(good_matches_2.at(i).queryIdx).x;
                temp_point_3[1][indeks] = point_3.at(good_matches_2.at(i).queryIdx).y;
                //cout << good_matches_1.at(j).distance << "  " << good_matches_2.at(i).distance << endl;
                indeks++;
                break;
            }
        }
    }

    for(unsigned int i = 0; i < indeks; i++)
    {
        point_1.at(i).x = temp_point_1[0][i];
        point_1.at(i).y = temp_point_1[1][i];
        point_2.at(i).x = temp_point_2[0][i];
        point_2.at(i).y = temp_point_2[1][i];
        point_3.at(i).x = temp_point_3[0][i];
        point_3.at(i).y = temp_point_3[1][i];
    }
    point_1.erase(point_1.begin()+indeks, point_1.end());
    point_2.erase(point_2.begin()+indeks, point_2.end());
    point_3.erase(point_3.begin()+indeks, point_3.end());
}

Mat createProjMat(Mat R, Mat t)
{
    Mat temp;
    hconcat(R,t, temp);
    //return K * temp;
    return temp;
}

/* From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997 */
Mat triangulate_Linear_LS(Mat mat_P_l, Mat mat_P_r, Mat warped_back_l, Mat warped_back_r)
{
    Mat A(4,3,CV_64FC1), b(4,1,CV_64FC1), X(3,1,CV_64FC1), X_homogeneous(4,1,CV_64FC1), W(1,1,CV_64FC1);
    W.at<double>(0,0) = 1.0;
    A.at<double>(0,0) = (warped_back_l.at<double>(0,0)/warped_back_l.at<double>(2,0))*mat_P_l.at<double>(2,0) - mat_P_l.at<double>(0,0);
    A.at<double>(0,1) = (warped_back_l.at<double>(0,0)/warped_back_l.at<double>(2,0))*mat_P_l.at<double>(2,1) - mat_P_l.at<double>(0,1);
    A.at<double>(0,2) = (warped_back_l.at<double>(0,0)/warped_back_l.at<double>(2,0))*mat_P_l.at<double>(2,2) - mat_P_l.at<double>(0,2);
    A.at<double>(1,0) = (warped_back_l.at<double>(1,0)/warped_back_l.at<double>(2,0))*mat_P_l.at<double>(2,0) - mat_P_l.at<double>(1,0);
    A.at<double>(1,1) = (warped_back_l.at<double>(1,0)/warped_back_l.at<double>(2,0))*mat_P_l.at<double>(2,1) - mat_P_l.at<double>(1,1);
    A.at<double>(1,2) = (warped_back_l.at<double>(1,0)/warped_back_l.at<double>(2,0))*mat_P_l.at<double>(2,2) - mat_P_l.at<double>(1,2);
    A.at<double>(2,0) = (warped_back_r.at<double>(0,0)/warped_back_r.at<double>(2,0))*mat_P_r.at<double>(2,0) - mat_P_r.at<double>(0,0);
    A.at<double>(2,1) = (warped_back_r.at<double>(0,0)/warped_back_r.at<double>(2,0))*mat_P_r.at<double>(2,1) - mat_P_r.at<double>(0,1);
    A.at<double>(2,2) = (warped_back_r.at<double>(0,0)/warped_back_r.at<double>(2,0))*mat_P_r.at<double>(2,2) - mat_P_r.at<double>(0,2);
    A.at<double>(3,0) = (warped_back_r.at<double>(1,0)/warped_back_r.at<double>(2,0))*mat_P_r.at<double>(2,0) - mat_P_r.at<double>(1,0);
    A.at<double>(3,1) = (warped_back_r.at<double>(1,0)/warped_back_r.at<double>(2,0))*mat_P_r.at<double>(2,1) - mat_P_r.at<double>(1,1);
    A.at<double>(3,2) = (warped_back_r.at<double>(1,0)/warped_back_r.at<double>(2,0))*mat_P_r.at<double>(2,2) - mat_P_r.at<double>(1,2);
    b.at<double>(0,0) = -((warped_back_l.at<double>(0,0)/warped_back_l.at<double>(2,0))*mat_P_l.at<double>(2,3) - mat_P_l.at<double>(0,3));
    b.at<double>(1,0) = -((warped_back_l.at<double>(1,0)/warped_back_l.at<double>(2,0))*mat_P_l.at<double>(2,3) - mat_P_l.at<double>(1,3));
    b.at<double>(2,0) = -((warped_back_r.at<double>(0,0)/warped_back_r.at<double>(2,0))*mat_P_r.at<double>(2,3) - mat_P_r.at<double>(0,3));
    b.at<double>(3,0) = -((warped_back_r.at<double>(1,0)/warped_back_r.at<double>(2,0))*mat_P_r.at<double>(2,3) - mat_P_r.at<double>(1,3));
    solve(A,b,X,DECOMP_SVD);
    vconcat(X,W,X_homogeneous);
    return X_homogeneous;
}

//Triagulate points
void TriangulatePoints(const vector<Point2f> pt_set1,
                       const vector<Point2f> pt_set2,
                       const Mat Kinv,
                       const Mat P,
                       const Mat P1,
                       vector<Point3d>& pointcloud)
{
    vector<Point2f> point_1_hom, point_2_hom;
    pointcloud.clear();

    unsigned int pts_size = pt_set1.size();
    for (unsigned int i=0; i < pts_size; i++)
    {
        Point2f kp = pt_set1.at(i);
        Point3d u(kp.x,kp.y,1.0);
        Mat u_nh = (Mat_<double>(3,1) << kp.x, kp.y, 1.0);
        Mat um = Kinv * u_nh;
        u.x = um.at<double>(0,0);
        u.y = um.at<double>(1,0);
        u.z = um.at<double>(2,0);
        Point2f kp1 = pt_set2.at(i);
        u_nh = (Mat_<double>(3,1) << kp1.x, kp1.y, 1.0);
        Point3d u1(kp1.x,kp1.y,1.0);
        Mat um1 = Kinv * u_nh;
        u1.x = um1.at<double>(0,0);
        u1.y = um1.at<double>(1,0);
        u1.z = um1.at<double>(2,0);

        Mat X = triangulate_Linear_LS(P, P1, Mat(u), Mat(u1));
        pointcloud.push_back(Point3d(X.at<double>(0,0)/X.at<double>(3,0),X.at<double>(1,0)/X.at<double>(3,0),X.at<double>(2,0)/X.at<double>(3,0)));
    }
}

double estimateScale(Mat P_1,Mat P_2,Mat P_3, vector<Point2f> matched_1, vector<Point2f> matched_2, vector<Point2f> matched_3)
{
    double scale;
    bool point_status = false;
    Mat point4d_1, point4d_2;
    double point3d_1[matched_1.size()][3], point3d_2[matched_1.size()][3];

    triangulatePoints(P_1, P_2, matched_1, matched_2, point4d_1);
    triangulatePoints(P_2, P_3, matched_2, matched_3, point4d_2);
    for(int i = 0; i < point4d_1.cols; i++)
    {
        point3d_1[i][0] = point4d_1.at<double>(0,i)/point4d_1.at<double>(3,i);
        point3d_1[i][1] = point4d_1.at<double>(1,i)/point4d_1.at<double>(3,i);
        point3d_1[i][2] = point4d_1.at<double>(2,i)/point4d_1.at<double>(3,i);
        point3d_2[i][0] = point4d_2.at<double>(0,i)/point4d_2.at<double>(3,i);
        point3d_2[i][1] = point4d_2.at<double>(1,i)/point4d_2.at<double>(3,i);
        point3d_2[i][2] = point4d_2.at<double>(2,i)/point4d_2.at<double>(3,i);
//        cout << "x1: " << point3d_1[i][0] <<  " y1: " << point3d_1[i][1] <<  " z1: " << point3d_1[i][2];
//        cout << " x2: " << point3d_2[i][0] <<  " y2: " << point3d_2[i][1] <<  " z2: " << point3d_2[i][2] << endl;
    }
    int indeks = 0, count = 0;
    bool get_first_point = false;
    int point_indeks[2];
    int max_point_th = MAX_POINT_THRESHOLD;
    //int min_point_th = MIN_POINT_THRESHOLD;

    do
    {
        /*kondisi untuk memilah titik yang layak untuk dijadikan acuan*/
        if((point3d_1[indeks][2] > 0 && point3d_2[indeks][2] > 0) && (point3d_1[indeks][2] < max_point_th && point3d_2[indeks][2] < max_point_th))
        {
            if(!get_first_point)
            {
                point_indeks[0] = indeks;
                get_first_point = true;
            }
            if(get_first_point)
            {
                point_indeks[1] = indeks;
                point_status = true;
            }
        }
        else
            indeks++;
        if(indeks == point4d_1.cols)
        {
            count++;
            indeks = 0;
            max_point_th += 20*count;
        }
    }while(point_status == false);
    scale = pow((point3d_1[point_indeks[0]][0] - point3d_1[point_indeks[1]][0]),2); //(X2-X1)^2
    scale += pow((point3d_1[point_indeks[0]][1] - point3d_1[point_indeks[1]][1]),2); //(Y2-Y1)^2
    scale += pow((point3d_1[point_indeks[0]][2] - point3d_1[point_indeks[1]][2]),2); //(Z2-Z1)^2
    scale = sqrt(scale);
    scale = scale / sqrt(pow((point3d_2[point_indeks[0]][0] - point3d_2[point_indeks[1]][0]),2) + pow((point3d_2[point_indeks[0]][1] - point3d_2[point_indeks[1]][1]),2) +pow((point3d_2[point_indeks[0]][2] - point3d_2[point_indeks[1]][2]),2));
    cout << scale << endl;
    return scale;
}

int main()
{
    /*Inisialisasi Mode*/
    first_img = captureImage(0);

    /*extract feature and compute descriptors*/
    star->detect(first_img,prev_keypoint);
    surf->compute(first_img, prev_keypoint, prev_descriptor);

    for(int frame = 1; frame < MAX_FRAME; frame++)
    {
        /*Start time elapsed*/
        time_per_frame = clock();
        /*Capture new frame*/
        curr_img = captureImage(frame);

        /*Detect keypoint and compute descriptors*/
        star->detect(curr_img, curr_keypoint);
        surf->compute(curr_img, curr_keypoint, curr_descriptor);

        /*Matching feature from two images*/
        vector<DMatch> good_matches = matchFeature(prev_descriptor, curr_descriptor);

        /*Membuat vector point sesuai feature yang cocok agar bisa digunakan findessentialmat()*/
        matchTwoPoints(good_matches, prev_keypoint, curr_keypoint, point1, point2);

        /*Hitung Essential matrix using ransac*/
        Mat E = findEssentialMat(point1, point2, focal, pp, RANSAC, 0.9999, 1.0);

        /*Hitung extrinsic matrix*/
        Mat R, t, mask;
        recoverPose(E, point1, point2, R, t, focal, pp, mask);

        Mat imgout;
        drawKeypoints(curr_img, curr_keypoint, imgout, Scalar(0,255,255),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        imshow("matches", imgout);

        /*Hitung matrix rotasi dan translasi*/
        /*Hitung scalenya*/
        double scale = getAbsoluteScale(frame);
        if(frame == 1)
        {
            t_w = -scale*R.t()*t;
            R_w = R.t();
            R_c = R;
            t_c = -R_w.t()*t_w;
        }
        else
        {
            R_w = R.t()*R_w;
            t_w = t_w - scale*R_w*t;
            R_c = R_w.t();
            t_c = -R_c*t_w;
        }

        /*tampilkan variabel untuk diamati*/
        Mat R_vec;
        Rodrigues(R_w, R_vec);
        cout << "Frame " << frame << endl;
        cout << "Key1:" << prev_keypoint.size() << " Key2:" << curr_keypoint.size();
        cout << " Match1:" << good_matches.size();
        cout << " Inlier:" << countNonZero(mask) << "  " << "time: " << (clock() - time_per_frame)/CLOCK_PER_SEC;
        cout << " X:" << t_w.at<double>(0,0) << " Y:" << t_w.at<double>(1,0) << " Z:" << t_w.at<double>(2,0) << endl;
        cout << " Yaw:" << 180/3.14*R_vec.at<double>(0,0) << " Pitch:" << 180/3.14*R_vec.at<double>(1,0) << " Roll:" << 180/3.14*R_vec.at<double>(2,0) << endl;
        cout << " ----------------------------------------------------------------------------" << endl;

        namedWindow( "Trajectory", WINDOW_AUTOSIZE );// Create a window for display.

        Point3f pose_true = getGroundTruth(frame);
        int x_true = int(pose_true.x) + 300;
        int y_true = -int(pose_true.z) + 500;
        circle(traj, Point(x_true, y_true) ,1, CV_RGB(0,0,255), 0);

        int x = int(t_w.at<double>(0,0)) + 300;
        int y = -int(t_w.at<double>(2,0)) + 500;
        circle(traj, Point(x, y) ,1, CV_RGB(255,0,0), 2);

        rectangle(traj, Point(10, 30), Point(550, 50), CV_RGB(0,0,0), CV_FILLED);
        sprintf(text, "Koordinat: x = %.3fm y = %.3fm z = %.3fm", t_w.at<double>(0,0), t_w.at<double>(1,0), t_w.at<double>(2,0));
        putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
        imshow( "Trajectory", traj );

        /*update variable*/
        prev_keypoint = curr_keypoint;
        prev_descriptor = curr_descriptor.clone();

        waitKey(1);
    }
    return 0;
}
