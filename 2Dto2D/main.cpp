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

#define MAX_FRAME 150
#define CLOCK_PER_SEC 1000
#define STAR_MAXSIZE 30
#define STAR_RESPONSE_TH 20
#define STAR_LINEBACKPROJ 10
#define STAR_LINEBACKBIN 8
#define STAR_NONMAXSUPP 3
#define MATCHING_TH 0.35
#define MAX_POINT_THRESHOLD 500
#define BA_FRAME 5
#define MAX_3D_POINT 5

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

vector<vector<DMatch>> match;
vector<Mat> R_BA, t_BA;
vector<Point3d> object_point;
vector<vector<KeyPoint>> feature_point;
vector<vector<int>> visibility;
vector<vector<Point2d>> point_correspondence;
vector<Mat> camera_matrix;
vector<Mat> dist_coeff;

double _dc[] = {0,0,0,0,0};

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
void TriangulatePoints(const vector<Point2d> pt_set1,
                       const vector<Point2d> pt_set2,
                       const Mat Kinv,
                       const Mat P,
                       const Mat P1,
                       vector<Point3d>& pointcloud)
{
    vector<Point2d> point_1_hom, point_2_hom;
    pointcloud.clear();

    unsigned int pts_size = pt_set1.size();
    for (unsigned int i=0; i < pts_size; i++)
    {
        Point2d kp = pt_set1.at(i);
        Point3d u(kp.x,kp.y,1.0);
        Mat u_nh = (Mat_<double>(3,1) << kp.x, kp.y, 1.0);
        Mat um = Kinv * u_nh;
        u.x = um.at<double>(0,0);
        u.y = um.at<double>(1,0);
        u.z = um.at<double>(2,0);
        Point2d kp1 = pt_set2.at(i);
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

void bundleInit()
{
    match.resize(BA_FRAME - 1);

    visibility.resize(BA_FRAME);
    for(int i = 0; i < BA_FRAME; i++)
    {
        visibility.at(i).resize(MAX_3D_POINT);
    }
    for(int i = 0; i < BA_FRAME; i++)
    {
        for(int j = 0; j < MAX_3D_POINT; j++)
            visibility.at(i).at(j) = 1;
    }

    R_BA.resize(BA_FRAME);
    t_BA.resize(BA_FRAME);
    feature_point.resize(BA_FRAME);

    for(int i = 0; i < BA_FRAME; i++)
    {
        R_BA.at(i) = Mat::eye(3,3,CV_64F);
        t_BA.at(i) = Mat::zeros(3,1,CV_64F);
    }

    camera_matrix.resize(BA_FRAME);
    for(int i = 0; i < BA_FRAME; i++)
    {
        camera_matrix.at(i) = K.clone();
    }

    dist_coeff.resize(BA_FRAME);
    for(int i = 0; i < BA_FRAME; i++)
    {
        dist_coeff.at(i) = Mat(1,4,CV_64FC1,_dc);
    }
}

void matchpoints(vector<vector<DMatch>> match_id, vector<vector<KeyPoint>> key_point, vector<vector<Point2d>> &output_point, int point_size)
{
    int num = match_id.size();
    for(int i = 0; i < num - 1; i++)
    {
        vector<DMatch> curr_match = match_id.at(i);
        vector<DMatch> next_match = match_id.at(i+1);
        vector<DMatch> temp_curr_match, temp_next_match;

        for(unsigned int j = 0; j < next_match.size(); j++)
        {
            for(unsigned int k = 0; k < curr_match.size(); k++)
            {
                if(next_match.at(j).trainIdx == curr_match.at(k).queryIdx)
                {
                    temp_curr_match.push_back(curr_match.at(k));
                    temp_next_match.push_back(next_match.at(j));
                    break;
                }
            }
            match_id.at(i) = temp_curr_match;
            match_id.at(i+1) = temp_next_match;
        }
    }
    for(int i = num - 1; i > 0; i--)
    {
        vector<DMatch> curr_match = match_id.at(i);
        vector<DMatch> prev_match = match_id.at(i-1);
        vector<DMatch> temp_prev_match;

        for(unsigned int j = 0; j < curr_match.size(); j++)
        {
            for(unsigned int k = 0; k < prev_match.size(); k++)
            {
                if(curr_match.at(j).trainIdx == prev_match.at(k).queryIdx)
                {
                    temp_prev_match.push_back(prev_match.at(k));
                    break;
                }
            }
            match_id.at(i-1) = temp_prev_match;
        }
    }

    int num_key = key_point.size();
    vector<vector<Point2f>> point;

    output_point.resize(num_key);
    point.resize(num_key);

    for(int i = 0; i < num_key; i++)
    {
        vector<Point2d> temp_point;
        KeyPoint::convert(key_point.at(i), point.at(i));
        vector<Point2f> curr_point = point.at(i);

        if(i < num_key - 1)
        {
            vector<DMatch> curr_match = match_id.at(i);
            for(unsigned int j = 0; j < curr_match.size(); j++)
            {
                temp_point.push_back((Point2d)curr_point.at(curr_match.at(j).trainIdx));
            }
        }
        else
        {
            vector<DMatch> curr_match = match_id.at(i - 1);
            for(unsigned int j = 0; j < curr_match.size(); j++)
            {
                temp_point.push_back((Point2d)curr_point.at(curr_match.at(j).queryIdx));
            }
        }
        output_point.at(i) = temp_point;
        if((int)output_point.at(i).size() > point_size)
        {
            output_point.at(i).erase(output_point.at(i).begin() + point_size, output_point.at(i).end());
        }
    }
    cout << "Match ID" << endl;
    cout << "1: " << match_id.at(0).size();
    cout << " 2: " << match_id.at(1).size();
    cout << " 3: " << match_id.at(2).size();
    cout << " 4: " << match_id.at(3).size() << endl;
    cout << "Output point" << endl;
    cout << "1: " << output_point.at(0).size();
    cout << " 2: " << output_point.at(1).size();
    cout << " 3: " << output_point.at(2).size();
    cout << " 4: " << output_point.at(3).size();
    cout << " 5: " << output_point.at(4).size() << endl;
}

/* inlier posisinya seseuai dengan train_point, ukurane harus sama */
void removeOutlier(vector<DMatch> &match_id, vector<KeyPoint> train_keypoint, vector<Point2f> train_point, Mat inlier)
{
    vector<Point2f> raw_train_point;
    vector<DMatch> temp_match;
    vector<Point2f> inlier_point;
    int pts_size = train_point.size();

    KeyPoint::convert(train_keypoint, raw_train_point);
//    cout << match_id.size() << " " << inlier.rows << " " << train_point.size() << " " << raw_train_point.size() << endl;

    for(int i = 0; i < pts_size; i++)
    {
        if(inlier.at<unsigned char>(i,0))
            inlier_point.push_back(train_point.at(i));
    }
//    cout << inlier_point.size() << endl;
    for(unsigned int j = 0; j < inlier_point.size(); j++)
    {
        for(int i = 0; i < pts_size; i++)
        {
            if(raw_train_point.at(match_id.at(i).trainIdx) == inlier_point.at(j))
            {
                temp_match.push_back(match_id.at(i));
                break;
            }
        }
    }
//    cout << temp_match.size() << endl;
    match_id = temp_match;
}

int main()
{
    /*Inisialisasi Mode*/
    bundleInit();
    first_img = captureImage(0);

    /*extract feature and compute descriptors*/
    star->detect(first_img,prev_keypoint);
    surf->compute(first_img, prev_keypoint, prev_descriptor);
    feature_point.at(0) = prev_keypoint;

    for(int frame = 1; frame < MAX_FRAME; frame++)
    {
        /*Start counting elapsed time*/
        time_per_frame = clock();
        /*Capture new frame*/
        curr_img = captureImage(frame);

        /*Detect keypoint and compute descriptors*/
        star->detect(curr_img, curr_keypoint);
        surf->compute(curr_img, curr_keypoint, curr_descriptor);
        /*store the keypoint*/
        if(frame < BA_FRAME)
        {
            feature_point.at(frame) = curr_keypoint;
        }
        else
        {
            for(int i = 0; i < BA_FRAME - 1; i++)
            {
                feature_point.at(i) = feature_point.at(i+1);
            }
            feature_point.at(BA_FRAME - 1) = curr_keypoint;
        }

        /*Matching feature from two images*/
        vector<DMatch> good_matches = matchFeature(prev_descriptor, curr_descriptor);

        /*Membuat vector point sesuai feature yang cocok agar bisa digunakan findessentialmat()*/
        matchTwoPoints(good_matches, prev_keypoint, curr_keypoint, point1, point2);

        /*Hitung Essential matrix using ransac*/
        Mat E = findEssentialMat(point1, point2, focal, pp, RANSAC, 0.9999, 1.0);

        /*Hitung extrinsic matrix*/
        Mat R, t, inlier;
        recoverPose(E, point1, point2, R, t, focal, pp, inlier);

        removeOutlier(good_matches, prev_keypoint, point1, inlier);

        Mat imgout;
        drawKeypoints(curr_img, curr_keypoint, imgout, Scalar(0,255,255),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        imshow("matches", imgout);

        /*store the matched feature*/
        if(frame < BA_FRAME)
        {
            match.at(frame - 1) = good_matches;
        }
        else
        {
            for(int i = 0; i < BA_FRAME - 2; i++)
            {
                match.at(i) = match.at(i+1);
            }
            match.at(BA_FRAME - 2) = good_matches;
        }

        if(frame >= BA_FRAME)
        {
            matchpoints(match, feature_point, point_correspondence, MAX_3D_POINT);
        }
        /*Update matrix rotasi dan translasi*/
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

        /*store R and t vector to be used by Bundle Adjustment*/
        if(frame < BA_FRAME)
        {
            Rodrigues(R_c, R_BA.at(frame));
            t_BA.at(frame) = t_c.clone();
        }
        else
        {
            for(int i = 0; i < BA_FRAME - 1; i++)
            {
                R_BA.at(i) = R_BA.at(i+1).clone();
                t_BA.at(i) = t_BA.at(i+1).clone();
            }
            Rodrigues(R_c, R_BA.at(BA_FRAME - 1));
            t_BA.at(BA_FRAME - 1) = t_c.clone();
        }

        Mat R_tr, t_tr;

        Rodrigues(R_BA.at(1), R_tr);
        t_tr = t_BA.at(1).clone();
        proj_mat_1 = createProjMat(R_tr, t_tr);

        Rodrigues(R_BA.at(2), R_tr);
        t_tr = t_BA.at(2).clone();
        proj_mat_2 = createProjMat(R_tr, t_tr);

        if(frame >= BA_FRAME)
        {
            TriangulatePoints(point_correspondence.at(0), point_correspondence.at(1), K.inv(), proj_mat_1, proj_mat_2, object_point);
        }
//        cout << "R1 : " << R_BA.at(0) << endl << " t1 : " << t_BA.at(0) << endl;
//        cout << " R2 : " << R_BA.at(1) << endl << " t2 : " << t_BA.at(1)<< endl;
//        cout << " R3 : " << R_BA.at(2) << endl <<  " t3 : " << t_BA.at(2)<< endl;
//        cout << " R4 : " << R_BA.at(3) << endl <<  " t4 : " << t_BA.at(3)<< endl;
//        cout << " R5 : " << R_BA.at(4) << endl <<  " t5 : " << t_BA.at(4) << endl;

        /*tampilkan variabel untuk diamati*/
        Mat R_vec;
        Rodrigues(R_w, R_vec);
        cout << "Frame " << frame << endl;
        cout << "Key1:" << prev_keypoint.size() << " Key2:" << curr_keypoint.size();
        cout << " Match1:" << good_matches.size();
        cout << " Inlier:" << countNonZero(inlier) << "  " << "time: " << (clock() - time_per_frame)/CLOCK_PER_SEC;
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
