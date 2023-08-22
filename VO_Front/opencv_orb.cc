#include  <iostream>
#include <opencv2/core.hpp>
//特征点匹配提取等
#include <opencv2/features2d/features2d.hpp>
//人机交互窗口使用
#include <opencv2/highgui/highgui.hpp>
#include <chrono> 

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    if( argc != 3)
    {
        cout << "usage: ./opencv_orb image1 image2" << endl;
    }

    //加载图片
     //第二个参素决定图片的加载方式，此处为BGR模式（opencv中的颜色通道，注意不同于RGB）
    Mat image1 = imread(argv[1], IMREAD_COLOR);    
    Mat image2 = imread(argv[2], IMREAD_COLOR);
    if(image1.data == nullptr && image2.data == nullptr)
    {
        //cerr << "no image" << endl;
        return -1;
    }

    vector<KeyPoint> kp1, kp2;      //定义两个cv类型的关键点
    Mat descriptors1,descriptors2;      //定义两个描述子矩阵

    //初始化特征检测器、描述子提取器和描述子匹配器
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //记录此刻时间，为和t2一起计算用时
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    // FeatureDetector即该类本名的成员函数，用来检测图像中的特征点
    detector ->detect(image1, kp1);
    detector ->detect(image2, kp2);                                                     
    // 计算指定图像及其上的特征点的描述子
    descriptor ->compute(image1, kp1, descriptors1);
    descriptor ->compute(image2, kp2, descriptors2);
    
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "提取ORB耗时" << time_used.count() << "s." << endl;

    //输出ORB图
    Mat outimg1;
    drawKeypoints(image1, kp1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    namedWindow("ORB 特征", WINDOW_NORMAL);  
    imshow("ORB 特征", outimg1);

    // 匹配
    vector<DMatch> matches;  //定义类型，匹配索引
    t1 = chrono::steady_clock::now();
    matcher ->match(descriptors1, descriptors2,matches);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;

    // 自动判别类型，从matches中找到最大距离和最小距离
    auto min_max = minmax_element(matches.begin(), matches.end(),
                                [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    vector<DMatch> good_matches;
    //遍历所有匹配对,将距离大于max(2 * min_dist, 30.0)的删除
    for (int i = 0; i < descriptors1.rows; i++)  
    {
        if (matches[i].distance <= max(2 * min_dist, 30.0))
        {
        good_matches.push_back(matches[i]);
        }
    }

    Mat img_match;
    Mat img_goodmatch;
    //  Matches from the first image to the second one, which means that keypoints1[i] has a corresponding point in keypoints2[matches[i]] .
    drawMatches(image1, kp1, image2, kp2, matches, img_match);
    drawMatches(image1, kp1,  image2, kp2, good_matches, img_goodmatch);
    namedWindow("all matches", WINDOW_NORMAL);  
    namedWindow("good matches", WINDOW_NORMAL);
    imshow("all matches", img_match);
    imshow("good matches", img_goodmatch);
    waitKey(0);

    return 0;    

}