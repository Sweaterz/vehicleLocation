#include "matplotlibcpp.h"
#include <QCoreApplication>
#include <QApplication>
#include <QWidget>

#include <iostream>
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>
#include <thread>
#include <vector>
#include <string>
#include <mutex>
#include <math.h>
#include <fstream>
#include <sstream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <map>
using namespace Eigen;


#define MAX_SCAN_NUM 5000
#define BASEH -500
#define PROCESSIDX 28
namespace plt = matplotlibcpp;
std::vector<float> xData, yData;

void filterPoints(const std::vector<float>& xData, const std::vector<float>& yData, std::vector<float>& filteredX, std::vector<float>& filteredY, float threshold);
void filterEdgePoints(const std::vector<float>& xData, const std::vector<float>& yData, std::vector<float>& filteredX, std::vector<float>& filteredY, size_t removeFront, size_t removeBack);
float calculate_slope(float x1, float y1, float x2, float y2);
std::vector<int> find_corners(const std::vector<float>& x, const std::vector<float>& y);

template <typename T>
std::vector<size_t> sort_indexes(std::vector<T> &v)
{
    std::vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });//这样是升序，改成>就是降序了
    return idx;
}

std::string convertToHex(int &dec)
{
    std::string hex; // 十六进制数，用字符串类型接收，方便后续通过+运算符进行拼接
    int reminder[100]; // 余数数组
    int count = 0; // 记录余数个数
    int old_value = dec;
    int i = 0; // 余数数组下标
    std::string first_string = "0";

    if (dec < 0) {
        dec = abs(dec);
        hex += "-";
    } else if (dec == 0) {
        return "00";
    }

    // 初始化余数数组
    while (dec != 0) {
        reminder[i] = dec % 16;
        dec /= 16;
        i++;
        if (dec > 0) {
            count++;
        }
    }
    std::string str[100];
    for (int i = count; i >= 0; i--) {
        if (reminder[i] >= 10) {
            switch (reminder[i]) {
            case 10:
                str[i] = "A";
                break;
            case 11:
                str[i] = "B";
                break;
            case 12:
                str[i] = "C";
                break;
            case 13:
                str[i] = "D";
                break;
            case 14:
                str[i] = "E";
                break;
            case 15:
                str[i] = "F";
                break;
            }
        } else if (reminder[i] < 10) {
            str[i] = std::to_string(reminder[i]);
        }
        hex += str[i];
    }
    if (old_value < 16) {
        hex = first_string + hex;
    }
    return hex;
}

void myPlt(std::vector<float>x, std::vector<float>y, std::string title, int process=1)
{
    if(x.size() == 0 || y.size() == 0 || x.size() != y.size()){
        std::cerr<<"Failed to plt scan data..."<<std::endl;
        return;
    }

    //    plt::figure();

    plt::scatter(x, y);
    plt::title(title);
    plt::pause(2);

    plt::plot();
    if(process == 1){
        plt::show();
    }


}
void smoothData(std::vector<float>&x, std::vector<float>&y)
{
    std::vector<float> x1; std::vector<float> y1;
    for(int i = 2; i<x.size()-2; i++){
        float x_new = (x[i-2] + x[i-1] + x[i] + x[i+1] + x[i+2]) / 5.0;
        x1.push_back(x_new);
        float y_new = (y[i-2] + y[i-1] + y[i] + y[i+1] + y[i+2]) / 5.0;
        y1.push_back(y_new);
    }
    x.swap(x1); y.swap(y1);
    x1.clear(); y1.clear();
}
void processScanData(std::vector<float>&x, std::vector<float>&y);

void parseHorizontalScanData(std::vector<std::string> &hex_data, int &scan_idx)
{
    try {
        float intensity = 150.0;
        int hexSize = hex_data.size();
        if(hexSize != 1206 || hex_data[0] != "FF" || hex_data[1] != "EE"){return;}
        int firstStartAngle = int32_t(std::stoi(hex_data[3], 0, 16) * 0x100 + std::stoi(hex_data[2], 0, 16)) / 100;
        uint64_t timestamp = int32_t(std::stoi(hex_data[hexSize - 3], 0, 16)) * 256 * 256 * 256 + int32_t(std::stoi(hex_data[hexSize - 2], 0, 16)) * 256 * 256
                             + int32_t(std::stoi(hex_data[hexSize - 4], 0, 16)) * 256 + int32_t(std::stoi(hex_data[hexSize - 3], 0, 16));

        for (int i=0; i<12; i=i+1) {
            int startIdx = i * 100;
            if(hex_data[startIdx] != "FF" || hex_data[startIdx+1] != "EE"){continue;}
            int startAngle = int32_t(std::stoi(hex_data[startIdx+3], 0, 16) * 0x100 + std::stoi(hex_data[startIdx+2], 0, 16)) / 100;
            if(startAngle == 0){ xData.clear(); yData.clear();}
            for (int j=startIdx+4;j<startIdx+100;j=j+6) {
                float distance = int32_t(std::stoi(hex_data[j+1], 0, 16) * 0x100 + std::stoi(hex_data[j], 0, 16));
                if(distance <= 0){continue;}
                int idx = (j - startIdx - 4 + 6) / 6;
                int pointIdx = startIdx * 16 + idx;
                float angle = startAngle + 0.25 * idx;
                float iHorizontalAngle = 180;
                float angle_, h, l;
                //                if (angle < iHorizontalAngle) {
                //                    angle_ = iHorizontalAngle - angle;
                //                    h = uint(sin(angle_ * M_PI / 180) * distance);
                //                    l = uint(cos(abs(angle_ * M_PI / 180)) * distance);
                //                } else if (angle > iHorizontalAngle) {
                //                    angle_ = angle - iHorizontalAngle;
                //                    h = - uint(sin(angle_ * M_PI / 180) * distance);
                //                    l = uint(cos(abs(angle_ * M_PI / 180)) * distance);
                //                } else {
                //                    h = 0;
                //                    l = distance;
                //                }


                h = sin(angle * M_PI / 180) * distance;
                l = std::fabs(cos(angle * M_PI / 180) * distance);
                xData.push_back(h);
                yData.push_back(l);
            }

        }
        std::cout << "firstStartAngle:" << firstStartAngle << std::endl;
        if(firstStartAngle == 336){
            std::vector<float> x(xData.size());
            std::vector<float> y(xData.size());
            std::cout << "xData.size():" << xData.size() << " yData.size():" << yData.size() << std::endl;
            // 帧数据处理函数
            for(size_t i = 0; i < xData.size(); i++) {
                x[i] = xData[i];
                y[i] = yData[i];
            }

            //            if(scan_idx == PROCESSIDX){
            //                processScanData(x, y);
            //            }
            processScanData(x, y);


            xData.clear();
            yData.clear();
            scan_idx++;
        }
    } catch (const char* msg) {
        std::cerr<<"parseDG270 error: "<<msg<<std::endl;
    }

}

int readOfflineDatByBin(std::string& path){
    std::ifstream infile(path.c_str(), std::ios::in|std::ios::binary);
    if(!infile){
        std::cerr<<"Failed to open .bin file!"<<std::endl;
        return 0;
    }
    char data[1206];

    int scan_idx = 0;
    while(infile.read((char* )&data, sizeof (data))){
        std::vector<std::string> hex_str;
        scan_idx = scan_idx < MAX_SCAN_NUM ? scan_idx  : 0;
        for (auto s : data) {
            auto buf1 = s;
            int buf_data = static_cast<int>(buf1);
            if (buf_data < 0) {
                buf_data = 256 + buf_data;
            }
            std::string str_hex = convertToHex(buf_data);
            hex_str.push_back(str_hex);
        }
        if(hex_str.size() > 0){
            //            std::cout<<"scanData Size: "<<hex_str.size()<<std::endl;
            parseHorizontalScanData(hex_str, scan_idx);
        }
    }
    infile.close();
    std::this_thread::sleep_for(std::chrono::milliseconds(200000000000000000));
    return 1;
}

int readofflineDatByHex(std::string& path){
    std::ifstream infile(path.c_str(), std::ios::in);
    if(!infile){
        std::cerr<<"Failed to open .bin file!"<<std::endl;
        return 0;
    }
    char data[1206];

    int scan_idx = 0;
    std::string lineData;
    std::stringstream ss;
    while (getline(infile, lineData)) {
        std::vector<std::string> hex_str;
        ss.str(lineData);
        std::string single;
        while (getline(ss, single, ' ')) {
            hex_str.push_back(single.c_str());
        }
        ss.clear();
        scan_idx = scan_idx < MAX_SCAN_NUM ? scan_idx  : 0;
        if(hex_str.size() > 0){
            parseHorizontalScanData(hex_str, scan_idx);
        }
    }
    infile.close();
    std::this_thread::sleep_for(std::chrono::milliseconds(200000000000000000));
    return 1;
}


int getUDPData()
{
    // 创建socket
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if(-1==sockfd){
        return 0;}
    // 设置地址与端口
    sockaddr_in g_LocalAddr;
    socklen_t  addr_len = sizeof(g_LocalAddr);

    memset(&g_LocalAddr, 0, sizeof(g_LocalAddr));

    g_LocalAddr.sin_family = AF_INET;       // Use IPV4
    g_LocalAddr.sin_port = htons(2368);    //
    g_LocalAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    int sin_len = sizeof(g_LocalAddr);

    if (bind(sockfd, (struct sockaddr*)&g_LocalAddr, addr_len) == -1){
        printf("Failed to bind socket on port: 2368");
        close(sockfd);
        return 0;
    }

    sockaddr_in g_remoteAddr;
    bzero(&g_remoteAddr,sizeof(g_remoteAddr));

    int packageIdx = -1;
    std::vector<std::string> scan_data;
    int scan_idx = 0;
    int free_flag = 0;
    auto old_end = std::chrono::high_resolution_clock::now();
    while (true){
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        std::vector<std::string> hex_str;
        char           recData[1460];
        memset(recData, 0, sizeof(recData));//数组清零
        socklen_t nAddrLen = sizeof(struct sockaddr_in);
        int receiveSize = 0;
        receiveSize = recvfrom(sockfd, recData, 1460, MSG_DONTWAIT, (sockaddr *)&g_remoteAddr, &nAddrLen);
        //       std::cout << "receiveSize:" << receiveSize << std::endl;
        if (receiveSize > 6)
        {
            free_flag = 0;
            scan_idx = scan_idx < MAX_SCAN_NUM ? scan_idx  : 0;
            for (int i=0; i<receiveSize; i++) {
                auto buf1 = recData[i];
                int buf_data = static_cast<int>(buf1);
                if (buf_data < 0) {
                    buf_data = 256 + buf_data;
                }
                std::string str_hex = convertToHex(buf_data);
                hex_str.push_back(str_hex);
                //  std::cout << str_hex << std::endl;     // 打印十六进制数据
            }
            auto end = std::chrono::high_resolution_clock::now();
            float time = (float)std::chrono::duration_cast<std::chrono::microseconds>(end - old_end).count() / 1000;
            old_end = end;
            if(hex_str.size() > 0){
                std::cout<<"scanData Size: "<<hex_str.size()<<std::endl;
                parseHorizontalScanData(hex_str, scan_idx);
            }
        }
    }
    close(sockfd);
    return 1;
}

void splitByX(std::vector<float>&x, float xThreshold, std::vector<std::vector<float>>& dstX, int orderNum=1){

    std::vector<int>idxV;
    for (int j=1;j<x.size()-1;j++) {
        if(std::fabs(x[j] - x[j -1]) > xThreshold){
            idxV.push_back(j);
        }
    }
    if(idxV.size() == 0){
        dstX.push_back(x);
        return;
    }
    int last = 0;
    for (int j=0;j<idxV.size()+1;j++) {
        std::vector<float> newx;
        int startIdx = 0;
        int endIdx = x.size() - 1;
        if(j == 0){
            endIdx = idxV[j];
        }
        else if(j == idxV.size()){
            startIdx = idxV[j - 1];
            //           last = 1;
        }
        else {
            startIdx = idxV[j-1];
            endIdx = idxV[j];
        }
        std::cout<<j<<" startIdx: "<<startIdx<<" "<<x[startIdx]<<" endIdx: "<<endIdx<<" "<<x[endIdx]<<" x.size: "<<x.size()<<std::endl;
        newx.resize(endIdx - startIdx);
        std::copy(x.begin() + startIdx, x.begin() + endIdx, newx.begin());

        float maxX = *(std::max_element(std::begin(newx), std::end(newx)));
        float minX = *(std::min_element(std::begin(newx), std::end(newx)));
        if(orderNum == 1){
            if((maxX - minX) < 1100){
                newx.clear();
                continue;
            }
            std::vector<float>newy(newx.size(), BASEH+200.0);
            dstX.push_back(newx);
            myPlt(newx, newy, "Horizontal laser filtered-1 data");
        }
        else if (orderNum == 2) {
            if((maxX - minX) < 100 || (maxX - minX) > 1300){
                continue;
            }
            std::vector<float>newy(newx.size(), BASEH+400.0);
            dstX.push_back(newx);

            myPlt(newx, newy, "Horizontal laser filtered-2 data", last);
        }
        else if (orderNum == 3) {
            if((maxX - minX) < 100 || (maxX - minX) > 1300){
                continue;
            }
            std::vector<float>newy(newx.size(), BASEH+600.0);
            dstX.push_back(newx);

            myPlt(newx, newy, "Horizontal laser filtered-3 data", last);
        }

    }
}

//// 功能：计算一阶差分
std::vector<float> computeDifferences(const std::vector<float>& data) {
    std::vector<float> diffs;
    for (size_t i = 1; i < data.size(); ++i) {
        diffs.push_back(abs(data[i] - data[i - 1]));
    }
    return diffs;
}

// 功能：找到最大差分的索引
int findMaxDiffIndex(const std::vector<float>& diffs) {
    return distance(diffs.begin(), max_element(diffs.begin(), diffs.end()));
}

//// 功能：使用线性回归计算斜率
//double linearRegression(const std::vector<float>& x, const std::vector<float>& y) {
//    int n = x.size();
//    VectorXd x_vec = Map<const VectorXf, Unaligned>(x.data(), n);
//    VectorXd y_vec = Map<const VectorXf, Unaligned>(y.data(), n);

//    VectorXd ones = VectorXd::Ones(n);
//    MatrixXd X(n, 2);
//    X << ones, x_vec;

//    VectorXd beta = (X.transpose() * X).ldlt().solve(X.transpose() * y_vec);
//    return beta(1);  // 斜率
//}


void fitLineRansac(const std::vector<cv::Point2f>& points,
                   cv::Vec4f &line,
                   int iterations = 1000,
                   double sigma = 1.,
                   double k_min = -7.,
                   double k_max = 7.)
{
    unsigned int n = points.size();

    if(n<2)
    {
        return;
    }

    cv::RNG rng;
    double bestScore = -1.;
    for(int k=0; k<iterations; k++)
    {
        int i1=0, i2=0;
        while(i1==i2)
        {
            i1 = rng(n);
            i2 = rng(n);
        }
        const cv::Point2f& p1 = points[i1];
        const cv::Point2f& p2 = points[i2];

        cv::Point2f dp = p2-p1;//直线的方向向量
        dp *= 1./norm(dp);
        double score = 0;

        if(dp.y/dp.x<=k_max && dp.y/dp.x>=k_min )
        {
            for(int i=0; i<n; i++)
            {
                cv::Point2f v = points[i]-p1;
                double d = v.y*dp.x - v.x*dp.y;//向量a与b叉乘/向量b的摸.||b||=1./norm(dp)
                //score += exp(-0.5*d*d/(sigma*sigma));//误差定义方式的一种
                if( fabs(d)<sigma )
                    score += 1;
            }
        }
        if(score > bestScore)
        {
            line = cv::Vec4f(dp.x, dp.y, p1.x, p1.y);
            bestScore = score;
        }
    }
}

void processScanData(std::vector<float>&x, std::vector<float>&y){
    std::vector<float> filtered_x, filtered_y, filtered_x2, filtered_y2;
    smoothData(x, y);
    smoothData(x, y);
    smoothData(x, y);

    plt::clf();

    filterPoints(x, y, filtered_x, filtered_y, 10); // 过滤噪声点距离单位10mm以外的点
    // 过滤边缘点
    // filterEdgePoints(x, y, filtered_x2, filtered_y2, 20, 20);
    std::vector<float> differences = computeDifferences(y);
    int splitIndex = findMaxDiffIndex(differences) + 5;
    std::vector<double>x_, y_;
    double x_index = x[splitIndex];
    double y_index = y[splitIndex];
    x_.push_back(x_index);
    y_.push_back(y_index);


    std::vector<cv::Point2f> points1, points2;
    std::vector<float> x1, y1, x2, y2;
    for(int i=0; i<x.size(); i++){
        if(x[i] > 0 &&  y[i] < 500 && y[i] > 200){
            points1.push_back(cv::Point2f(y[i], x[i]));
            x1.push_back(y[i]);
            y1.push_back(x[i]);
        }
        else if ( x[i] > -500 && x[i] < 0) {
            points2.push_back(cv::Point2f(x[i], y[i]));
            x2.push_back(x[i]);
            y2.push_back(y[i]);
        }
    }
    {
        cv::Vec4f lineParam;
        fitLineRansac(points1,lineParam,2000,10);
        double k = lineParam[1] / lineParam[0];
        double b = lineParam[3] - k*lineParam[2];

        std::cout<<"ransac_p1: "<<k<<" "<<b<<" "<<atan(k) / 3.1415926 * 360<<std::endl;
    }
    {
        //        cv::Vec4f lineParam;
        //        cv::fitLine(points1,lineParam,cv::DIST_L1,0,0.01,0.01);
        //        double k = lineParam[1] / lineParam[0];
        //        double b = lineParam[3] - k*lineParam[2];
        //        std::cout<<k<<" "<<b<<" "<<atan(k) / 3.1415926 * 360<<std::endl;
        //        auto k = linearRegression(x1, y1);
        //        std::cout<<"my_p1: "<<k<<" "<<atan(k) / 3.1415926 * 360<<std::endl;


    }

    {
        cv::Vec4f lineParam;
        fitLineRansac(points2,lineParam,2000,10);
        double k = lineParam[1] / lineParam[0];
        double b = lineParam[3] - k*lineParam[2];

        std::cout<<"ransac_p2: "<<k<<" "<<b<<" "<<atan(k) / 3.1415926 * 360<<std::endl;
    }
    {
        //        cv::Vec4f lineParam;
        //        cv::fitLine(points2,lineParam,cv::DIST_L2,0,0.01,0.01);
        //        double k = lineParam[1] / lineParam[0];
        //        double b = lineParam[3] - k*lineParam[2];
        //        std::cout<<k<<" "<<b<<" "<<" "<<atan(k) / 3.1415926 * 360<<std::endl;
        //        auto k = linearRegression(x2, y2);
        //        std::cout<<"my_p2: "<<k<<" "<<atan(k) / 3.1415926 * 360<<std::endl;
    }

    //     myPlt(x, y, "Horizontal laser orignal data");



    plt::scatter(filtered_x, filtered_y, 1);
    std::vector<int>corners = find_corners(filtered_x, filtered_y);
    sort(corners.begin(), corners.end());
    reverse(corners.begin(), corners.end());
    std::vector<float>corners_x, corners_y;
    int cornerIndex = corners[0];
    // for(int corner : corners)
    // {
    //     corners_x.push_back(filtered_x[corner]);
    //     corners_y.push_back(filtered_y[corner]);
    // }
    corners_x.push_back(filtered_x[cornerIndex]);
    corners_y.push_back(filtered_y[cornerIndex]);
    std::vector<cv::Point2f> frontPoints, leftPoints;
    for(int i=0; i<filtered_x.size(); i++){
        if(i <= cornerIndex){
            leftPoints.push_back(cv::Point2f(filtered_x[i], filtered_y[i]));
        }
        else{
            frontPoints.push_back(cv::Point2f(filtered_x[i], filtered_y[i]));
        }
    }
    cv::Vec4f leftLineParam, frontLineParam;
    std::vector<float> leftLineX, leftLineY, frontLineX, frontLineY;

    fitLineRansac(leftPoints, leftLineParam, 2000, 10);
    double k = leftLineParam[1] / leftLineParam[0];
    double b = leftLineParam[3] - k*leftLineParam[2];
    double leftAngle = atan(k) / 3.1415926 * 180;
    leftLineX.push_back(filtered_x[0]);
    leftLineX.push_back(filtered_x[cornerIndex]);
    leftLineY.push_back(filtered_y[0]);
    leftLineY.push_back(filtered_y[cornerIndex]);
    std::cout<<"ransac_left: "<<k<<" "<<b<<" "<<leftAngle<<std::endl;
    fitLineRansac(frontPoints, frontLineParam, 2000, 10);
    k = frontLineParam[1] / frontLineParam[0];
    b = frontLineParam[3] - k*frontLineParam[2];
    double frontAngle = atan(k) / 3.1415926 * 180;
    frontLineX.push_back(0);
    frontLineX.push_back(filtered_x[cornerIndex]);
    frontLineY.push_back(b);
    frontLineY.push_back(k*filtered_x[cornerIndex]+b);
    std::cout<<"ransac_front: "<<k<<" "<<b<<" "<<frontAngle<<std::endl;

    std::cout<< "The difference between the left and front angles is: " <<fabs(leftAngle - frontAngle) << std::endl;

    plt::plot(leftLineX, leftLineY, {{"color", "orange"}});
    plt::scatter(corners_x, corners_y, 100, { {"color", "red"}, {"marker", "o"} }); // 画出角点
    plt::plot(frontLineX, frontLineY, {{"color", "yellow"}});
    // plt::scatter(corners_x, corners_y, 1000, {{"color", "green"}, {"marker", "o"}, {"alpha", "0.5"}});

    // plt::scatter(x_, y_, 100, {{"color", "red"}, {"marker", "o"}, {"linestyle", "--"}});
    // plt::pause(0.1);
    // plt::clf();
    // plt::scatter(filtered_x, filtered_y);
    // plt::pause(0.1);
    // plt::clf();
    // plt::scatter(filtered_x2, filtered_y2);
    // plt::plot();
    plt::title("x size: " + std::to_string(x.size()) + " y size: " + std::to_string(y.size()));
    plt::xlim(0, 800);
    plt::ylim(0, 800);
    plt::pause(0.3);

    //  plt::plot();
    //  plt::show();
    std::cout << "333333333333333333333333" << std::endl;
}

void filterPoints(const std::vector<float>& xData, const std::vector<float>& yData, std::vector<float>& filteredX, std::vector<float>& filteredY, float threshold = 10) {
    if (xData.size() != yData.size()) {
        throw std::invalid_argument("xData and yData must have the same size");
    }

    size_t n = xData.size();
    filteredX.clear();
    filteredY.clear();

    for (size_t i = 0; i < n; ++i) {
        float x = xData[i];
        float y = yData[i];

        // 计算邻域内的平均值
        float sumX = 0.0;
        float sumY = 0.0;
        int count = 0;

        for (size_t j = std::max(0, static_cast<int>(i) - 5); j <= std::min(n - 1, i + 5); ++j) {
            sumX += xData[j];
            sumY += yData[j];
            ++count;
        }

        float avgX = sumX / count;
        float avgY = sumY / count;

        // 计算当前点与平均值的距离
        float distance = std::sqrt((x - avgX) * (x - avgX) + (y - avgY) * (y - avgY));

        // 如果距离小于阈值，则保留该点
        if (distance < threshold) {
            filteredX.push_back(x);
            filteredY.push_back(y);
        }
        else {

            plt::scatter(std::vector<float>{x}, std::vector<float>{y}, 100);
            plt::xlim(0, 800);
            plt::ylim(0, 800);
            // plt::pause(0.1);
        }
    }
}

// 过滤边缘点
void filterEdgePoints(const std::vector<float>& xData, const std::vector<float>& yData, std::vector<float>& filteredX, std::vector<float>& filteredY, size_t removeFront = 20, size_t removeBack = 20) {
    if (xData.size() != yData.size()) {
        throw std::invalid_argument("xData and yData must have the same size");
    }

    size_t n = xData.size();
    if (n <= removeFront + removeBack) {
        throw std::invalid_argument("Not enough points to remove");
    }

    filteredX.clear();
    filteredY.clear();

    for (size_t i = removeFront; i < n - removeBack; ++i) {
        filteredX.push_back(xData[i]);
        filteredY.push_back(yData[i]);
    }
}


// 计算斜率
float calculate_slope(float x1, float y1, float x2, float y2) {
    if (x2 == x1) return std::numeric_limits<float>::infinity(); // 垂直线
    return (y2 - y1) / (x2 - x1);
}

// 计算两条直线之间的夹角（以度数为单位）
float calculate_angle(float slope1, float slope2) {
    if (slope1 == std::numeric_limits<float>::infinity()) slope1 = 1e9; // 处理垂直线
    if (slope2 == std::numeric_limits<float>::infinity()) slope2 = 1e9; // 处理垂直线
    float angle = std::abs(std::atan((slope1 - slope2) / (1 + slope1 * slope2)));
    return angle * 180 / M_PI; // 转换为度数
}

// 找到角点
std::vector<int> find_corners(const std::vector<float>& x, const std::vector<float>& y) {
    std::vector<int> corners;
    int step = 10; // 步长为10
    if (x.size() < 3 || y.size() < 3) return corners; // 至少需要三个点

    std::vector<float> slopes;
    for (size_t i = step; i < x.size(); i += step) { // 步长为step
        slopes.push_back(calculate_slope(x[i-step], y[i-step], x[i], y[i]));
        // std::cout << atan(slopes.back()) / 3.1415926 * 180 << " ";
    }

    // 计算斜率变化
    std::vector<float> slope_changes;
    for (size_t i = 1; i < slopes.size(); ++i) {
        slope_changes.push_back(calculate_angle(slopes[i-1], slopes[i]));
        std::cout << slope_changes.back() << " ";
    }

    // 找到斜率变化较大的点
    float threshold = 20; // 可以根据实际情况调整阈值
    for (size_t i = 0; i < slope_changes.size(); ++i) {
        // std::cout << slope_changes[i] << " ";
        if (slope_changes[i] > threshold) {
            corners.push_back((i + 1) * step); // 角点是斜率变化较大的点的下一个点
        }
    }
    std::cout << std::endl;

    return corners;
}

// 取角点附近的点，精确度更高
int getFromCorners(const std::vector<float>& xData, const std::vector<float>& yData, std::vector<float>& nearCornerX, std::vector<float>& nearCornerY, int cornerIndex)
{
    int nearParam = 50; // 50 is the parameter controling about getting points range.
    if(cornerIndex - nearParam < 0 || cornerIndex + nearParam >= xData.size())
    {
        std::cerr << "There is no enough points near the corner point. Now param is " << nearParam << ", please check it." << std::endl;
        return 0;
    }
    else
    {
        for(int i = cornerIndex-50; i < cornerIndex + 50; i++)
        {
            nearCornerX.push_back(xData[i]);
            nearCornerY.push_back(yData[i]);
        }
        return 1;
    }
}


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    std::string path = "/home/zhy/文档/270mini数据/20240416135731.dat";  //7/8/9  19 22 23
    //    int ret = readofflineDatByHex(path);
    getUDPData();

    return a.exec();
}
