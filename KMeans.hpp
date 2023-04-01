#pragma once
#include <iostream>
#include <cmath>
#include "Point.hpp"
#include <vector>
#include <random>
#include <chrono>
namespace KMeans{
    //random seeds and devices
    static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    static std::mt19937 rand_num(seed);

    double euclidean_distance(BasePoint::Point center, BasePoint::Point dot);
    class KMeans{
    public:
        KMeans(double epsilon,int maxIterations,int clusters);
        void KMeansRun();
        void setData(const std::vector<BasePoint::Point> &pointsSet);
        const std::vector<std::vector<BasePoint::Point>> &getClusterData();
    private:
        void createClusters();
        bool hasCloseCenterBellowEpsilon(BasePoint::Point center, const std::vector<BasePoint::Point> &newCenters);
        std::vector<BasePoint::Point> updateCenter();
        void getRandomCenter();
        bool convergence(const std::vector<BasePoint::Point>&newCenters);
        int clusters;
        double epsilon;
        int maxIterations;
        std::vector<BasePoint::Point> center;
        std::vector<BasePoint::Point> points;
        std::vector<std::vector<BasePoint::Point>>clusterData;
    };
}