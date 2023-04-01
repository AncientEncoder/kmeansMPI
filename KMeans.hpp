#pragma once
#include <iostream>
#include <cmath>
#include "Point.hpp"
#include <vector>
#include <random>
#include <chrono>
#include <mpi.h>

namespace KMeans{
    //random seeds and devices
    static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    static std::mt19937 rand_num(seed);

    double euclidean_distance(BasePoint::Point center, BasePoint::Point dot);
    class KMeans{
    public:
        KMeans(int num_procs, int argc, char** argv);
        ~KMeans();
        void run();
    private:
        void initialize();
        void scatter();
        void update();
        void gather();
        bool hasConverged();
        std::vector<BasePoint::Point> computeCentroids();
        void printResult();
        int num_procs_;
        int rank_;
        int num_points_;
        int num_features_;
        int num_clusters_;
        int max_iterations_;
        bool converged_;
        std::vector<BasePoint::Point> points_;
        std::vector<BasePoint::Point> centroids_;
        std::vector<int> counts_;
        std::vector<std::vector<BasePoint::Point>> cluster_data_;
        MPI_Datatype MPI_POINT_;
        MPI_Datatype MPI_CLUSTER_;
        MPI_Comm comm_;
        char** argv_;
        double epsilon_;
        int k_;
    };
}