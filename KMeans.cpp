#include "KMeans.hpp"
#include <random>
#include <omp.h>
#include <mpi.h>
#include <algorithm>
#include "IOController.hpp"

KMeans::KMeans::KMeans(int num_procs, int argc, char** argv){
    // 获取输入参数，包括数据集文件名、epsilon、最大迭代次数和K值
    const std::string file_name = argv[1];
    double epsilon = std::stod(argv[2]);
    int max_iterations = std::stoi(argv[3]);
    int k = std::stoi(argv[4]);

    // 读取数据集文件，将点存入points_向量中
    points_=IOController::IOController::fileReader(file_name);


    // 初始化KMeans对象的数据成员
    epsilon_ = epsilon;
    max_iterations_ = max_iterations;
    k_ = k;
    num_procs_ = num_procs;
    rank_ = -1;

    // 创建MPI类型
    MPI_Type_contiguous(4, MPI_DOUBLE, &MPI_POINT_);
    MPI_Type_commit(&MPI_POINT_);
}

KMeans::KMeans::~KMeans() {
    // 释放MPI类型
    MPI_Type_free(&MPI_POINT_);
}

void KMeans::KMeans::initialize() {
    // 获取当前进程的MPI通信器和进程编号
    comm_ = MPI_COMM_WORLD;
    MPI_Comm_rank(comm_, &rank_);

    // 获取数据集的总点数和特征数
    num_points_ = points_.size();
    num_features_ = 3;

    // 计算每个进程负责处理的数据点数
    int points_per_process = num_points_ / num_procs_;
    int leftover_points = num_points_ % num_procs_;

    // 分配每个进程需要处理的数据点的数量
    int* sendcounts = new int[num_procs_];
    int* displs = new int[num_procs_];
    for (int i = 0; i < num_procs_; i++) {
        sendcounts[i] = points_per_process * num_features_;
        if (leftover_points > 0) {
            sendcounts[i] += num_features_;
            leftover_points--;
        }
        displs[i] = i * points_per_process * num_features_;
    }

    // 创建每个进程的本地数据点向量
    std::vector<BasePoint::Point> local_points(sendcounts[rank_] / num_features_);

    // 将每个进程需要处理的数据点分发给各个进程
    MPI_Scatterv(points_.data(), sendcounts, displs, MPI_POINT_,
                 local_points.data(), sendcounts[rank_], MPI_POINT_, 0, comm_);

    // 计算初始质心
    if (rank_ == 0) {
        std::shuffle(points_.begin(), points_.end(), rand_num);
        centroids_.resize(k_);
        for (int i = 0; i < k_; i++) {
            centroids_[i] = points_[i];
        }
    }

    // 广播初始质心到所有进程
    MPI_Bcast(centroids_.data(), k_, MPI_POINT_, 0, comm_);

    // 初始化每个聚类的数量
    counts_.resize(k_);

    // 创建每个进程的本地簇数据向量
    cluster_data_.resize(k_);
    for (int i = 0; i < k_; i++) {
        cluster_data_[i] = std::vector<BasePoint::Point>();
    }

    // 释放内存
    delete[] sendcounts;
    delete[] displs;
}

void KMeans::KMeans::scatter() {
    // 每个进程分配到的点数
    int points_per_proc = num_points_ / num_procs_;
    // 每个进程需要接收的点数
    int recv_counts[num_procs_];
    // 每个进程需要接收的点的位移量
    int displs[num_procs_];
    for (int i = 0; i < num_procs_; i++) {
        recv_counts[i] = points_per_proc;
        if (i == num_procs_ - 1) { // 最后一个进程
            recv_counts[i] += num_points_ % num_procs_;
        }
        displs[i] = i * points_per_proc;
    }

    // 每个进程接收自己分配到的点
    std::vector<BasePoint::Point> recv_buffer(points_per_proc);
    MPI_Scatterv(points_.data(), recv_counts, displs, MPI_POINT_, recv_buffer.data(), recv_counts[rank_], MPI_POINT_, 0, comm_);

    // 将接收的点存储到cluster_data_数组中
    cluster_data_.resize(num_clusters_);
    for (int i = 0; i < num_clusters_; i++) {
        cluster_data_[i].resize(counts_[rank_]);
    }
    int cluster_index;
    for (int i = 0; i < points_per_proc; i++) {
        cluster_index = rand_num() % num_clusters_; // 随机选择一个簇
        recv_buffer[i].center = cluster_index;
        cluster_data_[cluster_index][i] = recv_buffer[i];
    }
}
void KMeans::KMeans::update() {
    // 计算新的质心
    std::vector<BasePoint::Point> new_centroids = computeCentroids();

    // 将新质心广播给所有进程
    MPI_Bcast(new_centroids.data(), num_clusters_, MPI_POINT_, 0, comm_);

    // 计算本地的变化量
    double local_delta = 0;
    for (int i = 0; i < num_clusters_; i++) {
        local_delta += euclidean_distance(new_centroids[i], centroids_[i]);
    }

    // 所有进程的变化量求和，得到全局的变化量
    double global_delta;
    MPI_Allreduce(&local_delta, &global_delta, 1, MPI_DOUBLE, MPI_SUM, comm_);

    // 如果变化量小于某个阈值，则认为已经收敛
    if (global_delta < epsilon_) {
        converged_ = true;
    }
    else {
        // 更新质心
        centroids_ = new_centroids;

        // 清空cluster_data_，准备下一次聚类
        for (int i = 0; i < num_clusters_; i++) {
            cluster_data_[i].clear();
        }
        // 将每个点划分到最近的质心所属的簇中
        int nearest_cluster;
        double nearest_distance;
        for (int i = 0; i < num_points_; i++) {
            nearest_cluster = -1;
            nearest_distance = std::numeric_limits<double>::max();
            for (int j = 0; j < num_clusters_; j++) {
                double distance = euclidean_distance(points_[i], centroids_[j]);
                if (distance < nearest_distance) {
                    nearest_distance = distance;
                    nearest_cluster = j;
                }
            }
            points_[i].center = nearest_cluster;
            cluster_data_[nearest_cluster].push_back(points_[i]);
        }
    }
}
