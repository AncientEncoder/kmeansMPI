#include "KMeans.hpp"
#include <random>
#include <mpi.h>
#include <algorithm>
#include "IOController.hpp"

double KMeans::euclidean_distance(BasePoint::Point center, BasePoint::Point dot){
    return sqrt((center.x-dot.x)*(center.x-dot.x)+(center.y-dot.y)*(center.y-dot.y)+(center.z-dot.z)*(center.z-dot.z));
}
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
    size_=points_.size();
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
    // 计算全局平均值
    std::vector<BasePoint::Point> local_centroids(num_clusters_);
    std::vector<int> local_counts(num_clusters_, 0);
    for (int i = 0; i < points_.size(); i++) {
        int cluster_index = points_[i].center;
        local_centroids[cluster_index].x += points_[i].x;
        local_centroids[cluster_index].y += points_[i].y;
        local_centroids[cluster_index].z += points_[i].z;
        local_counts[cluster_index]++;
    }

    // 全局聚类中心的缓冲区
    std::vector<double> centroids_buffer(num_clusters_ * num_features_);
    for (int i = 0; i < num_clusters_; i++) {
        // 每个簇的点数不能为0
        if (local_counts[i] != 0) {
            local_centroids[i].x /= local_counts[i];
            local_centroids[i].y /= local_counts[i];
            local_centroids[i].z /= local_counts[i];
        }
        // 将每个簇的聚类中心拷贝到缓冲区中
        centroids_buffer[i * num_features_] = local_centroids[i].x;
        centroids_buffer[i * num_features_ + 1] = local_centroids[i].y;
        centroids_buffer[i * num_features_ + 2] = local_centroids[i].z;
    }

    // 将各进程的局部聚类中心汇总到rank=0的进程中
    MPI_Reduce(centroids_buffer.data(), NULL, num_clusters_ * num_features_, MPI_DOUBLE, MPI_SUM, 0, comm_);

    // 将全局更新后的聚类中心分发给各个进程
    if (rank_ == 0) {
        std::vector<BasePoint::Point> centroids(num_clusters_);
        int index = 0;
        for (int i = 0; i < num_clusters_; i++) {
            centroids[i].x = centroids_buffer[index++];
            centroids[i].y = centroids_buffer[index++];
            centroids[i].z = centroids_buffer[index++];
        }
        MPI_Bcast(centroids.data(), num_clusters_, MPI_POINT_, 0, comm_);
        centroids_ = centroids;
    } else {
        MPI_Bcast(centroids_.data(), num_clusters_, MPI_POINT_, 0, comm_);
    }

    // 将所有点的中心重新计算
    for (int i = 0; i < points_.size(); i++) {
        double min_distance = std::numeric_limits<double>::max();
        int min_index = -1;
        for (int j = 0; j < num_clusters_; j++) {
            double distance = euclidean_distance(points_[i], centroids_[j]);
            if (distance < min_distance) {
                min_distance = distance;
                min_index = j;
            }
        }
        points_[i].center = min_index;
    }
}
void KMeans::KMeans::gather() {
    // 计算每个进程需要发送的点数
    int num_local_points = points_.size();
    std::vector<int> send_counts(size_, 0);
    MPI_Allgather(&num_local_points, 1, MPI_INT, send_counts.data(), 1, MPI_INT, comm_);

    // 计算每个进程需要接收的点数
    std::vector<int> recv_counts(size_);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm_);

    // 计算发送和接收的点的起始位置
    std::vector<int> send_displs(size_);
    std::vector<int> recv_displs(size_);
    int send_displ = 0, recv_displ = 0;
    for (int i = 0; i < size_; i++) {
        send_displs[i] = send_displ;
        recv_displs[i] = recv_displ;
        send_displ += send_counts[i];
        recv_displ += recv_counts[i];
    }

    // 将所有点收集到rank=0的进程中
    std::vector<BasePoint::Point> recv_buffer(recv_displ);
    MPI_Alltoallv(points_.data(), send_counts.data(), send_displs.data(), MPI_POINT_, recv_buffer.data(),
                  recv_counts.data(), recv_displs.data(), MPI_POINT_, comm_);

    // 将收集到的点分发到各个进程中
    points_.swap(recv_buffer);
}

bool KMeans::KMeans::hasConverged() {
    // 计算本地误差平方和
    double local_sse = 0.0;
    for (int i = 0; i < points_.size(); i++) {
        local_sse += points_[i].distanceSquared(centroids_[points_[i].center]);
    }

    // 将各进程的本地误差平方和汇总到rank=0的进程中
    double global_sse;
    MPI_Reduce(&local_sse, &global_sse, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);

    // 判断是否收敛
    if (rank_ == 0) {
        double threshold = epsilon_ * epsilon_ * num_points_;
        return global_sse <= threshold;
    }
    return false;


}
