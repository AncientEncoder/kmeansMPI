#include <iostream>
#include <mpi.h>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>

static unsigned SEED = std::chrono::system_clock::now().time_since_epoch().count();
static std::mt19937 RAND_NUM(SEED);
std::string DIR_OBJECT;


typedef struct Point{
    double x=0.0;
    double y=0.0;
    double z=0.0;
    int center=-1;
}Point;


Point dataCutter(std::string str){
    Point point;
    std::istringstream ss(str);
    std::vector<std::string> words;
    std::string word;
    while(ss >> word) {
        words.push_back(word);
    }
    if (words.size()==4){
        point.x= std::stod(words[1]);
        point.y= std::stod(words[2]);
        point.z= std::stod(words[3]);
    }
    return point;
}



std::vector<Point>fileReader(std::string fileName){
    std::vector<Point> data;
    std::ifstream getFile;
    getFile.open(fileName,std::ios::in);
    if(!getFile.is_open()){
        std::cout<<"Error to open file !!! ";
        exit(-1);
    }
    std::string dataLine;
    while (std::getline(getFile,dataLine)){
        if (dataLine[0]=='v'){
            data.push_back(dataCutter(dataLine));
        }
    }

    return data;
}
void fileWriter(const std::string &fileName, std::vector<std::vector<Point>> clusterData) {
    std::ofstream fileWrite;
    fileWrite.open(fileName,std::ios::out | std::ios::trunc);
//    for (int i = 0; i < clusterData.size(); ++i) {
//        std::cout<<"cluster "<<i<<" has "<<clusterData[i].size()<<" datas"<<std::endl;
//    }
    for (int i = 0; i < clusterData.size(); ++i) {
        //std::cout<<"--------------------writing Center: "<<i<<"------------------------------"<<std::endl;
        //fileWrite<<"--------------------writing Center: "<<i<<"------------------------------"<<std::endl;
        for (const auto & j : clusterData[i]) {
            //fileWrite<<"x: "<<j.x<<" y: "<<j.y<<" z: "<<j.z<<" Center: "<<j.center<<std::endl;
            fileWrite<<j.x<<" "<<j.y<<" "<<j.z<<" "<<j.center<<" "<<std::endl;
        }
    }
    std::cout<<"=====Writing successfully!!====="<<std::endl;
}
double euclidean_distance(Point center, Point dot) {
    return sqrt((center.x - dot.x) * (center.x - dot.x) + (center.y - dot.y) * (center.y - dot.y) +
                (center.z - dot.z) * (center.z - dot.z));
}
void cluster(int K,std::vector<Point> &data,std::vector<Point>&cluster_center,std::vector<std::vector<Point>>&clusterData){
    int closestCenterId=0;
    double minDistance;
    double distance;
    for (int i = 0; i < data.size(); ++i) {
        minDistance= euclidean_distance(data[i],cluster_center[0]);
        for (int j = 0; j < K; ++j) {
            distance= euclidean_distance(data[i],cluster_center[j]);
            if (distance<minDistance){
                minDistance=distance;
                closestCenterId=j;
            }
        }
        data[i].center=closestCenterId;
    }
    for(const auto&point:data){
        clusterData[point.center].push_back(point);
    }
}




int main(int argc,char*argv[]){


    if (argc<5){
        exit(1);
    }
    DIR_OBJECT=argv[1];
    MPI_Status status;
    float temp1,temp2;
    std::vector<Point> data;//接收端数据
    std::vector<Point>totalData= fileReader(DIR_OBJECT);
    int K=std::atoi(argv[4]);
    int N=totalData.size();
    int D=3;  //聚类的数目，数据量，数据的维数
    std::vector<Point> *all_in_cluster;  //进程0标记每个点属于哪个聚类
    std::vector<Point> *local_in_cluster;  //其他进程标记每个点属于哪个聚类
    int *in_cluster;  //进程0标记每个点属于哪个聚类
    std::vector<std::vector<Point>>clusterData;//记录每一类的数据
    int count=0;
    float *sum_diff;
    float *global_sum_diff;
    std::vector<Point> cluster_center;  //存放每个聚类的中心点
    int rank,size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    std::vector<Point>pt;
    for (int i = 0; i < K; ++i) {
        clusterData.push_back(pt);
    }
    if (rank==0){
        int chunk_size=N/(size-1);//处理数据大小
        int offset=0;//数据偏移量
        for (int dest=1;dest<size;dest++){
            int start=offset+(dest-1)*chunk_size;
            int end=start+chunk_size;
            if (dest==size-1){
                end=N;
            }
            int count=end-start;
            MPI_Send(totalData.data()+start, count*sizeof(Point), MPI_BYTE, dest, 0, MPI_COMM_WORLD);
            // 更新偏移量
            offset = end;
        }
    } else{
        int chunk_size = N / (size - 1); // 处理数据大小
        int offset = (rank - 1) * chunk_size; // 数据偏移量
        int count = chunk_size;
        if (rank == size - 1) {
            count = N - (size - 2) * chunk_size;
        }
        data.resize(count); // 重新分配接收数据大小
        MPI_Recv(data.data(), count * sizeof(Point), MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
        }
    MPI_Barrier(MPI_COMM_WORLD);
    if (!rank){
        long areaLow=0;
        long areaUp=totalData.size()/K;
        long areaAdder=areaUp;
        for (int i = 0; i < K; ++i) {
            std::uniform_int_distribution<long long> dist(areaLow, areaUp);
            auto param=totalData[dist(RAND_NUM)];
            param.center=i;
            cluster_center.push_back(param);
        }
        areaLow+=areaAdder;
        areaUp+=areaAdder;
    }
    MPI_Bcast(cluster_center.data(), cluster_center.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (rank!=0){
        cluster(K,data,cluster_center,clusterData);
    }
    if (rank == 0) {
        // 计算每个进程发送的数据量和偏移量
        std::vector<int> sendcounts(size);
        std::vector<int> displs(size);
        int offset = 0;
        for (int i = 1; i < size; ++i) {
            int chunk_size = N / (size - 1);
            if (i == size - 1) {
                chunk_size = N - (size - 2) * chunk_size;
            }
            sendcounts[i] = chunk_size * sizeof(Point);
            displs[i] = offset * sizeof(Point);
            offset += chunk_size;
        }

        // 接收其他进程的数据
        totalData.resize(N);
        MPI_Gatherv(MPI_IN_PLACE, 0, MPI_BYTE,
                    totalData.data(), sendcounts.data(), displs.data(), MPI_BYTE,
                    0, MPI_COMM_WORLD);
    } else {
        int chunk_size = N / (size - 1); // 处理数据大小
        int offset = (rank - 1) * chunk_size; // 数据偏移量
        int count = chunk_size;
        if (rank == size - 1) {
            count = N - (size - 2) * chunk_size;
        }
        data.resize(count); // 重新分配接收数据大小
        MPI_Recv(data.data(), count * sizeof(Point), MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);

        // 发送数据给进程0
        MPI_Gatherv(data.data(), count * sizeof(Point), MPI_BYTE,
                    nullptr, nullptr, nullptr, MPI_BYTE,
                    0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
}
