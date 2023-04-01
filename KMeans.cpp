#include "KMeans.hpp"
#include <random>
#include <omp.h>
#include <mpi.h>
double KMeans::euclidean_distance(BasePoint::Point center, BasePoint::Point dot){
    return sqrt((center.x-dot.x)*(center.x-dot.x)+(center.y-dot.y)*(center.y-dot.y)+(center.z-dot.z)*(center.z-dot.z));
}
KMeans::KMeans::KMeans(double epsilon, int maxIterations, int clusters,int argc,char* argv[]) {
    this->epsilon=epsilon;
    this->clusters=clusters;
    this->maxIterations=maxIterations;
    this->argc=argc;
    this->argv=argv;
    std::vector<BasePoint::Point> pt;
    for (int i = 0; i < clusters; ++i) {
        clusterData.push_back(pt);
    }
}

void KMeans::KMeans::getRandomCenter() {
    center.clear();
    //----partly random center---------------------------------------------------//
    long areaLower=0;
    long areaUpper=points.size()/clusters;
    long areaAdder=areaUpper;

    for (int i = 0; i < clusters; ++i) {
        std::uniform_int_distribution<long long> dist(areaLower, areaUpper);
        auto param=points[dist(rand_num)];
        param.center=i;
        center.push_back(param);
        areaLower+=areaAdder;
        areaUpper+=areaAdder;
    }
    //-------------------full random center--------------

    //    std::uniform_int_distribution<long long> dist(0, points.size());
    //    for (int i = 0; i < clusters; ++i) {
    //        auto param=points[dist(rand_num)];
    //        param.center=i;
    //        center.push_back(param);
    //    }
}

void KMeans::KMeans::createClusters() {
    // Define variables
    int closedCenterID;
    double minDistance;
    double distance;
    MPI_Init(&argc,&argv);
    // Get the rank and size of the MPI communicator
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide the points equally among processes
    int points_per_proc = points.size() / size;
    int remainder = points.size() % size;
    int start, end;
    if (rank < remainder) {
        start = rank * (points_per_proc + 1);
        end = start + points_per_proc;
    } else {
        start = rank * points_per_proc + remainder;
        end = start + points_per_proc - 1;
    }

    // Define arrays for storing the closest center ID and distance for each point
    int *closedCenterIDs = new int[points_per_proc];
    double *minDistances = new double[points_per_proc];

    // Loop through points assigned to this process
    for (int i = start; i <= end; i++) {
        closedCenterID = 0;
        minDistance = euclidean_distance(points[i], center[0]);

        // Find the closest center
        for (int j = 1; j < clusters; j++) {
            distance = euclidean_distance(points[i], center[j]);
            if (distance < minDistance) {
                minDistance = distance;
                closedCenterID = j;
            }
        }

        // Store the closest center ID and distance
        closedCenterIDs[i - start] = closedCenterID;
        minDistances[i - start] = minDistance;
    }

    // Gather the closest center IDs and distances from all processes
    int *globalClosedCenterIDs = new int[points.size()];
    double *globalMinDistances = new double[points.size()];
    MPI_Allgather(closedCenterIDs, points_per_proc, MPI_INT, globalClosedCenterIDs, points_per_proc, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(minDistances, points_per_proc, MPI_DOUBLE, globalMinDistances, points_per_proc, MPI_DOUBLE, MPI_COMM_WORLD);

    // Update the points' centers with the closest center ID
    for (int i = 0; i < points.size(); i++) {
        if (globalMinDistances[i] < minDistances[i - start] || globalClosedCenterIDs[i] == -1) {
            minDistances[i - start] = globalMinDistances[i];
            closedCenterIDs[i - start] = globalClosedCenterIDs[i];
        }
        points[i].center = closedCenterIDs[i - start];
    }

    // Free memory
    delete[] closedCenterIDs;
    delete[] minDistances;
    delete[] globalClosedCenterIDs;
    delete[] globalMinDistances;

    // Clear cluster data
    for (int i = 0; i < clusters; i++) {
        clusterData[i].clear();
    }

    // Assign points to their corresponding clusters
    for (const auto& point : points) {
        clusterData[point.center].push_back(point);
    }
    MPI_Finalize();
}

std::vector<BasePoint::Point> KMeans::KMeans::updateCenter() {
    std::vector<BasePoint::Point> newCenter;
    BasePoint::Point paramPoint;
    for (int i = 0; i < clusterData.size(); ++i) {
        paramPoint.initToZero();
//#pragma omp parallel for shared(paramPoint,i) default(none)
        for (int j = 0; j < clusterData[i].size(); ++j) {
            paramPoint.x+=clusterData[i][j].x;
            paramPoint.y+=clusterData[i][j].y;
            paramPoint.z+=clusterData[i][j].z;
            paramPoint.center=clusterData[i][j].center;
        }
        paramPoint.x=paramPoint.x/double(clusterData[i].size());
        paramPoint.y=paramPoint.y/double(clusterData[i].size());
        paramPoint.z=paramPoint.z/double(clusterData[i].size());
        newCenter.push_back(paramPoint);
    }
    return newCenter;
}

bool KMeans::KMeans::hasCloseCenterBellowEpsilon(BasePoint::Point center, const std::vector<BasePoint::Point> &newCenters){
    bool belowEpsilon= false;
    for(const auto&newCenter:newCenters){
        if (euclidean_distance(center,newCenter)<=epsilon){
            belowEpsilon= true;
        }
    }
    return belowEpsilon;
}

bool KMeans::KMeans::convergence(const std::vector<BasePoint::Point> &newCenters) {
    bool conv= true;
//#pragma omp parallel for  shared(newCenters,conv) default(none)
    for (int i = 0; i < center.size(); ++i) {
        if(!hasCloseCenterBellowEpsilon(center[i], newCenters)){
            conv= false;
        }
    }
    if (conv){
        std::cout<<"---------successfully converged---------------"<<std::endl;
    }
    return conv;
}
void KMeans::KMeans::setData(const std::vector<BasePoint::Point> &pointsSet) {
    this->points=pointsSet;
}
void KMeans::KMeans::KMeansRun() {
    double start,end,average,fStart,fStop;
    average=0.0;
    getRandomCenter();
    createClusters();
    std::cout<<"=> Data size: ["<<points.size()<<"]"<< std::endl;
    for (int i = 0; i < maxIterations; ++i) {
        start=MPI_Wtime();
        auto newCenters=updateCenter();
        if (convergence(newCenters)){
            break;
        }
        //std::cout<<"Iteration: "<<i<<" th Center update"<<std::endl;
        center.clear();
        center=newCenters;
        for (auto &clusterEle:clusterData){
            clusterEle.clear();
        }
        createClusters();
        end=MPI_Wtime()-start;
        //std::cout<<"Iteration time (s):"<<end<<std::endl;
        if (i==0){
            average=end;
        } else{
            average=(average+end)/2;
        }
    }
    fStop=MPI_Wtime()-fStart;
    std::cout<<"--------------------------------------------------"<<std::endl;
    std::cout<<"*  Iteration end average time (s): "<<average<<"  "<<std::endl;
    std::cout<<"*  Iteration total time (s): "<<fStop<<"          "<<std::endl;
    std::cout<<"--------------------------------------------------"<<std::endl;

}

const std::vector<std::vector<BasePoint::Point>> &KMeans::KMeans::getClusterData() {
    return clusterData;
}



