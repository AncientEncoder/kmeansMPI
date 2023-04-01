#include "IOController.hpp"
#include "KMeans.hpp"
#include<mpi.h>
std::string DIR_OBJECT="'Ship.obj";
std::string DIR_WRITE="result.txt";
void PrintUsage(){
    std::cout<<"./main [file Name] [Epsilon] [max Iteration] [clusters (k)]"<<std::endl;
}
int main(int argc,char*argv[]){
    if (argc<5){
        PrintUsage();
        exit(1);
    }
    DIR_OBJECT=argv[1];
    IOController::IOController ioController;
    MPI_Init(&argc, &argv);
    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // only rank 0 creates KMeans object
    KMeans::KMeans* kmeans;
    if (rank == 0) {
        kmeans = new KMeans::KMeans(std::stod(argv[2]),std::atoi(argv[3]),std::atoi(argv[4]),argc,argv);
        kmeans->setData(ioController.fileReader(DIR_OBJECT));
        kmeans->KMeansRun();
        ioController.fileWriter(DIR_WRITE,kmeans->getClusterData());

    }
    delete kmeans;

    MPI_Finalize();
    //KMeans::KMeans kMeans(std::stod(argv[2]),std::atoi(argv[3]),std::atoi(argv[4]),argc,argv);

    return 0;
}