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
    MPI_Init(&argc,&argv);
    DIR_OBJECT=argv[1];
    IOController::IOController ioController;
    KMeans::KMeans kMeans(std::stod(argv[2]),std::atoi(argv[3]),std::atoi(argv[4]));
    kMeans.setData(ioController.fileReader(DIR_OBJECT));
    kMeans.KMeansRun();
    ioController.fileWriter(DIR_WRITE,kMeans.getClusterData());
    MPI_Finalize();
    return 0;
}