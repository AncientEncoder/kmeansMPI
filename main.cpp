#include <iostream>
#include <mpi.h>
#include "KMeans.hpp"

int main(int argc, char* argv[]) {
    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    KMeans::KMeans* kmeans = nullptr;
    if (rank == 0) {
        kmeans = new KMeans::KMeans(num_procs, argc, argv);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank != 0) {
        kmeans = new KMeans::KMeans(num_procs, argc, argv);
    }

    kmeans->run();

    if (rank == 0) {
        delete kmeans;
    }

    MPI_Finalize();
    return 0;
}