#include <cstdlib>
#include <cmath>
#include <ctime>
#include <mpi.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "Point.hpp"
float **array(int m,int n);
void writeData(float **dataArray,int *inCluster,int N);
float **loadData(const std::string &fileName,int size);
float getDistance(float vector1[], float point2[], int n);
void cluster(int n,int k,int d,float **data,float **cluster_center,int *localInCluster);
float getDifference(int k, int n, int d, int *in_cluster, float **data, float **clusterCenter, float *sum);
void getCenter(int k,int d,int n,int *in_cluster,float **data,float **clusterCenter);
BasePoint::Point dataCutter(std::string str);

int  main(int argc,char *argv[]){
    int i,j,it;
    it=0;
    double start,end,epsilon;
    int loop=0;
    MPI_Status status;
    float temp1,temp2;
    int K,N,D;  //K ,data size
    float **data;  //image data
    int *allInCluster;  //root mark
    int *localInCluster;  //process mark
    int *inCluster;  //mark rcv
    int count=0;
    float *sumDiff;
    float *globalSumDiff;
    float **clusterCenter;  //center save
    int rank,size;


    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(!rank){
        K=std::atoi(argv[4]);
        D=3;
        N=std::atoi(argv[5]);
        epsilon=std::atoi(argv[2]);
        loop=std::atoi(argv[3]);
        data=loadData(argv[1],std::atoi(argv[5]));  //rd data
        if(size==1||size>N||N%(size-1)){
            std::cout<<"error N exit!";
            MPI_Abort(MPI_COMM_WORLD,1);  //exit while impossible to divide data
        }
    }
    MPI_Bcast(&K,1,MPI_INT,0,MPI_COMM_WORLD);  //BCast from root
    MPI_Bcast(&N,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&D,1,MPI_INT,0,MPI_COMM_WORLD);
    if(rank){
        data=array(N/(size-1),D);  //process malloc
    }
    allInCluster=(int *)malloc(N / (size - 1) * size * sizeof(int));  //root
    localInCluster=(int *)malloc(N / (size - 1) * sizeof(int));  //process
    inCluster=(int *)malloc(N * sizeof(int));  //process 0
    sumDiff=(float *)malloc(K * sizeof(float));  //calc distance
    globalSumDiff=(float *)malloc(K * sizeof(float));
    for(i=0;i<K;i++){
        sumDiff[i]=0.0;  //init
    }

    if(!rank){//give away data
        for(i=0;i<N;i+=(N/(size-1))){
            for(j=0;j<(N/(size-1));j++){
                MPI_Send(data[i+j],D,MPI_FLOAT,(i+j)/(N/(size-1))+1,99,MPI_COMM_WORLD);
            }
        }
    }else{  //recv data from root
        for(i=0;i<(N/(size-1));i++){
            MPI_Recv(data[i],D,MPI_FLOAT,0,99,MPI_COMM_WORLD,&status);
        }

    }
    MPI_Barrier(MPI_COMM_WORLD);  //sync
    start=MPI_Wtime();
    clusterCenter=array(K, D);  //center
    if(!rank){  //rand center
        srand((unsigned int)(time(NULL)));
        for(i=0;i<K;i++){
            for(j=0;j<D;j++){
                clusterCenter[i][j]=data[(int)((double)N * rand() / (RAND_MAX + 1.0))][j];
            }
        }

    }
    for(i=0;i<K;i++){
        MPI_Bcast(clusterCenter[i], D, MPI_FLOAT, 0, MPI_COMM_WORLD);  //BCast center
    }
    if(rank){
        cluster(N/(size-1), K, D, data, clusterCenter, localInCluster);  //process start it
        getDifference(K,N/(size-1), D, localInCluster, data, clusterCenter, sumDiff);
    }
    MPI_Gather(localInCluster, N / (size - 1), MPI_INT, allInCluster, N / (size - 1), MPI_INT, 0, MPI_COMM_WORLD);  //root rcv data
    MPI_Reduce(sumDiff, globalSumDiff, K, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);  //calc
    if(!rank){
        for(i=N/(size-1);i<N+N/(size-1);i++){
            inCluster[i - N / (size - 1)]=allInCluster[i];  //deal with marked data
        }
        temp1=0.0;
        for(i=0;i<K;i++) temp1+=globalSumDiff[i];
        count++;
    }
    MPI_Bcast(&temp1,1,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    do{   //main loop
        temp1=temp2;
        if(!rank){
            getCenter(K, D, N, inCluster, data, clusterCenter);  //update center
        }
        for(i=0;i<K;i++){
            MPI_Bcast(clusterCenter[i], D, MPI_FLOAT, 0, MPI_COMM_WORLD);  //BCast center
        }
        if(rank){
            cluster(N/(size-1), K, D, data, clusterCenter, localInCluster);
            for(i=0;i<K;i++){
                sumDiff[i]=0.0;
            }
            getDifference(K,N/(size-1), D, localInCluster, data, clusterCenter, sumDiff);
        }
        MPI_Gather(localInCluster, N / (size - 1), MPI_INT, allInCluster, N / (size - 1), MPI_INT, 0, MPI_COMM_WORLD);
        if(!rank){
            for(i=0;i<K;i++){
                globalSumDiff[i]=0.0;
            }
        }

        MPI_Reduce(sumDiff, globalSumDiff, K, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        if(!rank){
            for(i=N/(size-1);i<N+N/(size-1);i++){
                inCluster[i - N / (size - 1)]=allInCluster[i];
            }

            temp2=0.0;
            for(i=0;i<K;i++){
                temp2+=globalSumDiff[i];
            }
            count++;
        }
        MPI_Bcast(&temp2,1,MPI_FLOAT,0,MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        it++;
        //std::cout<<"Its "<<it<<" Times loop"<<std::endl;
    }while(fabs(temp2-temp1)>epsilon);
    end=MPI_Wtime();
    if(!rank)   {
    std::cout<<"|*------------------------------------------------------"<<std::endl;
    std::cout<<"|*Total time usage: "<<end-start<<" s"<<std::endl;
    std::cout<<"|*Average time usage: "<<(end-start)/it<<" s"<<std::endl;
    std::cout<<"|*Expected iteration is: "<<loop<<" times "<<"it should be in "<<(end-start)/it*loop<<" s"<<std::endl;

        writeData(data, inCluster, N);
    }
    MPI_Finalize();

}


//动态创建二维数组
float **array(int m,int n){
    int i;
    float **p;
    p=(float **)malloc(m*sizeof(float *));
    p[0]=(float *)malloc(m*n*sizeof(float));
    for(i=1;i<m;i++){
        p[i]=p[i-1]+n;
    }
    return p;
}

BasePoint::Point dataCutter(std::string str){
    BasePoint::Point point;
    std::istringstream ss(str);
    std::vector<std::string> words;
    std::string word;
    while(ss >> word) {
        words.push_back(word);
    }
//    for(std::string x : words) {
//        std::cout << x << std::endl;
//    }
    if (words.size()==4){
        point.x= std::stod(words[1]);
        point.y= std::stod(words[2]);
        point.z= std::stod(words[3]);
    }
    return point;
}
//从data.txt导入数据，要求首行格式：K=聚类数目,D=数据维度,N=数据量
float **loadData(const std::string &fileName,int size){
    int i=0;
    float **arrayData;
    arrayData=array(size, 3);  //生成数据数组
    std::ifstream getFile;
    getFile.open(fileName,std::ios::in);
    if(!getFile.is_open()){
        std::cout<<"Error to open file !!! "<<std::endl;
        std::cout<<"your file name is "<<fileName;
        exit(-1);
    }
    BasePoint::Point pt;
    std::string dataLine;
    while (std::getline(getFile,dataLine)&&i<size){
        if (dataLine[0]=='v'){
            pt=dataCutter(dataLine);
            arrayData[i][0]=pt.x;
            arrayData[i][1]=pt.y;
            arrayData[i][2]=pt.z;
            i++;
        }
    }
    return arrayData;
}

//计算欧几里得距离
float getDistance(float vector1[], float point2[], int n){
    int i;
    float sum=0.0;
    for(i=0;i<n;i++){
        sum+=pow(vector1[i] - point2[i], 2);
    }
    return sqrt(sum);
}

//把N个数据点聚类，标出每个点属于哪个聚类
void cluster(int n,int k,int d,float **data,float **cluster_center,int *localInCluster){
    int i,j;
    float min;
    float **distance=array(n,k);  //存放每个数据点到每个中心点的距离
    for(i=0;i<n;++i){
        min=9999.0;
        for(j=0;j<k;++j){
            distance[i][j] = getDistance(data[i],cluster_center[j],d);
            if(distance[i][j]<min){
                min=distance[i][j];
                localInCluster[i]=j;
            }
        }
    }
    free(distance);
}

//计算所有聚类的中心点与其数据点的距离之和
float getDifference(int k, int n, int d, int *in_cluster, float **data, float **clusterCenter, float *sum){
    int i,j;
    for(i=0;i<k;++i)
        for(j=0;j<n;++j)
            if(i==in_cluster[j])
                sum[i]+=getDistance(data[j], clusterCenter[i], d);
}

//计算每个聚类的中心点
void getCenter(int k,int d,int n,int *in_cluster,float **data,float **clusterCenter){
    float **sum=array(k,d);  //存放每个聚类中心
    int i,j,q,count;
    for(i=0;i<k;i++)
        for(j=0;j<d;j++)
            sum[i][j]=0.0;
    for(i=0;i<k;i++){
        count=0;  //统计属于某个聚类内的所有数据点
        for(j=0;j<n;j++){
            if(i==in_cluster[j]){
                for(q=0;q<d;q++)
                    sum[i][q]+=data[j][q];  //计算所属聚类的所有数据点的相应维数之和
                count++;
            }
        }
        for(q=0;q<d;q++)
            clusterCenter[i][q]= sum[i][q] / count;
    }

    free(sum);
}
void writeData(float **dataArray,int *inCluster,int N){
    std::ofstream fileWrite;
    fileWrite.open("result.txt",std::ios::out | std::ios::trunc);
    for (int k = 0; k < N; ++k) {
        fileWrite<<dataArray[k][0]<<" "<<dataArray[k][1]<<" "<<dataArray[k][2]<<" "<<inCluster[k]<<" "<<std::endl;
    }
    std::cout<<"=====Writing successfully!!====="<<std::endl;
}