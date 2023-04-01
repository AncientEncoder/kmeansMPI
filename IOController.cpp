#include "IOController.hpp"
#include <fstream>


BasePoint::Point IOController::dataCutter(std::string str){
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

std::vector<BasePoint::Point> IOController::IOController::fileReader(const std::string &fileName) {
    std::vector<BasePoint::Point> data;
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

void IOController::IOController::fileWriter(const std::string &fileName, std::vector<std::vector<BasePoint::Point>> clusterData) {
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
