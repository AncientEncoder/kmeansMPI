#pragma once
#include "Point.hpp"
#include <iostream>
#include <vector>
#include <sstream>

namespace IOController{
    BasePoint::Point dataCutter(std::string str);

    class IOController{
    public:
        static std::vector<BasePoint::Point> fileReader(const std::string &fileName);
        static void fileWriter(const std::string&fileName,std::vector<std::vector<BasePoint::Point>>clusterData);
    };

}