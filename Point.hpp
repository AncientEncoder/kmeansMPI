#pragma once

namespace BasePoint{
    class Point{
    public:
        double x;
        double y;
        double z;
        int center=-1;
        void initToZero();

        double distanceSquared(const BasePoint::Point& p1, const BasePoint::Point& p2);
    };

}