#include "Point.hpp"

void BasePoint::Point::initToZero() {
        x=0.0;
        y=0.0;
        z=0.0;
        center=-1;
}

double BasePoint::Point::distanceSquared(const BasePoint::Point& p1, const BasePoint::Point& p2) {
    double dist = 0.0;
    dist += (p1.x - p2.x) * (p1.x - p2.x);
    dist += (p1.y - p2.y) * (p1.y - p2.y);
    dist += (p1.z - p2.z) * (p1.z - p2.z);
    return dist;
}
