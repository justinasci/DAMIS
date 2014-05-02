///////////////////////////////////////////////////////////
//  DimReductionMethod.cpp
//  Implementation of the Class DimReductionMethod
//  Created on:      07-Lie-2013 20:07:29
//  Original author: Povilas
///////////////////////////////////////////////////////////

/*! \class DimReductionMethod
    \brief A class of methods and attributes for initialization of projection matrix.
 */

#include "DimReductionMethod.h"
#include "DistanceMetrics.h"
#include "AdditionalMethods.h"
#include "Statistics.h"
#include <cmath>


DimReductionMethod::DimReductionMethod(){
    d = 2;
}

DimReductionMethod::~DimReductionMethod(){

}

int DimReductionMethod::getProjectionDimension(){
	return d;
}

void DimReductionMethod::initializeProjectionMatrix(){
    int n = X.getObjectCount();
    Y = ObjectMatrix(n);
    std::vector<double> DataObjectItem;
    DataObjectItem.reserve(0);//push_back(0.0);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < d; j++)
        {
            DataObjectItem.push_back(Statistics::getRandom(-0.1, 0.1, (i + j * 5)));
        }
        Y.addObject(DataObject(DataObjectItem));
        DataObjectItem.clear();
    }
}

void DimReductionMethod::setProjectionDimension(int dimension){
    d = dimension;
}

double DimReductionMethod::getStress()
{
    int m = X.getObjectCount(), n = Y.getObjectAt(0).getFeatureCount(),i, j, z;
    double stress = 0.0, s, distY, distX, diff;
    int noOfBytes = sizeof(double); //for file reading
    std::vector<double> ob1, ob2;

    FILE *distFile;
    distFile = fopen(AdditionalMethods::tempFileSavePath.c_str(), "rb");

// TODO (Povilas#1#): Check if file exists

    for (i = 0; i < m - 1; i++)
    {
        ob1 = Y.getObjectAt(i).getFeatures();
        for (j = i + 1; j < m; j++)
        {
            ob2 = Y.getObjectAt(j).getFeatures();
            s = 0.0;

            for (z = 0; z < n; z++)
            {
                diff = ob1.at(z) - ob2.at(z);
                s += diff * diff;
            }
            distY = std::sqrt(s);

            fread(&distX, noOfBytes, 1, distFile);

            stress += std::pow((distX - distY), 2);
        }
    }
    fclose(distFile);

    return stress * 1. / X.getWeight();
}

ObjectMatrix DimReductionMethod::getProjection(){
    return Y;
}