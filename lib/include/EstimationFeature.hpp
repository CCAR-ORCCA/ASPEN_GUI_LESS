#ifndef HEADER_ESTIMATIONFEATURE
#define HEADER_ESTIMATIONFEATURE

#include <PointCloud.hpp>
#include <PointDescriptor.hpp>

template <class T,class U> class EstimationFeature{

public:

	EstimationFeature(const T & pc);
	virtual void estimate(double radius_neighbors,int N_bins,U & output_pc) = 0; 


protected:
	const T & pc;
};


#endif