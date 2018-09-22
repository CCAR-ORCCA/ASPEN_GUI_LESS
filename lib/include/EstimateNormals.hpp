#ifndef HEADER_NORMALESTIMATOR
#define HEADER_NORMALESTIMATOR

#include <PointCloud.hpp>

template <class T> class EstimateNormals{

public:

	EstimateNormals(T & pc);
	
	void estimate_normals(int N_neighbors,const arma::vec::fixed<3> & los_dir);
	void estimate_normals(double radius_neighbors,const arma::vec::fixed<3> & los_dir);


protected:
	T & pc;
};



#endif