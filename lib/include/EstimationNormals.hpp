#ifndef HEADER_NORMALESTIMATOR
#define HEADER_NORMALESTIMATOR

#include <EstimationFeature.hpp>

template <class T,class U> class EstimationNormals : public EstimationFeature<T,U>{

public:

	EstimationNormals(const PointCloud<T> & input_pc,PointCloud<U> & output_pc);
	virtual	void estimate(double radius_neighbors);
	virtual	void estimate(int N_neighbors);

	void set_los_dir(const arma::vec::fixed<3> & los_dir);


protected:
	arma::vec::fixed<3> los_dir = {0,0,0};
	
};



#endif