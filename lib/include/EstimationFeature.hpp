#ifndef HEADER_ESTIMATIONFEATURE
#define HEADER_ESTIMATIONFEATURE

#include <PointCloud.hpp>
#include <PointDescriptor.hpp>

template <class T,class U> class EstimationFeature{

public:

	EstimationFeature(const PointCloud<T> & input_pc,PointCloud<U> & output_pc);
	virtual void estimate(double radius_neighbors,bool force_use_previous = false) = 0; 
	virtual	void estimate(int N_neighbors,bool force_use_previous = false) = 0;


	static arma::vec compute_distances_to_center(const arma::vec & center , const PointCloud<U> & pc);
	arma::vec compute_distances_to_center();

	static void disable_common_features(double const & beta, const arma::vec & distances,PointCloud<U> & pc);
	void disable_common_features(double const & beta, const arma::vec & distances);

	void compute_center();
	static arma::vec compute_center(const PointCloud<U> & pc);
	static arma::mat compute_principal_axes(const PointCloud<U> & pc,const arma::vec & center);


	void prune(double deadband);


protected:
	const PointCloud<T> & input_pc;
	PointCloud<U> & output_pc;
	arma::vec center;

};


#endif