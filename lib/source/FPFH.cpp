#include "FPFH.hpp"
#include "PointNormal.hpp"
#include <armadillo>


FPFH::FPFH(const int & query_point,
	const std::vector<std::vector<int> > & points,
	const std::vector<SPFH> & spfhs,
	const PointCloud<PointNormal> & pc,
	bool scale_distance,
	bool is_valid_feature) : PointDescriptor(){

	this -> type = 1;
	this -> is_valid_feature = is_valid_feature;
	this -> histogram = spfhs[query_point].get_histogram();
	int N_bins = static_cast<int>(this -> histogram.size()/3);

	

	double distance_to_closest_neighbor;
	if (scale_distance){
		distance_to_closest_neighbor = spfhs[query_point].get_distance_to_closest_neighbor();
	}
	else{
		distance_to_closest_neighbor = 1.;
	}
	
	for (unsigned int k = 0 ; k < points[query_point].size(); ++k){
		double distance = arma::norm(pc.get_point_coordinates(query_point)- pc.get_point_coordinates(points[query_point][k]));
		
		if (distance>0){
			this -> histogram += distance_to_closest_neighbor * spfhs[points[query_point][k]]. get_histogram() / ( distance * points[query_point].size());
		}

	}




	// The uncorrelated histograms are normalized
	if (arma::max(this -> histogram) > 0){
		this -> histogram.subvec(0, N_bins - 1) *= 100./arma::sum(this -> histogram.subvec(0, N_bins - 1));
		this -> histogram.subvec(N_bins,2 * N_bins - 1) *= 100./arma::sum(this -> histogram.subvec(N_bins,2 * N_bins - 1));
		this -> histogram.subvec(2 * N_bins,3 * N_bins - 1) *= 100./arma::sum(this -> histogram.subvec(2 * N_bins,3 * N_bins - 1));
	}
	else{
		this -> is_valid_feature = false;
	}


	if (this -> histogram.has_nan()){
		throw(std::runtime_error("FPFH for point " + std::to_string(query_point) 
			+ " has nans. Query point had " + std::to_string(points[query_point].size()) + " neighbors"));
	}
	

}


FPFH::FPFH() : PointDescriptor(){}


