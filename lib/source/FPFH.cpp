#include "FPFH.hpp"
#include "PointNormal.hpp"
#include <armadillo>

FPFH::FPFH(const std::shared_ptr<PointNormal> & query_point,const std::vector<std::shared_ptr<PointNormal> > & points) : PointDescriptor(){

	this -> histogram = query_point -> get_SPFH() -> get_histogram();

	double distance_to_closest_neighbor = query_point -> get_SPFH() -> get_distance_to_closest_neighbor();
	
	for (unsigned int k = 0 ; k < points. size(); ++k){
		double distance = arma::norm(query_point -> get_point_coordinates() - points . at(k) -> get_point_coordinates());
		
		if (distance>0){
			// this -> histogram += 1. * points. at(k) -> get_SPFH() -> get_histogram() / (distance * points.size());	
			this -> histogram += distance_to_closest_neighbor * points. at(k) -> get_SPFH() -> get_histogram() / (distance * points.size());
		}
	}

	// The uncorrelated histograms are normalized



	this -> histogram.subvec(0,10) *= 100./arma::sum(this -> histogram.subvec(0,10));
	this -> histogram.subvec(11,21) *= 100./arma::sum(this -> histogram.subvec(11,21));
	this -> histogram.subvec(22,32) *= 100./arma::sum(this -> histogram.subvec(22,32));

	

	this -> type = 1;
	

}



FPFH::FPFH(const int & query_point,
	const std::vector<std::vector<int> > & points,
	const std::vector<SPFH> & spfhs,
	const PointCloud<PointNormal> & pc,
	bool scale_distance) : PointDescriptor(){

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
	this -> histogram.subvec(0, N_bins - 1) *= 100./arma::sum(this -> histogram.subvec(0, N_bins - 1));
	this -> histogram.subvec(N_bins,2 * N_bins - 1) *= 100./arma::sum(this -> histogram.subvec(N_bins,2 * N_bins - 1));
	this -> histogram.subvec(2 * N_bins,3 * N_bins - 1) *= 100./arma::sum(this -> histogram.subvec(2 * N_bins,3 * N_bins - 1));

	
	this -> type = 1;
	

}


FPFH::FPFH() : PointDescriptor(){}


