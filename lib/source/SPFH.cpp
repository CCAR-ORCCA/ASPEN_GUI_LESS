#include "SPFH.hpp"
#include <armadillo>
#include "PointNormal.hpp"


SPFH::SPFH() : PointDescriptor(){}

SPFH::SPFH(std::shared_ptr<PointNormal> & query_point,
	const std::vector<std::shared_ptr<PointNormal> > & points,
	bool keep_correlations,int N_bins) : PointDescriptor(){

	arma::vec::fixed<3> u,v,w,p_i,p_j,n_j;

	int N_global_bins;

	if (keep_correlations){
		N_global_bins = std::pow(3,N_bins);
	}
	else{
		N_global_bins = 3 * N_bins;
	}

	this -> histogram = arma::zeros<arma::vec>(N_global_bins);

	this -> distance_to_closest_neighbor = std::numeric_limits<double>::infinity();
	
	for (int j = 0; j < points.size(); ++j){
		double distance = arma::norm(points.at(j) -> get_point() - query_point -> get_point());

		if (distance > 0){

			int alpha_bin_index,phi_bin_index,theta_bin_index;

			this -> distance_to_closest_neighbor = std::min(this-> distance_to_closest_neighbor,distance);

			this -> compute_darboux_frames_local_hist( alpha_bin_index,phi_bin_index,theta_bin_index, N_bins,
				query_point -> get_point(),query_point -> get_normal(),points.at(j) -> get_point(),points.at(j) -> get_normal());


			if (keep_correlations){
				int global_bin_index = alpha_bin_index +  phi_bin_index * (N_bins) + theta_bin_index * (N_bins * N_bins);
				this -> histogram(global_bin_index) += 1.;

			}
			else{

				this -> histogram(alpha_bin_index) += 1.;
				this -> histogram(N_bins + phi_bin_index) += 1.;
				this -> histogram(2 * N_bins + theta_bin_index) += 1.;
			}

		}

	}

	if (arma::max(this -> histogram) > 0 ){
		this -> histogram = this -> histogram / arma::max(this -> histogram);
	}

}

double SPFH::get_distance_to_closest_neighbor() const{
	return this -> distance_to_closest_neighbor;
}
