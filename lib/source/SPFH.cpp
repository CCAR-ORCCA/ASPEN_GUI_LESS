#include "SPFH.hpp"
#include <armadillo>
#include "PointNormal.hpp"


SPFH::SPFH() : PointDescriptor(){}

SPFH::SPFH(std::shared_ptr<PointNormal> & query_point,
	std::vector<std::shared_ptr<PointNormal> > & points,
	bool keep_correlations,int N_bins) : PointDescriptor(){

	arma::vec::fixed<3> u,v,w,p_i,p_j,n_j;

	int N_global_bins;

	if (keep_correlations){
		N_global_bins = std::pow(3,N_bins);
	}
	else{
		N_global_bins = 3 * N_bins;
	}


	this -> neighbors_exclusive.clear();
	this -> histogram = arma::zeros<arma::vec>(N_global_bins);

	
	for (int j = 0; j < points.size(); ++j){

		if (arma::norm(points.at(j) -> get_point() - query_point -> get_point()) > 0){


			int alpha_bin_index,phi_bin_index,theta_bin_index;

			this -> neighbors_exclusive.push_back(points.at(j));
			this -> compute_darboux_frames_local_hist( alpha_bin_index,phi_bin_index,theta_bin_index, N_bins,
				query_point -> get_point(),query_point -> get_normal(),points.at(j) -> get_point(),points.at(j) -> get_normal());


			if (keep_correlations){
				int global_bin_index = alpha_bin_index +  phi_bin_index * (N_bins) + theta_bin_index * (N_bins * N_bins);
				this -> histogram(global_bin_index) += 1.;

			}
			else{

				this -> histogram(alpha_bin_index) += 1;
				this -> histogram(N_bins + phi_bin_index) += 1.;
				this -> histogram(2 * N_bins + theta_bin_index) += 1.;
			}

		}

	}


}
