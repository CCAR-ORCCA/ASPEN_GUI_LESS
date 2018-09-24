#include "PFH.hpp"
#include <armadillo>
#include "PointNormal.hpp"

PFH::PFH() : PointDescriptor(){

}

PFH::PFH(std::vector<std::shared_ptr<PointNormal> > & points,
	bool keep_correlations,int N_bins) : PointDescriptor(){

	this -> type = 0;

	arma::vec::fixed<3> u,v,w,p_i,p_j,n_j;

	int N_global_bins;

	if (keep_correlations){
		N_global_bins = std::pow(3,N_bins);
	}
	else{
		N_global_bins = 3 * N_bins;
	}

	this -> histogram = arma::zeros<arma::vec>(N_global_bins);

	
	for (int i = 0; i < points.size(); ++i){

		for (int j = 0; j < i; ++j){

			int alpha_bin_index,phi_bin_index,theta_bin_index;
			
			this -> compute_darboux_frames_local_hist( alpha_bin_index,phi_bin_index,theta_bin_index, N_bins,
				points.at(i) -> get_point_coordinates(),points.at(i) -> get_normal_coordinates(),points.at(j) -> get_point_coordinates(),points.at(j) -> get_normal_coordinates());

			if (keep_correlations){
				int global_bin_index = theta_bin_index +  alpha_bin_index * (N_bins) + phi_bin_index * (N_bins * N_bins);
				this -> histogram[global_bin_index] += 1.;
			}
			else{

				this -> histogram[theta_bin_index] += 1.;
				this -> histogram[N_bins + alpha_bin_index] += 1.;
				this -> histogram[2 * N_bins + phi_bin_index] += 1.;

			}

		}
	}

	if (arma::max(this -> histogram) > 0){
		this -> histogram = this -> histogram / arma::max(this -> histogram) * 100;
	}

}


PFH::PFH(const std::vector<int> & indices, 
	const int & N_bins,
	const PointCloud<PointNormal> & pc): PointDescriptor(){

	this -> type = 0;
	arma::vec::fixed<3> u,v,w,p_i,p_j,n_j;
	int N_global_bins = 3 * N_bins;
	
	this -> histogram = arma::zeros<arma::vec>(N_global_bins);

	int alpha_bin_index,phi_bin_index,theta_bin_index;
	
	for (int i = 0; i < indices.size(); ++i){
		const PointNormal & pi = pc.get_point(indices[i]);

		for (int j = 0; j < i; ++j){

			const PointNormal & pj = pc. get_point(indices[j]);

			
			this -> compute_darboux_frames_local_hist( alpha_bin_index,phi_bin_index,theta_bin_index, N_bins,
				pi.get_point_coordinates(),
				pi.get_normal_coordinates(),
				pj.get_point_coordinates(),
				pj.get_normal_coordinates());

			this -> histogram[theta_bin_index] += 1.;
			this -> histogram[N_bins + alpha_bin_index] += 1.;
			this -> histogram[2 * N_bins + phi_bin_index] += 1.;

		}
	}

	if (arma::max(this -> histogram) > 0){
		this -> histogram = this -> histogram / arma::max(this -> histogram) * 100;
	}





}



