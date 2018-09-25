#include "PFH.hpp"
#include <armadillo>
#include "PointNormal.hpp"

PFH::PFH() : PointDescriptor(){

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



