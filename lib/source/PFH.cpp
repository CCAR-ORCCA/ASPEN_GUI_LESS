#include "PFH.hpp"
#include <armadillo>
#include "PointNormal.hpp"

PFH::PFH() : PointDescriptor(){

}

PFH::PFH(std::vector<std::shared_ptr<PointNormal> > & points,
	bool keep_correlations,int N_bins) : PointDescriptor(){

	arma::vec::fixed<3> u,v,w,p_i,p_j,n_j;

	int N_features = points.size() * (points.size() -1) / 2;
	int N_global_bins;

	if (keep_correlations){
		N_global_bins = std::pow(3,N_bins);
	}
	else{
		N_global_bins = 3 * N_bins;
	}


	this -> histogram = arma::zeros<arma::vec>(N_global_bins);

	for (int i = 0; i < points.size(); ++i){

		u = points.at(i) -> get_normal();
		p_i = points.at(i) -> get_point();

		for (int j = 0; j < i; ++j){

			p_j = points.at(j) -> get_point();
			n_j = points.at(j) -> get_normal();
			
			v = arma::cross(u,arma::normalise(p_j - p_i));
			w = arma::cross(u,v);

			// All angles are within [0,pi]
			double alpha = std::acos(arma::dot(v,n_j));
			double phi = std::acos(arma::dot(u,arma::normalise(p_j - p_i)));
			double theta = std::atan(arma::dot(w,n_j)/arma::dot(u,n_j)) + arma::datum::pi/2;
			
			int alpha_bin_index = (int)(std::floor(alpha/  (arma::datum::pi/N_bins)));
			int phi_bin_index = (int)(std::floor(phi / (arma::datum::pi/N_bins)));
			int theta_bin_index = (int)(std::floor(theta / (arma::datum::pi/N_bins)));

			if (keep_correlations){
				int global_bin_index = alpha_bin_index +  phi_bin_index * (N_bins) + theta_bin_index * (N_bins * N_bins);
				this -> histogram[global_bin_index] += 1./N_features;
			}
			else{

				this -> histogram[alpha_bin_index] += 1./N_features;
				this -> histogram[N_bins + phi_bin_index] += 1./N_features;
				this -> histogram[2 * N_bins + theta_bin_index] += 1./N_features;

			}

		}
	}

	if (arma::max(this -> histogram) > 0){
		this -> histogram = this -> histogram / arma::max(this -> histogram);
	}
}

