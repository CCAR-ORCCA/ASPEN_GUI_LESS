#include "PointDescriptor.hpp"
#include "PointNormal.hpp"

#define POINTDESCRIPTOR_DEBUG 0

PointDescriptor::PointDescriptor(){

}


PointDescriptor::PointDescriptor(arma::vec histogram){
	this -> histogram = histogram;
}

arma::vec PointDescriptor::get_histogram() const{
	return this -> histogram;
}

double PointDescriptor::get_histogram_value(int bin_index) const{
	return this -> histogram(bin_index);
}


double PointDescriptor::distance_to_descriptor(const PointDescriptor * descriptor) const{

	double distance = 0;

	for (int i = 0; i < this -> histogram.size(); ++i){
		double pi = this -> get_histogram_value(i);
		double qi = descriptor -> get_histogram_value(i);

		if (pi > 0 || qi > 0){
			distance += std::pow(pi - qi,2)/(pi + qi);
		}
	}
	return distance;

}

unsigned int PointDescriptor::get_histogram_size() const{
	return this -> histogram.size();
}



void PointDescriptor::compute_darboux_frames_local_hist( int & alpha_bin_index,
	int & phi_bin_index,int & theta_bin_index, 
	const int & N_bins,
	const arma::vec::fixed<3> & p_i,
	const arma::vec::fixed<3> & n_i,
	const arma::vec::fixed<3> & p_j,
	const arma::vec::fixed<3> & n_j) {


	arma::vec::fixed<3> v = arma::cross(arma::normalise(p_j - p_i),n_i);
	arma::vec::fixed<3> w = arma::cross(n_i,v);


	// All values are within [0,1]
	double alpha = arma::dot(v,n_j);
	double phi = arma::dot(n_i,arma::normalise(p_j - p_i));
	double theta = std::atan2(arma::dot(w,n_j),arma::dot(n_i,n_j));

	// All values are wrapped within [0,1]
	alpha = 0.5 * ( 1. + alpha);
	phi = 0.5 * ( 1. + phi);
	theta = (theta + arma::datum::pi) * 1.0 / (2.0 * arma::datum::pi);


	alpha_bin_index = (int)(std::floor(alpha * N_bins));
	phi_bin_index = (int)(std::floor(phi  * N_bins));
	theta_bin_index = (int)(std::floor(theta  * N_bins));

}






int PointDescriptor::get_type() const{
	return this -> type;
}


