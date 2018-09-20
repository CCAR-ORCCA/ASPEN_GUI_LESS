#include "PointDescriptor.hpp"
#include "PointNormal.hpp"


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


	arma::vec::fixed<3> v = arma::cross(n_i,arma::normalise(p_j - p_i));
	arma::vec::fixed<3> w = arma::cross(n_i,v);

	// All angles are within [0,pi]
	double alpha = std::acos(arma::dot(v,n_j));
	double phi = std::acos(arma::dot(n_i,arma::normalise(p_j - p_i)));
	double theta = std::atan(arma::dot(w,n_j)/arma::dot(n_i,n_j)) + arma::datum::pi/2;

	alpha_bin_index = (int)(std::floor(alpha/  (arma::datum::pi/N_bins)));
	phi_bin_index = (int)(std::floor(phi / (arma::datum::pi/N_bins)));
	theta_bin_index = (int)(std::floor(theta / (arma::datum::pi/N_bins)));

}






int PointDescriptor::get_type() const{
	return this -> type;
}


