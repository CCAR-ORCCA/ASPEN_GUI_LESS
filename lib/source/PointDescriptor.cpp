#include "PointDescriptor.hpp"
#include "PointNormal.hpp"

#define POINTDESCRIPTOR_DEBUG 0

PointDescriptor::PointDescriptor(){

}


PointDescriptor::PointDescriptor(arma::vec histogram){
	this -> histogram = histogram;
}

const arma::vec & PointDescriptor::get_histogram() const{
	return this -> histogram;
}

double PointDescriptor::get_histogram_value(int bin_index) const{
	return this -> histogram(bin_index);
}




double PointDescriptor::distance_to_descriptor(const PointDescriptor & descriptor) const{

	return this -> distance_to_descriptor(descriptor.get_histogram());
}




double PointDescriptor::distance_to_descriptor(const arma::vec & histogram) const{

	double distance = 0;

	for (int i = 0; i < this -> histogram.size(); ++i){
		double pi = this -> get_histogram_value(i);
		double qi = histogram(i);

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



	arma::vec::fixed<3> d = arma::normalise(p_j - p_i);
	arma::vec::fixed<3> n1(n_i);
	arma::vec::fixed<3> n2(n_j);


	double angle_1 = arma::dot(n_i,d);
	double angle_2 = arma::dot(n_j,d);
	double phi;

	// Ensures consistency of pair orientations
	if (std::acos(std::abs(angle_1)) > std::acos(std::abs(angle_2))){
		n1 = n_j;
		n2 = n_i;
		d *= -1.;
		phi = - angle_2;
	}
	else{
		phi = angle_1;
	}


	arma::vec::fixed<3> v = arma::normalise(arma::cross(d,n1));
	arma::vec::fixed<3> w = arma::cross(n1,v);

	double alpha = arma::dot(v,n2);
	double theta = std::atan2(arma::dot(w,n2),arma::dot(n1,n2));


	// All values are wrapped within [0,1]
	alpha = 0.5 * ( 1. + alpha);
	phi = 0.5 * ( 1. + phi);
	theta = (theta + arma::datum::pi) * 1.0 / (2.0 * arma::datum::pi);

	alpha_bin_index = static_cast<int>(std::floor(alpha * N_bins));
	phi_bin_index = static_cast<int>(std::floor(phi  * N_bins));
	theta_bin_index = static_cast<int>(std::floor(theta  * N_bins));

}


int PointDescriptor::get_type() const{
	return this -> type;
}



int PointDescriptor::get_global_index() const{
	return this -> global_index; 
}
void PointDescriptor::set_global_index(int index){
	this -> global_index = index;
}


bool PointDescriptor::get_is_valid_feature() const{
	return this -> is_valid_feature;
}

void PointDescriptor::set_is_valid_feature(bool active_feature){
	this -> is_valid_feature = active_feature;
}

