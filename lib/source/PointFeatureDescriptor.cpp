#include "PointFeatureDescriptor.hpp"
#include "PointNormal.hpp"


PointFeatureDescriptor::PointFeatureDescriptor(){

}

PointFeatureDescriptor::PointFeatureDescriptor(std::vector<std::shared_ptr<PointNormal> > & points){

	arma::vec::fixed<3> u,v,w,p_i,p_j,n_j;

	int N_features = points.size() * (points.size() -1) / 2;
	int N_bins = 3;
	int N_global_bins = std::pow(3,N_bins);


	this -> histogram.clear();
	this -> histogram = std::vector<double>(N_global_bins,0.);

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
			

			int alpha_bin_index = (int)(std::floor(alpha/  (arma::datum::pi/3)));
			int phi_bin_index = (int)(std::floor(phi / (arma::datum::pi/3)));
			int theta_bin_index = (int)(std::floor(theta / (arma::datum::pi/3)));

			int global_bin_index = alpha_bin_index +  phi_bin_index * (N_bins) + theta_bin_index * (N_bins * N_bins);
			this -> histogram[global_bin_index] += 1./N_features;
		}

	}
	
}

std::vector<double> PointFeatureDescriptor::get_histogram() const{
	return this -> histogram;
}

double PointFeatureDescriptor::get_histogram_value(int bin_index) const{
	return this -> histogram[bin_index];
}

double PointFeatureDescriptor::distance_to(const PointFeatureDescriptor & descriptor) const{

	double distance = 0;

	for (int i = 0; i < this -> histogram.size(); ++i){
		double pi = this -> get_histogram_value(i);
		double qi = descriptor.get_histogram_value(i);

		if (pi > 0 || qi > 0){
			distance += std::pow(pi - qi,2)/(pi + qi);
		}
	}
	return distance;

}
