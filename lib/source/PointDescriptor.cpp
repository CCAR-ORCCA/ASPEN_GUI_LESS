#include "PointDescriptor.hpp"
#include "PointNormal.hpp"


PointDescriptor::PointDescriptor(){
	
}


std::vector<double> PointDescriptor::get_histogram() const{
	return this -> histogram;
}

double PointDescriptor::get_histogram_value(int bin_index) const{
	return this -> histogram[bin_index];
}

double PointDescriptor::distance_to(const PointDescriptor & descriptor) const{

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


double PointDescriptor::distance_to(PointDescriptor * descriptor) const{

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


