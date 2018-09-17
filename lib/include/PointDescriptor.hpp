#ifndef HEADER_POINTDESCRIPTOR
#define HEADER_POINTDESCRIPTOR

#include <armadillo>

class PointNormal;

class PointDescriptor{

public:

	PointDescriptor();

	arma::vec get_histogram() const;
	unsigned int get_histogram_size() const;
	double get_histogram_value(int bin_index) const;
	double distance_to(const PointDescriptor & descriptor) const;
	double distance_to(PointDescriptor * descriptor) const;

protected:
	
	arma::vec histogram;
};





#endif