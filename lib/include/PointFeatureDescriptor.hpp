#ifndef HEADER_PFD
#define HEADER_PFD

#include <vector>
class PointNormal;

class PointFeatureDescriptor{

public:

	PointFeatureDescriptor(std::vector<std::shared_ptr<PointNormal> > & points);
	PointFeatureDescriptor();


	std::vector<double> get_histogram() const;
	double get_histogram_value(int bin_index) const;
	double distance_to(const PointFeatureDescriptor & descriptor) const;

protected:
	
	std::vector<double> histogram;
};





#endif