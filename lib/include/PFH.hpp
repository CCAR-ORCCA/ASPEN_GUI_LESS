#ifndef HEADER_PFH
#define HEADER_PFH

#include "PointDescriptor.hpp"
#include "PointCloud.hpp"
#include "PointNormal.hpp"

class PFH : public PointDescriptor{

public:

	PFH();
	PFH(std::vector<std::shared_ptr<PointNormal> > & points,
		bool keep_correlations = true,int N_bins = 3);
	PFH(const std::vector<int> & indices, 
		const int & N_bins,
		const PointCloud<PointNormal> & pc);


protected:

	
	
};





#endif