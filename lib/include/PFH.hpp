#ifndef HEADER_PFH
#define HEADER_PFH

#include "PointDescriptor.hpp"

class PFH : public PointDescriptor{

public:

	PFH(std::vector<std::shared_ptr<PointNormal> > & points,
		bool keep_correlations = true,int N_bins = 3);
	PFH();


protected:

	
	
};





#endif