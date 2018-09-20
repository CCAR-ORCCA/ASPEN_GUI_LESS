#ifndef HEADER_FPFH
#define HEADER_FPFH

#include "PointDescriptor.hpp"

class FPFH : public PointDescriptor{

public:

	FPFH();

	FPFH(const std::shared_ptr<PointNormal> & query_point,const std::vector<std::shared_ptr<PointNormal> > & points);


protected:
	
};


#endif