#ifndef HEADER_FPFH
#define HEADER_FPFH

#include "PointDescriptor.hpp"
#include "SPFH.hpp"

class FPFH : public PointDescriptor{

public:

	FPFH();

	FPFH(const std::shared_ptr<PointNormal> & query_point,const std::vector<std::shared_ptr<PointNormal> > & points);

	FPFH(const int & query_point,
		const std::vector<std::vector<int> >  & points,
		const std::vector<SPFH> & spfhs,
		const PointCloud<PointNormal> & pc,
		bool scale_distance);

protected:
	
};


#endif