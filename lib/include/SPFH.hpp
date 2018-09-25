#ifndef HEADER_SPFH
#define HEADER_SPFH

#include "PointDescriptor.hpp"
#include "PointCloud.hpp"

class SPFH : public PointDescriptor{

public:

	
	SPFH();


	SPFH(const int & query_point,
		const std::vector<int > & points,int N_bins,
		const PointCloud<PointNormal> & pc);


	double get_distance_to_closest_neighbor() const;



protected:

	double distance_to_closest_neighbor;
};


#endif