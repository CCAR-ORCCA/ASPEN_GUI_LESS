#ifndef HEADER_SPFH
#define HEADER_SPFH

#include "PointDescriptor.hpp"
#include "PointCloud.hpp"

class SPFH : public PointDescriptor{

public:

	SPFH(std::shared_ptr<PointNormal> & query_point,
		const std::vector<std::shared_ptr<PointNormal> > & points,
		bool keep_correlations = true,int N_bins = 3);
	
	SPFH();


	SPFH(const int & query_point,
		const std::vector<int > & points,int N_bins,
		const PointCloud<PointNormal> & pc);


	double get_distance_to_closest_neighbor() const;



protected:

	double distance_to_closest_neighbor;
};


#endif