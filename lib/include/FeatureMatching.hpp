#ifndef HEADER_MATCHFEATURE
#define HEADER_MATCHFEATURE

#include <PointCloud.hpp>
#include <PointDescriptor.hpp>
typedef typename std::pair<int, int > PointPair ;

template <class T> class FeatureMatching{

public:

	/**
	Constructor
	@param pc1 Reference to first descriptor point cloud . The features in pc1 will be associated with their nearest neighbors in pc2
	@param pc2 Reference to second descriptor point cloud. Will have its kd tree computed/recomputed.
	@param N number of closest neighbors in feature space to search
	*/	
	FeatureMatching(const PointCloud<T> & pc1, PointCloud<T> & pc2);
	void match(std::map<int, std::vector< int > > & matches,const int & N); 
	void match(std::vector< PointPair > & matches,const int & N);
	static void save_matches(std::string path,const std::vector<PointPair> & matches,
		const PointCloud<T> & input_pc1,
		const PointCloud<T> & input_pc2) ;

protected:
	const PointCloud<T> & pc1;
	PointCloud<T> & pc2;




};


#endif