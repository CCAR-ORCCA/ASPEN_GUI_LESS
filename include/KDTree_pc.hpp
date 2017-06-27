#ifndef HEADER_KDTree_pc
#define HEADER_KDTree_pc

#include <memory>
#include "PointNormal.hpp"

class KDTree_pc {

public:
	std::shared_ptr<KDTree_pc> left;
	std::shared_ptr<KDTree_pc> right;
	std::vector<std::shared_ptr<PointNormal> > points_normals;

	KDTree_pc();

	std::shared_ptr<KDTree_pc> build(std::vector< std::shared_ptr<PointNormal> > & points_normals, int depth, bool verbose = false);


	int get_depth() const;
	void set_depth(int depth);

	unsigned int get_size() const;

	std::vector<std::shared_ptr<PointNormal> > * get_points_normals();


protected:

	int depth;
	int max_depth = 1000;

};


#endif