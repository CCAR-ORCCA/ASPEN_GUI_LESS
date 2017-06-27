#ifndef HEADER_KDTree_Shape
#define HEADER_KDTree_Shape

#include "Facet.hpp"
#include "FrameGraph.hpp"
#include "BBox.hpp"
#include "Lidar.hpp"

#include <set>



// Implementation of a KDTree based on the
// very informative yet incomplete post found at
// https://blog.frogslayer.com/kd-trees-for-faster-ray-tracing-with-triangles/
//
class Ray;

class KDTree_Shape {

public:
	BBox bbox;
	std::shared_ptr<KDTree_Shape> left;
	std::shared_ptr<KDTree_Shape> right;
	std::vector<Facet * > facets;

	KDTree_Shape();

	

	std::shared_ptr<KDTree_Shape> build(std::vector<Facet *> & facets, int depth, bool verbose = false);
	bool hit(KDTree_Shape * node, Ray * ray) const;
	bool hit_bbox(Ray * ray ) const;

	int get_depth() const;
	void set_depth(int depth);

protected:

	int depth;
	int max_depth = 1000;




};


#endif