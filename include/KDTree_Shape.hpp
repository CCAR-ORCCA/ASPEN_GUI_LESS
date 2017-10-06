#ifndef HEADER_KDTree_Shape
#define HEADER_KDTree_Shape

#include "Facet.hpp"
#include "BBox.hpp"
#include "Ray.hpp"

#include <set>


// Implementation of a KDTree based on the
// very informative yet incomplete post found at
// https://blog.frogslayer.com/kd-trees-for-faster-ray-tracing-with-triangles/
//
class Ray;

class KDTree_shape {

public:
	BBox bbox;
	std::shared_ptr<KDTree_shape> left;
	std::shared_ptr<KDTree_shape> right;
	std::vector<std::shared_ptr<Facet> > facets;

	KDTree_shape();



	std::shared_ptr<KDTree_shape> build(std::vector<std::shared_ptr<Facet >> & facets, int depth, bool verbose = false);
	bool hit(KDTree_shape * node, Ray * ray, bool computed_mes) const;
	bool hit_bbox(Ray * ray, bool computed_mes ) const;

	int get_depth() const;
	void set_depth(int depth);

protected:

	int depth;
	int max_depth = 1000;




};


#endif