#ifndef HEADER_KDNode
#define HEADER_KDNode

#include "Facet.hpp"
#include "ShapeModel.hpp"
#include "FrameGraph.hpp"
#include "BBox.hpp"
#include "Lidar.hpp"
#include <set>
// #include "Ray.hpp"


// Implementation of a KDTree based on the
// very informative yet incomplete post found at
// https://blog.frogslayer.com/kd-trees-for-faster-ray-tracing-with-triangles/
//
class Ray;

class KDNode {

public:
	BBox bbox;
	std::shared_ptr<KDNode> left;
	std::shared_ptr<KDNode> right;
	std::vector<Facet * > facets;

	KDNode();

	std::shared_ptr<KDNode> build(std::vector<Facet *> & facets, int depth, bool verbose = false);
	bool hit(KDNode * node, Ray * ray) const;
	bool hit_bbox(Ray * ray ) const;

	int get_depth() const;
	void set_depth(int depth);

protected:

	int depth;
	int max_depth = 1000;
	ShapeModel * shape_model;




};


#endif