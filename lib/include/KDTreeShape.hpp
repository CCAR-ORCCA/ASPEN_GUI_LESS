#ifndef HEADER_KDTreeShape
#define HEADER_KDTreeShape

#include "Element.hpp"
#include "Facet.hpp"

#include "BBox.hpp"
#include "Ray.hpp"

#include <set>


// Implementation of a KDTree based on the
// very informative yet incomplete post found at
// https://blog.frogslayer.com/kd-trees-for-faster-ray-tracing-with-triangles/
//
class Ray;
class ShapeModel;
class ShapeModelTri;
class ShapeModelBezier;


class KDTreeShape {

public:
	BBox bbox;
	std::shared_ptr<KDTreeShape> left;
	std::shared_ptr<KDTreeShape> right;
	std::vector<int > elements;

	KDTreeShape(ShapeModelTri * owning_shape);

	void build(const std::vector<int> & elements, int depth);
	bool hit_bbox(Ray * ray) const;	
	bool hit(const std::shared_ptr<KDTreeShape> & node, Ray * ray, ShapeModelBezier * shape_model_bezier = nullptr) const;

	int get_depth() const;
	void set_depth(int depth);

protected:

	int depth;
	int max_depth = 1000;

	ShapeModelTri * owning_shape;

};


#endif