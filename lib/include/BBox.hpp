#ifndef HEADER_BBOX
#define HEADER_BBOX

#include <time.h>
#include <vector>

class ShapeModel;
/**
Declaration of the BBox class, representing
the bounding box of a KDTree node
*/
class BBox {

public:

	BBox(ShapeModel * owning_shape);
	BBox();
	void set_owning_shape(ShapeModel * owning_shape);

	/**
	Computes the bounding box boundaries
	using provided geometric data
	@param elements Facets bounded by this box
	*/
	void update(const std::vector<int> & element_indices);


	/**
	Computes the bounding box boundaries
	using provided geometric data
	@param element Element bounded by this box
	*/
	void update(int element_index);

	void print() const;


	unsigned int get_longest_axis() const;

	double get_xmin() const;
	double get_xmax() const;
	double get_ymin() const;
	double get_ymax() const;
	double get_zmin() const;
	double get_zmax() const;

	void save_to_file(std::string path) const;
	void reset_bbox();

protected:

	double xmin, xmax;
	double ymin, ymax;
	double zmin, zmax;

	ShapeModel * owning_shape;



};

#endif