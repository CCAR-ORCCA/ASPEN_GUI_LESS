#ifndef HEADER_BBOX
#define HEADER_BBOX

#include "FrameGraph.hpp"
#include "Facet.hpp"
#include <time.h>

/**
Declaration of the BBox class, representing
the bounding box of a KDTree node
*/
class BBox {

public:

	BBox();

	/**
	Computes the bounding box boundaries
	using provided geometric data
	@param facets Facets bounded by this box
	*/
	void update(std::vector<std::shared_ptr<Facet> > facets);


	/**
	Computes the bounding box boundaries
	using provided geometric data
	@param facet Facet bounded by this box
	*/
	void update(std::shared_ptr<Facet> facet);

	void print() const;


	unsigned int get_longest_axis() const;

	double get_xmin() const;
	double get_xmax() const;
	double get_ymin() const;
	double get_ymax() const;
	double get_zmin() const;
	double get_zmax() const;

	void save_to_file(std::string path) const;


protected:
	double xmin, xmax;
	double ymin, ymax;
	double zmin, zmax;



};

#endif