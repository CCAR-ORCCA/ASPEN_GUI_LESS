#ifndef HEADER_Bezier
#define HEADER_Bezier
#include <armadillo>
#include "ControlPoint.hpp"
#include "Element.hpp"

#include <memory>
#include <iostream>

#include <set>

class ControlPoint;

class Bezier : public Element{

public:

	/**
	Constructor
	@param vertices pointer to vector storing the vertices owned by this facet
	*/
	Bezier( std::vector<std::shared_ptr<ControlPoint > > vertices);

	/**
	Get neighbors
	@param if false, only return neighbors sharing an edge. Else, returns all neighbords
	@return Pointer to neighboring facets, plus the calling facet
	*/
	virtual std::set < Element * > get_neighbors(bool all_neighbors) const;


	/**
	Returns pointer to the first vertex owned by $this that is
	neither $v0 and $v1. When $v0 and $v1 are on the same edge,
	this method returns a pointer to the vertex of $this that is not
	on the edge but still owned by $this
	@param v0 Pointer to first vertex to exclude
	@param v1 Pointer to first vertex to exclude
	@return Pointer to the first vertex of $this that is neither $v0 and $v1
	*/
	std::shared_ptr<ControlPoint> vertex_not_on_edge(std::shared_ptr<ControlPoint> v0,
	        std::shared_ptr<ControlPoint>v1) const ;

	/**
	Returns patch order
	@param order
	*/
	unsigned int get_order() const;
	


protected:

	virtual void compute_normal();
	virtual void compute_area();
	virtual void compute_center();

	unsigned int order;


	unsigned int split_counter = 0;
	unsigned int hit_count = 0;

};
#endif