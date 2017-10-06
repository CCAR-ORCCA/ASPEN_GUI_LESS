#ifndef HEADER_ELEMENT
#define HEADER_ELEMENT

#include <armadillo>
#include <iostream>
#include <memory>
#include <set>
#include "ControlPoint.hpp"

class ControlPoint;
class Element {


public:

	/**
	Constructor
	@param control_points pointer to vector storing the control_points owned by this element
	*/
	Element(std::vector<std::shared_ptr<ControlPoint > > control_points);

	/**
	Get neighbors
	@param if false, only return neighbors sharing an edge. Else, returns all neighbors
	@return Pointer to neighboring elements, plus the calling element
	*/
	std::set < Element * > get_neighbors(bool all_neighbors) const;


	/**
	Recomputes elements
	*/
	virtual void update() = 0;

	/**
	Return pointer to facet center
	@return pointer to facet center
	*/
	arma::vec * get_facet_center() ;

protected:

	virtual void compute_facet_center() = 0;

	std::vector<std::shared_ptr<ControlPoint > >  control_points ;
	arma::vec facet_center;

};











#endif 