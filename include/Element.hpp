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
	virtual std::set < Element *> get_neighbors(bool all_neighbors) const = 0;


	/**
	Recomputes elements
	*/
	virtual void update() = 0;

	/**
	Return pointer to facet center
	@return pointer to facet center
	*/
	arma::vec * get_facet_center() ;


	/**
	Return the control points owned by this element
	@return owned control points
	*/	
	std::vector<std::shared_ptr<ControlPoint > >  * get_control_points();

protected:

	virtual void compute_facet_center() = 0;

	std::vector<std::shared_ptr<ControlPoint > >  control_points ;
	arma::vec facet_center;

};











#endif 