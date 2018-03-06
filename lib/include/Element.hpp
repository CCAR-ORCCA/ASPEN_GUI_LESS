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
	Recomputes element-specific values
	*/
	void update() ;

	/**
	Get element normal. 
	Definition varies depending upon Element type: 
	- the facet normal if the element is a triangular facet
	- the normal evaluated at the center of the element if the element is a Bezier patch
	@return element normal. 
	*/
	arma::vec get_normal() const;

	/**
	Get element center. 
	Definition varies depending upon Element type:
	- vertices average position if the element is a triangular facet
	- Bezier patch evaluated at u == v == w == 1./3 if the element is a Bezier patch
	@return element center
	*/
	arma::vec get_center() const;


	/**
	Return the control points owned by this element
	@return owned control points
	*/	
	std::vector<std::shared_ptr<ControlPoint > >  * get_control_points();

	

	/**
	Return surface area of element
	@return surface area of element
	*/
	double get_area() const;


protected:

	virtual void compute_center() = 0;
	virtual void compute_normal() = 0;
	virtual void compute_area() = 0;


	std::vector<std::shared_ptr<ControlPoint > >  control_points ;
	std::map<std::shared_ptr<ControlPoint> ,unsigned int> pointer_to_global_index;

	
	arma::vec normal;
	arma::vec center;

	double area;

};











#endif 