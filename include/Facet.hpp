#ifndef HEADER_FACET
#define HEADER_FACET
#include <armadillo>
#include "ControlPoint.hpp"
#include "Element.hpp"

#include <memory>
#include <iostream>

#include <set>

class ControlPoint;
class Facet {

public:

	/**
	Constructor
	@param vertices pointer to vector storing the vertices owned by this facet
	*/
	Facet( std::vector<std::shared_ptr<ControlPoint > > vertices);


	/**
	Get neighbors
	@param if false, only return neighbors sharing an edge. Else, returns all neighbords
	@return Pointer to neighboring facets, plus the calling facet
	*/
	std::set < Facet * > get_neighbors(bool all_neighbors) const;

	/**
	Get outbound facet normal
	@return pointer to facet normal
	*/
	arma::vec * get_facet_normal()  ;


	/**
	Checks the degeneracy of the facet
	@param angle minimum angle indicating that the facet is degenerate
	@return True if facet is not degenerate, false otherwise
	*/
	bool has_good_surface_quality(double angle) const;


	/**
	Increases the hit count by one
	*/
	void increase_hit_count();


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
	Recomputes the facet normal, surface area and center
	*/
	void update();



	/**
	Return facet surface area
	@return facet surface area
	*/
	double get_area() const;

	/**
	Return pointer to facet center
	@return pointer to facet center
	*/
	arma::vec * get_facet_center() ;

	/**
	Return the vector storing the vertices owned by this facet
	@return vector storing the vertices owned by this facet
	*/
	std::vector<std::shared_ptr<ControlPoint > > * get_vertices() ;


	/**
	Sets the number of time this facet has been split
	@param split_counter Number of time this facet and its parents
	have been split
	*/
	void set_split_counter(unsigned int split_counter);

	/**
	Returns the number of times the (deceased)
	parents of that face were split
	@return number of time the parents of that
	facet where
	*/
	unsigned int get_split_count() const;



	/**
	Get the number of times this facet has been hit
	@return hit_count Number of time this facet has been hit by an inbound ray
	*/
	unsigned int get_hit_count() const;

	unsigned int recycle_count = 0;

protected:

	void compute_normal();
	void compute_area();
	void compute_facet_center();


	std::vector<std::shared_ptr<ControlPoint > >  vertices ;
	arma::vec facet_normal;
	arma::vec facet_center;

	double area;
	unsigned int split_counter = 0;
	unsigned int hit_count = 0;

};
#endif