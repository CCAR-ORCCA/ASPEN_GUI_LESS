#ifndef HEADER_Bezier
#define HEADER_Bezier
#include <armadillo>
#include "ControlPoint.hpp"
#include "Element.hpp"
#include <boost/math/special_functions/factorials.hpp>
#include <memory>
#include <iostream>

#include <set>
#include <map>

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
	Returns patch n
	@param n
	*/
	unsigned int get_n() const;

	/**
	Elevates patch n
	*/
	void elevate_n();


	/**	
	Returns the outgoing normal vector at (u,v)
	@param u first barycentric coordinate
	@param v second barycentric coordinate
	@return normal vector at the queried point
	*/
	arma::vec get_normal(double u, double v) const;



	/**
	Evaluates the bezier patch at the barycentric 
	coordinates (u,v). Note that 0<= u + v <= 1
	@param u first barycentric coordinate
	@param v second barycentric coordinate
	@return point at the surface of the bezier patch
	*/
	arma::vec evaluate(const double u, const double v) const;


		
	/**
	Evaluates the Berstian polynomial
	@param u first barycentric coordinate
	@param v first barycentric coordinate
	@param i first index
	@param v second index
	@return evaluated bernstein polynomial
	*/
	double bernstein(
		const double u, 
		const double v,
		const unsigned int i,
		const unsigned int j) const;




protected:

	virtual void compute_normal();
	virtual void compute_area();
	virtual void compute_center();

	void construct_index_tables();

	unsigned int n;


	std::map<unsigned int,unsigned int> global_to_i;
	std::map<unsigned int,unsigned int> global_to_j;




};
#endif