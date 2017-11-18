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

struct NewPoint{
	NewPoint(std::shared_ptr<ControlPoint> point ,
		std::shared_ptr<ControlPoint> end_point ,
		std::tuple<unsigned int, unsigned int, unsigned int> indices_newpoint,
		std::tuple<unsigned int, unsigned int, unsigned int> indices_endpoint){

		this -> point = point;
		this -> end_point= end_point;
		this -> indices_newpoint = indices_newpoint;
		this -> indices_endpoint = indices_endpoint;
	} 

	std::shared_ptr<ControlPoint> point;
	std::shared_ptr<ControlPoint> end_point;
	std::tuple<unsigned int, unsigned int, unsigned int> indices_newpoint;
	std::tuple<unsigned int, unsigned int, unsigned int> indices_endpoint;

};


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
	Returns patch degree
	@param degree
	*/
	unsigned int get_n() const;

	/**
	Elevates patch degree by one
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
	Adds a new control point to this element from one of its neighbords
	@param element pointer to neighboring element
	@new_point structure storing all the information required to add the provided
	point to this Bezier element
	*/
	void add_point_from_neighbor(Element * element, NewPoint & new_point);


	/**
	Returns the control point given its i and j indices (k = n - i - j)
	@param i first index
	@param j second index
	@return pointer to control point
	*/	
	std::shared_ptr<ControlPoint> get_control_point(unsigned int i, unsigned int j);

	/**
	Returns the tuple of local indices (i,j,k) of a control point within a bezier patch
	given its local index
	@param local_index index comprised between 0 and (n + 1)(n + 2)/ - 1, where n is the patch degree
	*/
	std::tuple<unsigned int, unsigned int,unsigned int> get_local_indices(unsigned int local_index);






protected:

	virtual void compute_normal();
	virtual void compute_area();
	virtual void compute_center();

	void construct_index_tables();

	/**
	Evaluates the quadrature function for surface area computation
	@param u first barycentric coordinate
	@param v second barycentric coordinate (w = 1 - u - v)
	@param g as in A = \int g du dv 
	*/
	double g(double u, double v) const;

	/**
	Evaluates the Berstein polynomial
	@param u first barycentric coordinate
	@param v first barycentric coordinate
	@param i first index
	@param v second index
	@param n polynomial order
	@return evaluated bernstein polynomial
	*/
	static double bernstein(
		const double u, 
		const double v,
		const unsigned int i,
		const unsigned int j,
		const unsigned int n) ;

	/**
	Returns the number of combinations of k items among n.
	Returns 0 if k < 0 or k > n
	@param k subset size
	@param n set size
	@return number of combinations
	*/
	static unsigned int combinations(unsigned int k, unsigned int n);

	/**
	Generates the forward table associating a local index l to the corrsponding triplet (i,j,k) 
	@param n degree
	@return forward look up table
	*/
	static std::vector<std::tuple<unsigned int, unsigned int, unsigned int> > forward_table(unsigned int n);



	/**
	Generates the reverse table associating a local index triplet (i,j,k) to the corresponding
	global index l
	@param n degree
	@return reverse look up table
	*/
	static std::map< std::tuple<unsigned int, unsigned int, unsigned int> ,unsigned int> reverse_table(unsigned int n);


	unsigned int n;

	std::vector<std::tuple<unsigned int, unsigned int, unsigned int> > forw_table;
	std::map< std::tuple<unsigned int, unsigned int, unsigned int> ,unsigned int> rev_table;



	std::map< Element * , std::vector<NewPoint> > new_points;


};
#endif