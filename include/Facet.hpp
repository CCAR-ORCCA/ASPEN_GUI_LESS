#ifndef HEADER_FACET
#define HEADER_FACET
#include <armadillo>
#include "Edge.hpp"
#include "Vertex.hpp"
#include <memory>

#include <set>

class Edge;
class Vertex;

class Facet {

public:

	/**
	Constructor
	@param vertices pointer to vector storing the vertices owned by this facet
	*/
	Facet(std::shared_ptr< std::vector<std::shared_ptr<Vertex > > > vertices);

	
	/**
	Get neighbors
	@param if false, only return neighbors sharing an edge. Else, returns all neighbords
	@return Pointer to neighboring facets, plus the calling facet
	*/
	std::set < Facet * > get_neighbors(bool all_neighbors) const;

	/**
	Add an edge to this facet
	@param Pointer to the edge to be added
	*/
	void add_edge(Edge * edge);


	/**
	Removes the specified edge from the 
	facet. The edge has to be present in the facet
	for this to work
	@param Pointer to the edge to be removed
	*/
	void remove_edge(Edge * edge);

	/**
	Get outbound facet normal
	@return pointer to facet normal
	*/
	arma::vec * get_facet_normal()  ;

	/**
	Get facet dyad
	@return pointer to facet dyad
	*/
	arma::mat * get_facet_dyad() ;

	/**
	Checks the degeneracy of the facet
	@return True if facet is not degenerate, false otherwise
	*/
	bool has_good_quality() const;


	/**
	Not implemented
	*/
	std::vector<std::shared_ptr <Edge> > get_facet_edges() const;


	/**
	Returns pointer to the first vertex owned by $this that is
	neither $v0 and $v1. When $v0 and $v1 are on the same edge,
	this method returns a pointer to the vertex of $this that is not
	on the edge but still owned by $this
	@param v0 Pointer to first vertex to exclude
	@param v1 Pointer to first vertex to exclude
	@return Pointer to the first vertex of $this that is neither $v0 and $v1
	*/
	std::shared_ptr<Vertex> vertex_not_on_edge(std::shared_ptr<Vertex> v0,
	        std::shared_ptr<Vertex>v1) const ;

	/**
	Recomputes the facet dyads, normal, surface area and center
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
	std::vector<std::shared_ptr<Vertex > > * get_vertices() ;


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
	facet where split
	*/
	unsigned int get_split_count() const;

protected:

	void compute_facet_dyad();
	void compute_normal();
	void compute_area();
	void compute_facet_center();


	std::shared_ptr< std::vector<std::shared_ptr<Vertex > > > vertices ;
	std::shared_ptr<arma::mat> facet_dyad;
	std::shared_ptr<arma::vec> facet_normal;
	std::shared_ptr<arma::vec> facet_center;

	std::set<Edge * > facet_edges;

	double area;
	unsigned int split_counter = 0;

};
#endif