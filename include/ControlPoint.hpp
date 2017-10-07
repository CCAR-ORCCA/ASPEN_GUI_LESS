#ifndef HEADER_VERTEX
#define HEADER_VERTEX
#include <armadillo>
#include "Element.hpp"
#include <memory>

#include <set>

class Element;
class ControlPoint {

public:

	/**
	Getter to the vertex's coordinates
	@return coordinates vertex coordinates
	*/
	arma::vec * get_coordinates() ;

	/**
	Setter to the vertex's coordinates
	@param coordinates vertex coordinates
	*/
	void set_coordinates(std::shared_ptr<arma::vec> coordinates);


	/**
	Adds $facet to the vector of std::shared_ptr<Element>  that own this vertex.
	Nothing happens if the facet is already listed
	@param facet Pointer to the facet owning this vertex
	*/
	void add_ownership(Element *  el);


	/**
	Finds the facets owming both $this and $vertex
	@return commons_facets Vector of Facet * owning the two vertices
	*/
	std::set< Element * >  common_facets(std::shared_ptr<ControlPoint> vertex) const;

	/**
	Determines if $this is owned by $facet
	@param facet Facet whose relationship with the facet is to be tested
	@return true is $this is owned by $facet, false otherwise
	*/
	bool is_owned_by( Element *  el) const;


	/**
	Delete $facet from the list of Element * owning $this
	Nothing happens if the facet was not listed (maybe throw a warning)>
	@param facet Pointer to the facet owning this vertex
	*/
	void remove_ownership(Element *  el);


	/**
	Returns the owning facets
	@return Owning facets
	*/
	std::set< Element *  > get_owning_elements() const;




	/**
	Returns the number of facets owning this vertex
	@return N number of owning of facets
	*/
	unsigned int get_number_of_owning_elements() const ;

protected:
	std::shared_ptr<arma::vec> coordinates;
	std::set<Element * > owning_elements;




};


#endif