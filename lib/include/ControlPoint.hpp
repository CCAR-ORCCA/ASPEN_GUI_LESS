#ifndef HEADER_CONTROL_POINT
#define HEADER_CONTROL_POINT
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
	arma::vec get_coordinates() const;


	/**
	Getter to the vertex's mean coordinates
	@return coordinates vertex mean coordinates
	*/
	arma::vec get_mean_coordinates() const;

	/**
	Setter to the vertex's coordinates
	@param coordinates vertex coordinates
	*/
	void set_coordinates(arma::vec coordinates);


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
	Returns pointer to coordinates
	@return pointer to coordinates
	*/
	double * get_coordinates_pointer();

	/**
	Returns pointer to coordinates
	@return pointer to coordinates
	*/
	arma::vec * get_coordinates_pointer_arma();


	/**
	Delete $facet from the list of Element * owning $this
	Nothing happens if the facet was not listed (maybe throw a warning)>
	@param facet Pointer to the facet owning this vertex
	*/
	void remove_ownership(Element *  el);

	/**
	Removes all ownership relationships 
	*/
	void reset_ownership();


	/**
	Returns the owning facets
	@return Owning facets
	*/
	std::set< Element *  > get_owning_elements() const;


	/**
	Copies the current coordinates into a vector of mean coordinates. To be called before running
	Monte Carlo simulations
	*/
	void set_mean_coordinates();

	/**
	Returns point covariance
	@return point covariance
	*/
	arma::mat get_covariance() const;

	/**
	@param element owning element
	@param local_indices triplet of indices numbering this control point within the owning element
	*/
	void add_local_numbering(Element * element,const arma::uvec & local_indices);

	/**
	Returns the local numbering of this control point within the specified element
	@param element pointer to element to consider
	*/
	arma::uvec get_local_numbering(Element * element) const;

	/**
	Sets the control point covariance
	@param P covariance
	*/
	void set_covariance(arma::mat P);

	void set_deviation(arma::vec d) { this -> deviation = d;}
	arma::vec get_deviation() const { return this -> deviation; }

	/**
	Returns the number of facets owning this vertex
	@return N number of owning of facets
	*/
	unsigned int get_number_of_owning_elements() const ;

	/**
	Get global index (shape wise)
	@return global index
	*/
	int get_global_index() const;


	/**
	Set global index (shape wise)
	@param global index
	*/
	void set_global_index(int index);

protected:
	arma::vec coordinates;
	arma::vec mean_coordinates;

	std::set<Element * > owning_elements;
	std::map<Element *,arma::uvec> local_numbering;
	arma::mat covariance = arma::zeros<arma::mat>(3,3);
	arma::vec deviation = arma::zeros<arma::vec>(3);

	int global_index;

};


#endif