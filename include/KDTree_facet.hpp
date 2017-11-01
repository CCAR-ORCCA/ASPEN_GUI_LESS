#ifndef HEADER_KDTree_facets
#define HEADER_KDTree_facets

#include <memory>
#include "Element.hpp"
#include "Facet.hpp"




// Implementation of a KDTree based on the
// very informative post found at
// http://andrewd.ces.clemson.edu/courses/cpsc805/references/nearest_search.pdf

class KDTree_facet {

public:
	std::shared_ptr<KDTree_facet> left;
	std::shared_ptr<KDTree_facet> right;
	std::vector<std::shared_ptr<Element> > facets;

	KDTree_facet();

	std::shared_ptr<KDTree_facet> build(std::vector< std::shared_ptr<Element> > & facets, int depth, bool verbose = false);


	void closest_point_search(
	    const arma::vec & test_point,
	    std::shared_ptr<KDTree_facet> node,
	    std::shared_ptr<Element> & best_guess,
	    double & distance);

	void closest_point_search(const arma::vec & test_point,
	                          std::shared_ptr<KDTree_facet> node,
	                          std::shared_ptr<Element> & best_guess,
	                          double & distance,
	                          std::vector< std::shared_ptr<Element> > & closest_points);



	int get_depth() const;
	void set_depth(int depth);

	unsigned int get_size() const;

	std::vector<std::shared_ptr<Element> > * get_facets();

	double get_value() const;
	unsigned int get_axis() const;

	void set_value(double value) ;
	void set_axis(unsigned int axis);


protected:

	int depth;
	int max_depth = 1000;
	double value;
	unsigned int axis;

};


#endif