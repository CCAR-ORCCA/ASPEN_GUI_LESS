#ifndef HEADER_KDTree_pc
#define HEADER_KDTree_pc

#include <memory>
#include "PointNormal.hpp"




// Implementation of a KDTree based on the
// very informative post found at
// http://andrewd.ces.clemson.edu/courses/cpsc805/references/nearest_search.pdf

class KDTree_pc {

public:
	std::shared_ptr<KDTree_pc> left;
	std::shared_ptr<KDTree_pc> right;
	std::vector<std::shared_ptr<PointNormal> > points_normals;

	KDTree_pc();

	std::shared_ptr<KDTree_pc> build(std::vector< std::shared_ptr<PointNormal> > & points_normals, int depth, bool verbose = false);


	void closest_point_search(
	    const arma::vec & test_point,
	    std::shared_ptr<KDTree_pc> node,
	    std::shared_ptr<PointNormal> & best_guess,
	    double & distance);

	void closest_point_search(const arma::vec & test_point,
	                          std::shared_ptr<KDTree_pc> node,
	                          std::shared_ptr<PointNormal> & best_guess,
	                          double & distance,
	                          std::vector< std::shared_ptr<PointNormal> > & closest_points);



	int get_depth() const;
	void set_depth(int depth);

	unsigned int get_size() const;

	std::vector<std::shared_ptr<PointNormal> > * get_points_normals();

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