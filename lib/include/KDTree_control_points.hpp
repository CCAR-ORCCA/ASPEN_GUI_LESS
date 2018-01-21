#ifndef HEADER_KDTree_control_points
#define HEADER_KDTree_control_points

#include <memory>
#include "ControlPoint.hpp"




// Implementation of a KDTree based on the
// very informative post found at
// http://andrewd.ces.clemson.edu/courses/cpsc805/references/nearest_search.pdf

class KDTree_control_points {

public:
	std::shared_ptr<KDTree_control_points> left;
	std::shared_ptr<KDTree_control_points> right;
	std::vector<std::shared_ptr<ControlPoint> > control_points;

	KDTree_control_points();

	std::shared_ptr<KDTree_control_points> build(std::vector< std::shared_ptr<ControlPoint> > & control_points, int depth);


	void closest_point_search(
	    const arma::vec & test_point,
	    std::shared_ptr<KDTree_control_points> node,
	    std::shared_ptr<ControlPoint> & best_guess,
	    double & distance);

	void closest_point_search(const arma::vec & test_point,
	                          std::shared_ptr<KDTree_control_points> node,
	                          std::shared_ptr<ControlPoint> & best_guess,
	                          double & distance,
	                          std::vector< std::shared_ptr<ControlPoint> > & closest_points);



	int get_depth() const;
	void set_depth(int depth);

	unsigned int get_size() const;

	std::vector<std::shared_ptr<ControlPoint> > * get_control_points();

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