#ifndef HEADER_KDTreeDescriptors
#define HEADER_KDTreeDescriptors

#include <memory>
#include "PointNormal.hpp"


// Implementation of a KDTree based on the
// very informative post found at
// http://andrewd.ces.clemson.edu/courses/cpsc805/references/nearest_search.pdf

class KDTreeDescriptors {

public:
	std::shared_ptr<KDTreeDescriptors> left;
	std::shared_ptr<KDTreeDescriptors> right;
	std::vector<std::shared_ptr<PointNormal> > points_with_descriptors;

	KDTreeDescriptors();

	std::vector<std::shared_ptr<PointNormal> > * get_points_with_descriptors() ;

	std::shared_ptr<KDTreeDescriptors> build(std::vector< std::shared_ptr<PointNormal> > & points, int depth);

	void closest_point_search(
		const std::shared_ptr<PointNormal> & query_point,
		std::shared_ptr<KDTreeDescriptors> node,
		std::shared_ptr<PointNormal> & best_guess,
		double & distance);


	void closest_N_point_search(const std::shared_ptr<PointNormal> & query_point,
		const unsigned int & N_points,
		std::shared_ptr<KDTreeDescriptors> node,
		double & distance,
		std::map<double,std::shared_ptr<PointNormal> > & closest_points) ;


	int get_depth() const;
	void set_depth(int depth);

	unsigned int get_size() const;

	double get_value() const;
	unsigned int get_axis() const;

	void set_value(double value) ;
	void set_axis(unsigned int axis);

	void set_is_cluttered(bool cluttered);
	bool get_is_cluttered() const;


protected:

	int depth;
	int max_depth = 1000;
	double value;
	unsigned int axis = 0;
	bool cluttered = false;

};


#endif