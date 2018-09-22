#ifndef HEADER_KDTreePC
#define HEADER_KDTreePC

#include <memory>
#include "PointNormal.hpp"




// Implementation of a KDTree based on the
// very informative post found at
// http://andrewd.ces.clemson.edu/courses/cpsc805/references/nearest_search.pdf

class KDTreePC {

public:
	std::shared_ptr<KDTreePC> left;
	std::shared_ptr<KDTreePC> right;
	std::vector<std::shared_ptr<PointNormal> > points_normals;

	KDTreePC();

	std::shared_ptr<KDTreePC> build(std::vector< std::shared_ptr<PointNormal> > & points_normals, int depth);

	std::shared_ptr<PointNormal> get_normal_coordinates(const int & i) const;

	void closest_point_search(
		const arma::vec & test_point,
		const std::shared_ptr<KDTreePC> & node,
		std::shared_ptr<PointNormal> & best_guess,
		double & distance);

	void closest_N_point_search(const arma::vec & test_point,
		const unsigned int & N_points,
		const std::shared_ptr<KDTreePC> & node,
		double & distance,
		std::map<double,std::shared_ptr<PointNormal> > & closest_points);


	void radius_point_search(const arma::vec & test_point,
		const std::shared_ptr<KDTreePC> & node,
		const double & distance,
		std::vector<std::shared_ptr<PointNormal> > & closest_points);

	int get_depth() const;
	void set_depth(int depth);

	unsigned int get_size() const;

	std::vector<std::shared_ptr<PointNormal> > * get_points_normals();

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