#ifndef HEADER_KDTreePC_simple
#define HEADER_KDTreePC_simple

#include <memory>
#include <vector>
#include <armadillo>
class PC;

// Implementation of a KDTree based on the
// very informative post found at
// http://andrewd.ces.clemson.edu/courses/cpsc805/references/nearest_search.pdf

class KDTreePC {

public:
	std::shared_ptr<KDTreePC> left;
	std::shared_ptr<KDTreePC> right;
	std::vector<int> indices;

	KDTreePC(PC * owning_pc);

	void build(const std::vector< int > & indices, int depth) ;

	void closest_point_search(const arma::vec & test_point,
	const std::shared_ptr<KDTreePC> & node,
	int & best_guess_index,
	double & distance) const;

	void closest_N_point_search(const arma::vec & test_point,
		const unsigned int & N_points,
		const std::shared_ptr<KDTreePC> & node,
		double & distance,
		std::map<double, int > & closest_points) const;

	void radius_point_search(const arma::vec & test_point,
		const std::shared_ptr<KDTreePC> & node,
		const double & distance,
		std::vector< int > & closest_points);


	int get_depth() const;
	void set_depth(int depth);

	unsigned int size() const;

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

	PC * owning_pc;

};


#endif