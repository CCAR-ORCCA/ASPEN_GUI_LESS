#ifndef HEADER_PC
#define HEADER_PC

#include <armadillo>
#include <memory>

#include "KDTree_pc.hpp"
#include "Ray.hpp"
#include "PointNormal.hpp"


class PC {

public:

	PC(arma::vec los_dir, std::vector<std::vector<std::shared_ptr<Ray> > > * focal_plane);
	PC(arma::vec los_dir, arma::mat & points);

	int get_closest_point_index_brute_force(arma::vec & test_point) const;
	arma::uvec get_closest_points_indices_brute_force(arma::vec & test_point, unsigned int N) const ;

	std::shared_ptr<PointNormal> get_closest_point(arma::vec & test_point) const;

protected:



	void construct_kd_tree(std::vector< std::shared_ptr<PointNormal> > & points_normals);
	void construct_normals();

	std::shared_ptr<KDTree_pc> kd_tree;

};


#endif