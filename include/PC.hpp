#ifndef HEADER_PC
#define HEADER_PC

#include <armadillo>
#include "KDTree_pc.hpp"
#include "Ray.hpp"
#include <memory>


class PC {

public:

	PC(arma::vec los_dir, std::vector<std::vector<std::shared_ptr<Ray> > > * focal_plane);

protected:

	std::vector<std::shared_ptr<arma::vec> > points;
	std::vector<std::shared_ptr<arma::vec> > normals;

	void construct_kd_tree();
	void construct_normals();

	KDTree_pc kd_tree;

};


#endif