#ifndef HEADER_PC
#define HEADER_PC

#include <armadillo>
#include "Lidar.hpp"
#include "KDTree_pc.hpp"

class PC {

public:

	PC(Lidar * lidar);

protected:

	std::vector<std::shared_ptr<arma::vec> > points;
	std::vector<std::shared_ptr<arma::vec> > normals;

	void construct_kd_tree();
	void construct_normals();

	KDTree_pc kd_tree;
};


#endif