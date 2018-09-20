#ifndef HEADER_ITERATIVE_CLOSEST_POINT_TO_PLANE
#define HEADER_ITERATIVE_CLOSEST_POINT_TO_PLANE

#include "ICPBase.hpp"

class IterativeClosestPointToPlane : public ICPBase {

public:

	IterativeClosestPointToPlane(std::shared_ptr<PC> pc_destination, std::shared_ptr<PC> pc_source) ;

	static void compute_pairs(
		std::vector<PointPair> & point_pairs,
		std::shared_ptr<PC> source_pc,
		std::shared_ptr<PC> destination_pc, 
		int h,
		const arma::mat::fixed<3,3> & dcm_S = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_S = arma::zeros<arma::vec>(3),
		const arma::mat::fixed<3,3> & dcm_D = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_D = arma::zeros<arma::vec>(3));

	virtual double compute_distance(const PointPair & point_pair, 
		const arma::mat::fixed<3,3> & dcm_S = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_S = arma::zeros<arma::vec>(3),
		const arma::mat::fixed<3,3> & dcm_D = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_D = arma::zeros<arma::vec>(3)) const;

protected:
	


	virtual void compute_pairs(
		int h,
		const arma::mat::fixed<3,3> & dcm = arma::eye<arma::mat>(3,3),
		const arma::vec::fixed<3> & x = arma::zeros<arma::vec>(3));

	virtual void build_matrices(const int pair_index,const arma::vec::fixed<3> & mrp, 
		const arma::vec::fixed<3> & x,arma::mat::fixed<6,6> & info_mat_temp,
		arma::vec::fixed<6> & normal_mat_temp,const double & w);

};

#endif