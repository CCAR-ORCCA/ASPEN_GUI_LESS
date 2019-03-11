#ifndef HEADER_ITERATIVE_CLOSEST_POINT
#define HEADER_ITERATIVE_CLOSEST_POINT

#include "ICPBase.hpp"


class IterativeClosestPoint : public ICPBase {

public:
	IterativeClosestPoint();
	

	static void compute_pairs(
		const PC &  source_pc,
		const PC &  destination_pc, 
		std::vector<PointPair> & point_pairs,
		int h,
		const arma::mat::fixed<3,3> & dcm_S = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_S = arma::zeros<arma::vec>(3),
		const arma::mat::fixed<3,3> & dcm_D = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_D = arma::zeros<arma::vec>(3));
	

	virtual double compute_distance(
		const PC &  source_pc,
		const PC &  destination_pc, 
		const PointPair & point_pair, 
		const arma::mat::fixed<3,3> & dcm_S = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_S = arma::zeros<arma::vec>(3),
		const arma::mat::fixed<3,3> & dcm_D = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_D = arma::zeros<arma::vec>(3)) const;
	
	
	virtual void compute_pairs(
		const PC & source_pc,
		const PC & destination_pc,
		int h,
		const arma::mat::fixed<3,3> & dcm = arma::eye<arma::mat>(3,3),
		const arma::vec::fixed<3> & x = arma::zeros<arma::vec>(3));

protected:
	

	virtual void build_matrices(
		const PC & source_pc,
		const PC & destination_pc,
		const int pair_index,
		const arma::vec::fixed<3> & mrp, 
		const arma::vec::fixed<3> & x,
		arma::mat::fixed<6,6> & info_mat_temp,
		arma::vec::fixed<6> & normal_mat_temp,
		arma::vec & residual_vector,
		arma::vec & sigma_vector,
		const double & w,
		const double & los_noise_sd_baseline,
		const arma::mat::fixed<3,3> & M_pc_D);



};






#endif