#ifndef HEADER_ITERATIVE_CLOSEST_POINT
#define HEADER_ITERATIVE_CLOSEST_POINT

#include "ICPBase.hpp"


class IterativeClosestPoint : public ICPBase {

public:
	IterativeClosestPoint();
	IterativeClosestPoint(std::shared_ptr<PC> pc_destination, 
		std::shared_ptr<PC> pc_source) ;

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

	
	static double compute_distance(const PointPair & point_pair, 
		const arma::mat::fixed<3,3> & dcm_S ,
		const arma::vec::fixed<3> & x_S ,
		const arma::mat::fixed<3,3> & dcm_D ,
		const arma::vec::fixed<3> & x_D,
		const PC * pc_S,
		const PC * pc_D);

	static void ransac(
		const std::vector<PointPair> & all_pairs,
		int N_feature_pairs,
		int minimum_N_icp_pairs,
		double residuals_threshold,
		int N_iter_ransac,
		std::shared_ptr<PC> pc_source,
		std::shared_ptr<PC> pc_destination);

	
	virtual void compute_pairs(
		int h,
		const arma::mat::fixed<3,3> & dcm = arma::eye<arma::mat>(3,3),
		const arma::vec::fixed<3> & x = arma::zeros<arma::vec>(3));

protected:
	

	virtual void build_matrices(const int pair_index,
		const arma::vec::fixed<3> & mrp, 
		const arma::vec::fixed<3> & x,
		arma::mat::fixed<6,6> & info_mat_temp,
		arma::vec::fixed<6> & normal_mat_temp,
		const double & w);



};






#endif