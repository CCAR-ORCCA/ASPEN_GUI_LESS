#ifndef HEADER_ICP
#define HEADER_ICP
#include <armadillo>
#include <memory>
#include "PC.hpp"
#include "CustomException.hpp"
#include "OMP_flags.hpp"
#include "DebugFlags.hpp"

#pragma omp declare reduction (+ : arma::mat::fixed<6,6> : omp_out += omp_in)\
initializer( omp_priv = omp_orig )
#pragma omp declare reduction (+ : arma::vec::fixed<6> : omp_out += omp_in)\
initializer( omp_priv = omp_orig )

typedef typename std::pair<std::shared_ptr<PointNormal>, std::shared_ptr<PointNormal> > PointPair ;


class ICP {
public:
	ICP(std::shared_ptr<PC> pc_destination, std::shared_ptr<PC> pc_source,
		arma::mat dcm_0 = arma::eye<arma::mat>(3, 3),
		arma::vec X_0 = arma::zeros<arma::vec>(3),
		bool verbose = true);

	arma::vec get_X() const;
	arma::mat get_M() const;
	arma::mat get_R() const;
	double get_J_res() const;

	std::vector<PointPair > * get_point_pairs();

	static void compute_pairs(
		std::vector<PointPair> & point_pairs,
		std::shared_ptr<PC> source_pc,
		std::shared_ptr<PC> destination_pc, 
		int h,
		const arma::mat & dcm_S = arma::eye<arma::mat>(3, 3),
		const arma::vec & x_S = arma::zeros<arma::vec>(3),
		const arma::mat & dcm_D = arma::eye<arma::mat>(3, 3),
		const arma::vec & x_D = arma::zeros<arma::vec>(3));
	
	static  double compute_rms_residuals(
		const std::vector<PointPair> & point_pairs,
		const arma::mat & dcm_S = arma::eye<arma::mat>(3, 3),
		const arma::vec & x_S = arma::zeros<arma::vec>(3),
		const arma::mat & dcm_D = arma::eye<arma::mat>(3, 3),
		const arma::vec & x_D = arma::zeros<arma::vec>(3));

	static double compute_mean_residuals(
		const std::vector<PointPair> & point_pairs,
		const arma::mat & dcm_S = arma::eye<arma::mat>(3, 3),
		const arma::vec & x_S = arma::zeros<arma::vec>(3),
		const arma::mat & dcm_D = arma::eye<arma::mat>(3, 3),
		const arma::vec & x_D = arma::zeros<arma::vec>(3));

	static double compute_normal_distance(const PointPair & point_pair, 
		const arma::mat & dcm_S = arma::eye<arma::mat>(3, 3),
		const arma::vec & x_S = arma::zeros<arma::vec>(3),
		const arma::mat & dcm_D = arma::eye<arma::mat>(3, 3),
		const arma::vec & x_D = arma::zeros<arma::vec>(3));

	static arma::rowvec dGdSigma_multiplicative(const arma::vec & mrp, const arma::vec & P, const arma::vec & n);
	
protected:
	std::shared_ptr<PC> pc_destination;
	std::shared_ptr<PC> pc_source;

	void register_pc_mrp_multiplicative_partials(
		const unsigned int iterations_max,
		const double rel_tol,
		const double stol,
		arma::mat dcm_0 ,
		arma::vec X_0,
		bool verbose = true);


	double compute_rms_residuals(
		const arma::mat & dcm,
		const arma::vec & x) ;

	void compute_pairs(
		int h,
		const arma::mat & dcm = arma::eye<arma::mat>(3,3),
		const arma::vec & x = arma::zeros<arma::vec>(3));

	
	arma::vec X;
	arma::mat DCM;
	arma::mat R;
	double J_res;

	std::vector<PointPair> point_pairs;


};






#endif