#ifndef HEADER_ICP
#define HEADER_ICP
#include <armadillo>
#include <memory>
#include "PC.hpp"
#include "CustomException.hpp"
#include "OMP_flags.hpp"
#include "DebugFlags.hpp"

typedef typename std::pair<std::shared_ptr<PointNormal>, std::shared_ptr<PointNormal> > PointPair ;


class ICP {
public:
	ICP(std::shared_ptr<PC> pc_destination, std::shared_ptr<PC> pc_source,
		bool verbose = true,
		const arma::mat & M_save = arma::eye<arma::mat>(3,3),
		const arma::vec & X_save = arma::zeros<arma::vec>(3));

	arma::vec get_x() const;
	arma::mat get_dcm() const;
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
	void register_pc(const double rel_tol = 1e-8,
		const double stol = 1e-2,
		arma::mat dcm_0 = arma::eye<arma::mat>(3,3),
		arma::vec X_0  = arma::zeros<arma::vec>(3),
		bool verbose = true);

	void set_use_true_pairs(bool use_true_pairs);

	void set_iterations_max(unsigned int iterations_max);



protected:
	std::shared_ptr<PC> pc_destination;
	std::shared_ptr<PC> pc_source;

	


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
	arma::mat M_save;
	arma::vec X_save;
	double J_res;
	bool use_true_pairs = false;

	std::vector<PointPair> point_pairs;
	unsigned int iterations_max = 100;

};






#endif