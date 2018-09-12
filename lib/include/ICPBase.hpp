#ifndef HEADER_ICP_BASE
#define HEADER_ICP_BASE
#include <armadillo>
#include <memory>
#include "PC.hpp"
#include "CustomException.hpp"
#include "OMP_flags.hpp"
#include "DebugFlags.hpp"

typedef typename std::pair<std::shared_ptr<PointNormal>, std::shared_ptr<PointNormal> > PointPair ;



/
class ICPBase {
public:

	ICPBase(std::shared_ptr<PC> pc_destination, std::shared_ptr<PC> pc_source,
		bool verbose = true,
		const arma::mat::fixed<3,3> & M_save = arma::eye<arma::mat>(3,3),
		const arma::vec::fixed<3> & X_save = arma::zeros<arma::vec>(3));

	arma::vec::fixed<3> get_X() const;
	arma::mat::fixed<3,3> get_M() const;
	arma::mat::fixed<6,6> get_R() const;
	
	double get_J_res() const;

	std::vector<PointPair > * get_point_pairs();

	virtual static void compute_pairs(
		std::vector<PointPair> & point_pairs,
		std::shared_ptr<PC> source_pc,
		std::shared_ptr<PC> destination_pc, 
		int h,
		const arma::mat::fixed<3,3> & dcm_S = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_S = arma::zeros<arma::vec>(3),
		const arma::mat::fixed<3,3> & dcm_D = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_D = arma::zeros<arma::vec>(3)) = 0;
	
	virtual static  double compute_rms_residuals(
		const std::vector<PointPair> & point_pairs,
		const arma::mat::fixed<3,3> & dcm_S = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_S = arma::zeros<arma::vec>(3),
		const arma::mat::fixed<3,3> & dcm_D = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_D = arma::zeros<arma::vec>(3)) = 0;

	virtual static double compute_mean_residuals(
		const std::vector<PointPair> & point_pairs,
		const arma::mat::fixed<3,3> & dcm_S = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_S = arma::zeros<arma::vec>(3),
		const arma::mat::fixed<3,3> & dcm_D = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_D = arma::zeros<arma::vec>(3)) = 0;

	static double compute_normal_distance(const PointPair & point_pair, 
		const arma::mat::fixed<3,3> & dcm_S = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_S = arma::zeros<arma::vec>(3),
		const arma::mat::fixed<3,3> & dcm_D = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_D = arma::zeros<arma::vec>(3));

	static arma::rowvec dGdSigma_multiplicative(const arma::vec & mrp, const arma::vec & P, const arma::vec & n);
	
	virtual void register_pc(
		const unsigned int iterations_max,
		const double rel_tol,
		const double stol,
		arma::mat::fixed<3,3> dcm_0 = arma::eye<arma::mat>(3,3),
		arma::vec::fixed<3> X_0  = arma::zeros<arma::vec>(3),
		bool verbose = true) = 0;

	void set_use_true_pairs(bool use_true_pairs);
	void set_rel_tol(double rel_tol);
	void set_s_tol(double s_tol);
	void set_iterations_max(unsigned int iterations_max);



	bool get_use_true_pairs() const;
	double get_rel_tol() const;
	double get_s_tol() const;
	unsigned int get_iterations_max() const;


protected:
	std::shared_ptr<PC> pc_destination;
	std::shared_ptr<PC> pc_source;


	virtual double compute_rms_residuals(
		const arma::mat::fixed<3,3> & dcm,
		const arma::vec::fixed<3> & x) = 0;

	virtual void compute_pairs(
		int h,
		const arma::mat::fixed<3,3> & dcm = arma::eye<arma::mat>(3,3),
		const arma::vec::fixed<3> & x = arma::zeros<arma::vec>(3))= 0;

	
	arma::vec::fixed<3> X;
	arma::mat::fixed<3,3> DCM;
	arma::mat::fixed<6,6> R;
	arma::mat::fixed<3,3> M_save;
	arma::vec::fixed<3> X_save;
	
	double J_res;
	double rel_tol = 1e-8;
	double s_tol = 1e-2;
	unsigned int iterations_max = 100;

	bool use_true_pairs = false;
	std::vector<PointPair> point_pairs;


};






#endif