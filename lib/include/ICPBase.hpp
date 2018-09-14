#ifndef HEADER_ICP_BASE
#define HEADER_ICP_BASE
#include <armadillo>
#include <memory>
#include "PC.hpp"
#include "CustomException.hpp"
#include "OMP_flags.hpp"
#include "DebugFlags.hpp"
#include <RigidBodyKinematics.hpp>

typedef typename std::pair<std::shared_ptr<PointNormal>, std::shared_ptr<PointNormal> > PointPair ;

class ICPBase {
public:

	ICPBase(std::shared_ptr<PC> pc_destination, std::shared_ptr<PC> pc_source);

	arma::vec::fixed<3> get_x() const;
	arma::mat::fixed<3,3> get_dcm() const;
	arma::mat::fixed<6,6> get_R() const;
	
	double get_J_res() const;

	std::vector<PointPair > * get_point_pairs();

	static  double compute_rms_residuals(
		const std::vector<PointPair> & point_pairs,
		const arma::mat::fixed<3,3> & dcm_S = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_S = arma::zeros<arma::vec>(3),
		const arma::mat::fixed<3,3> & dcm_D = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_D = arma::zeros<arma::vec>(3));

	static double compute_mean_residuals(
		const std::vector<PointPair> & point_pairs,
		const arma::mat::fixed<3,3> & dcm_S = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_S = arma::zeros<arma::vec>(3),
		const arma::mat::fixed<3,3> & dcm_D = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_D = arma::zeros<arma::vec>(3));

	static double compute_normal_distance(const PointPair & point_pair, 
		const arma::mat::fixed<3,3> & dcm_S = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_S = arma::zeros<arma::vec>(3),
		const arma::mat::fixed<3,3> & dcm_D = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_D = arma::zeros<arma::vec>(3));
	
	void register_pc(
		arma::mat::fixed<3,3> dcm_0 = arma::eye<arma::mat>(3,3),
		arma::vec::fixed<3> X_0  = arma::zeros<arma::vec>(3));

	void set_use_true_pairs(bool use_true_pairs);
	void set_rel_tol(double rel_tol);
	void set_s_tol(double s_tol);
	void set_iterations_max(unsigned int iterations_max);

	bool get_use_true_pairs() const;
	double get_rel_tol() const;
	double get_s_tol() const;
	unsigned int get_iterations_max() const;


	void set_save_rigid_transform(const arma::vec::fixed<3> & x_save,
		const arma::mat::fixed<3,3> & dcm_save);


	bool get_use_pca_prealignment() const;
	void set_use_pca_prealignment(bool use_pca_prealignment);

	unsigned int get_minimum_h() const;
	void set_minimum_h(unsigned int min_h);

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

	virtual void build_matrices(const int pair_index,const arma::vec::fixed<3> & mrp, 
		const arma::vec::fixed<3> & x,arma::mat::fixed<6,6> & info_mat_temp,
		arma::vec::fixed<6> & normal_mat_temp) = 0;

	void pca_prealignment(arma::vec::fixed<3> & mrp,arma::vec::fixed<3> & x) const;

	arma::vec::fixed<3> x;
	arma::mat::fixed<3,3> dcm;

	arma::vec::fixed<3> x_save = arma::zeros<arma::vec>(3);
	arma::mat::fixed<3,3> dcm_save = arma::eye<arma::mat>(3,3);

	arma::mat::fixed<6,6> R;
	
	double J_res;
	double rel_tol = 1e-8;
	double s_tol = 1e-2;
	unsigned int iterations_max = 100;
	unsigned int minimum_h = 0;
	bool use_true_pairs = false;
	bool use_pca_prealignment = false;
	
	std::vector<PointPair> point_pairs;


};






#endif