#ifndef HEADER_ICP_BASE
#define HEADER_ICP_BASE
#include <armadillo>
#include <memory>
#include "PC.hpp"
#include "CustomException.hpp"
#include "OMP_flags.hpp"
#include "DebugFlags.hpp"
#include <RigidBodyKinematics.hpp>

class ICPBase {
public:

	ICPBase(std::shared_ptr<PC> pc_destination, std::shared_ptr<PC> pc_source);





	arma::vec::fixed<3> get_x() const;
	arma::mat::fixed<3,3> get_dcm() const;
	arma::mat::fixed<6,6> get_R() const;
	
	double get_J_res() const;

	std::vector<PointPair > * get_point_pairs();

	double compute_rms_residuals(
		const std::vector<PointPair> & point_pairs,
		const arma::mat::fixed<3,3> & dcm_S = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_S = arma::zeros<arma::vec>(3),
		const arma::vec & weights = {},
		const arma::mat::fixed<3,3> & dcm_D = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_D = arma::zeros<arma::vec>(3)) const;

	double compute_mean_residuals(
		const std::vector<PointPair> & point_pairs,
		const arma::mat::fixed<3,3> & dcm_S = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_S = arma::zeros<arma::vec>(3),
		const arma::vec & weights = {},
		const arma::mat::fixed<3,3> & dcm_D = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_D = arma::zeros<arma::vec>(3)) const;


	void register_pc(
		const double rel_tol,
		const double stol,
		const arma::mat::fixed<3,3>  & dcm_0,
		const arma::vec::fixed<3> &  X_0);
	
	void register_pc(
		const arma::mat::fixed<3,3> & dcm_0 = arma::eye<arma::mat>(3,3),
		const arma::vec::fixed<3> & X_0  = arma::zeros<arma::vec>(3));

	void register_pc_RANSAC(double fraction_inliers_used,
		double fraction_inliers_requested,
		unsigned int iter_ransac_max,
		const arma::mat::fixed<3,3> & dcm_0 = arma::eye<arma::mat>(3,3),
		const arma::vec::fixed<3> & X_0  = arma::zeros<arma::vec>(3));


	void register_pc_bf(unsigned int iter_bf_max,
		int N_possible_matches,
		int N_tentative_source_points,
		const arma::mat::fixed<3,3>  & dcm_0 = arma::eye<arma::mat>(3,3),
		const arma::vec::fixed<3>  & X_0 = arma::zeros<arma::vec>(3));

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


	bool get_use_FPFH() const;
	void set_use_FPFH(bool use_FPFH);

	void set_keep_correlations( bool keep_correlations);
	bool get_keep_correlation()  const;

	unsigned int get_minimum_h() const;
	void set_minimum_h(unsigned int min_h);


	unsigned int get_maximum_h() const;
	void set_maximum_h(unsigned int max_h);

	unsigned int get_N_bins() const;
	void set_N_bins(unsigned int N_bins);

	double get_neighborhood_radius() const;
	void set_neighborhood_radius(double neighborhood_radius);
	
	virtual double compute_distance(const PointPair & point_pair, 
		const arma::mat::fixed<3,3> & dcm_S = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_S = arma::zeros<arma::vec>(3),
		const arma::mat::fixed<3,3> & dcm_D = arma::eye<arma::mat>(3, 3),
		const arma::vec::fixed<3> & x_D = arma::zeros<arma::vec>(3)) const = 0;

protected:
	std::shared_ptr<PC> pc_destination;
	std::shared_ptr<PC> pc_source;

	double static compute_Huber_loss(const arma::vec & y, double threshold);

	arma::vec compute_y_vector(const std::vector<PointPair> & point_pairs,
		const arma::mat::fixed<3,3> & dcm_S ,
		const arma::vec::fixed<3> & x_S) const ;

	double compute_rms_residuals(
		const arma::mat::fixed<3,3> & dcm,
		const arma::vec::fixed<3> & x,
		const arma::vec & weights = {});

	double compute_mean_residuals(
		const arma::mat::fixed<3,3> & dcm,
		const arma::vec::fixed<3> & x,
		const arma::vec & weights = {});

	virtual void compute_pairs(
		int h,
		const arma::mat::fixed<3,3> & dcm = arma::eye<arma::mat>(3,3),
		const arma::vec::fixed<3> & x = arma::zeros<arma::vec>(3))= 0;

	virtual void build_matrices(const int pair_index,const arma::vec::fixed<3> & mrp, 
		const arma::vec::fixed<3> & x,arma::mat::fixed<6,6> & info_mat_temp,
		arma::vec::fixed<6> & normal_mat_temp,const double & w) = 0;

	void pca_prealignment(arma::vec::fixed<3> & mrp,arma::vec::fixed<3> & x) const;
	
	static void save_pairs(const std::vector<PointPair> & pairs,
		std::string path,
		std::shared_ptr<PC> pc_source, 
		std::shared_ptr<PC> pc_destination,
		const arma::mat::fixed<3,3> & dcm = arma::eye<arma::mat>(3,3),
		const arma::vec::fixed<3> & x = arma::zeros<arma::vec>(3));


	arma::vec weigh_ransac_pairs(const std::vector<PointPair> & matched_pairs,double radius);

	double compute_point_weight(const std::shared_ptr<PC> & origin_pc, const int & origin_point,
		const std::shared_ptr<PC> & target_pc,
		const double & radius) const;

	arma::vec::fixed<3> x = arma::zeros<arma::vec>(3);
	arma::vec::fixed<3> mrp = arma::zeros<arma::vec>(3);

	arma::vec::fixed<3> x_save = arma::zeros<arma::vec>(3);
	arma::mat::fixed<3,3> dcm_save = arma::eye<arma::mat>(3,3);

	arma::mat::fixed<6,6> R;
	
	double J_res;
	double rel_tol = 1e-6;
	double s_tol = 1e-2;
	double neighborhood_radius = -1;

	unsigned int iterations_max = 30;
	unsigned int minimum_h = 0;
	unsigned int maximum_h = 7;
	unsigned int N_bins = 3;
	
	bool use_true_pairs = false;
	bool use_pca_prealignment = false;
	bool keep_correlations = true;
	bool use_FPFH = false;


	std::vector<PointPair> point_pairs;


};






#endif