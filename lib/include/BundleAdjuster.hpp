#ifndef HEADER_BUNDLE_ADJUSTER
#define HEADER_BUNDLE_ADJUSTER
#include <armadillo>
#include <memory>
#include "PC.hpp"

#include <Eigen/Sparse>
#include <Eigen/Jacobi>
#include <Eigen/Dense>



typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
typedef Eigen::Triplet<double> T;
typedef Eigen::VectorXd EigVec;

class FlyOverMap;


class BundleAdjuster {

public:

	BundleAdjuster(int t0, 
		int tf, 
		std::map<int,arma::mat> & M_pcs, 
		std::map<int,arma::vec> & X_pcs,
		std::vector<arma::mat> & BN_estimated,
		std::vector< std::shared_ptr<PC> > * all_registered_pc_, 
		int N_iter,
		const arma::mat & LN_t0,
		const arma::vec & x_t0,
		const std::vector<arma::vec> & mrps_LN,
		bool save_connectivity);


	// BundleAdjuster(
	// 	std::vector<arma::mat> & M_pcs,
	// 	std::vector<arma::vec> & X_pcs,
	// 	std::vector< std::shared_ptr<PC> > * all_registered_pc_,
	// 	int N_iter,
	// 	FlyOverMap * fly_over_map,
	// 	arma::mat & longitude_latitude,
	// 	const arma::mat & LN_t0 = arma::eye<arma::mat>(3,3),
	// 	const arma::vec & x_t0 = arma::zeros<arma::vec>(3),
	// 	bool save_connectivity = true,
	// 	bool look_for_closure = true);

	struct PointCloudPair {
		int S_k = -1;
		int D_k;
		double error;
		double N_accepted_pairs;
		double N_pairs;

	};

	int get_cutoff_index() const;

protected:
	

	std::vector< std::shared_ptr<PC> > * all_registered_pc;
	std::vector< PointCloudPair > point_cloud_pairs;
	int N_iter;

	void save_connectivity() const;

	void assemble_subproblem(arma::mat & Lambda_k,arma::vec & N_k,const PointCloudPair & point_cloud_pair);

	void add_subproblem_to_problem(
		std::vector<T>& coeffs,
		EigVec & N,
		const arma::mat & Lambda_k,
		const arma::vec & N_k,
		const PointCloudPair & point_cloud_pair);

	void apply_deviation(const EigVec & deviation);

	void update_flyover_map(arma::mat & longitude_latitude);

	void solve_bundle_adjustment();


	void create_pairs(bool look_for_closure = false);

	void update_point_cloud_pairs();
	void update_point_clouds(std::map<int,arma::mat> & M_pcs, 
		std::map<int,arma::vec> & X_pcs,
		std::vector<arma::mat> & BN_estimated,
		const std::vector<arma::vec> & mrps_LN);

	arma::mat LN_t0;
	arma::vec x_t0;
	arma::vec X;

	int ground_pc_index = 0;

	FlyOverMap * fly_over_map;

	std::vector<int> local_pc_index_to_global_pc_index;


	int cutoff_index;
	int h = 5;
};






#endif