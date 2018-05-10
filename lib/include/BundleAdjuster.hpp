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

	BundleAdjuster(arma::vector<arma::mat> & M_pcs, 
		arma::vector<arma::vec> & X_pcs,
		std::vector< std::shared_ptr<PC> > * all_registered_pc_, 
		int N_iter,
		const arma::mat & LN_t0,
		const arma::vec & x_t0,
		bool save_connectivity);


	BundleAdjuster(std::vector< std::shared_ptr<PC> > * all_registered_pc_,
		int N_iter,
		FlyOverMap * fly_over_map,
		arma::mat & longitude_latitude,
		const arma::mat & LN_t0 = arma::eye<arma::mat>(3,3),
		const arma::vec & x_t0 = arma::zeros<arma::vec>(3),
		bool save_connectivity = true,
		bool look_for_closure = true);

	struct PointCloudPair {
		int S_k = -1;
		int D_k;
		double error;
		double N_accepted_pairs;
		double N_pairs;

	};

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
	void update_point_clouds(arma::vector<arma::mat> * M_pcs, arma::vector<arma::vec> & X_pcs);

	arma::mat LN_t0;
	arma::vec x_t0;
	arma::vec X;

	int ground_pc_index = 0;

	FlyOverMap * fly_over_map;

	// Holds the correction to apply to the considered rigid transforms
	std::vector<arma::mat> rotation_increment;
	std::vector<arma::vec> position_increment;


};






#endif