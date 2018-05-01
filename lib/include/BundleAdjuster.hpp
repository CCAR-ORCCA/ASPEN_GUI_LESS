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


class BundleAdjuster {

public:
	BundleAdjuster(std::vector< std::shared_ptr<PC> > * all_registered_pc_,int N_iter,arma::mat LN_t0,arma::vec x_t0,
		const arma::mat & longitude_latitude);

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

	void apply_deviation(const arma::vec & dX);

	void ICP_pass();

	void solve_bundle_adjustment();

	void create_pairs(const arma::mat & longitude_latitude);

	void update_point_cloud_pairs();

	arma::mat LN_t0;
	arma::vec x_t0;

	int ground_pc_index;



};






#endif