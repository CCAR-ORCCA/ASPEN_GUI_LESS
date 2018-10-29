#ifndef HEADER_BUNDLE_ADJUSTER
#define HEADER_BUNDLE_ADJUSTER
#include <armadillo>
#include <memory>

#include <Eigen/Sparse>
#include <Eigen/Jacobi>
#include <Eigen/Dense>
#include <Adjacency_List.hpp>

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
typedef Eigen::Triplet<double> T;
typedef Eigen::VectorXd EigVec;

class PointNormal;

template <class PointType> class PointCloud;

class BundleAdjuster {

public:

	
	BundleAdjuster(
		std::vector< std::shared_ptr<PointCloud<PointNormal > > > * all_registered_pc_, 
		int N_iter,
		int h,
		const arma::mat & LN_t0,
		const arma::vec & x_t0,
		std::string dir);


	BundleAdjuster(
		std::vector< std::shared_ptr<PointCloud<PointNormal > > > * all_registered_pc_,
		const arma::mat & LN_t0,
		const arma::vec & x_t0,
		std::string dir);


	void run(std::map<int,arma::mat> & M_pcs,
		std::map<int,arma::vec> & X_pcs,
		std::vector<arma::mat> & BN_measured,
		const std::vector<arma::vec> & mrps_LN,
		bool save_connectivity,
		int & previous_closure_index);

	struct PointCloudPair {
		int S_k = -1;
		int D_k;
		double error;
		double N_accepted_pairs;
		double N_pairs;

	};

	int get_cutoff_index() const;

	void set_use_true_pairs(bool use_true_pairs);


	void update_overlap_graph();

protected:
	

	std::vector< std::shared_ptr<PointCloud<PointNormal > > > * all_registered_pc;
	std::vector< PointCloudPair > point_cloud_pairs;

	std::map<double,int> find_overlap_with_pc(int pc_global_index,int start_index,int end_index,
		bool prune_overlaps = true) const;
	void save_connectivity() const;

	void assemble_subproblem(arma::mat & Lambda_k,arma::vec & N_k,const PointCloudPair & point_cloud_pair);

	void add_subproblem_to_problem(
		std::vector<T>& coeffs,
		EigVec & N,
		const arma::mat & Lambda_k,
		const arma::vec & N_k,
		const PointCloudPair & point_cloud_pair);

	void apply_deviation(const EigVec & deviation);

	void solve_bundle_adjustment();


	void create_pairs( int & previous_closure_index);

	void update_point_cloud_pairs();
	void update_point_clouds(std::map<int,arma::mat> & M_pcs, 
		std::map<int,arma::vec> & X_pcs,
		std::vector<arma::mat> & BN_measured,
		const std::vector<arma::vec> & mrps_LN);

	arma::mat LN_t0;
	arma::vec x_t0;
	arma::vec X;

	int ground_pc_index = 0;

	bool use_true_pairs = false;

	int closure_index = 0;
	int h;
	int N_iter;

	std::string dir;

	Adjacency_List<int,double> graph;


};






#endif