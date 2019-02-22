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

	BundleAdjuster(double sigma_rho,
		std::vector< std::shared_ptr<PointCloud<PointNormal > > > * all_registered_pc_, 
		int N_iter,
		int h,
		arma::mat * LN_t0,
		arma::vec * x_t0,
		std::string dir,
		std::vector<arma::vec::fixed<3> > * mrps_LN,
		std::vector<arma::mat::fixed<3,3 > > * BN_measured);

	BundleAdjuster(double sigma_rho,
		std::vector< std::shared_ptr<PointCloud<PointNormal > > > * all_registered_pc_,
		arma::mat * LN_t0,
		arma::vec * x_t0,
		std::string dir);

	void run(std::map<int,arma::mat::fixed<3,3> > & M_pcs,
		std::map<int,arma::vec::fixed<3> > & X_pcs,
		std::map<int,arma::mat::fixed<6,6> > & R_pcs,
		std::vector<arma::mat::fixed<3,3> > & BN_measured,
		const std::vector<arma::vec::fixed<3> > & mrps_LN,
		bool save_connectivity,
		bool apply_deviation = true);


	void set_h(int h);


	struct PointCloudPair {
		int S_k = -1;
		int D_k;
		double error;
		double N_accepted_pairs;
		double N_pairs;

	};

	void set_use_true_pairs(bool use_true_pairs);

	/**
	Returns true if the update of the overlap graph has revealed a loop closure 
	between the latest point cloud and another point cloud more that cluster_size indices away
	@return true if bundle adjustment should be run, false otherwise
	*/
	bool update_overlap_graph();

	/**
	Sets the maximum number of point clouds in a given cluster
	*/
	void set_cluster_size(int size);


protected:

	void remove_edges_from_graph();
		

	std::vector< std::shared_ptr<PointCloud<PointNormal > > > * all_registered_pc;
	std::vector< PointCloudPair > point_cloud_pairs;

	std::map<double,int> find_overlap_with_pc(int pc_global_index,int start_index,int end_index,
		bool prune_overlaps = true) const;
	void save_connectivity() const;

	void assemble_subproblem(arma::mat & Lambda_k,arma::vec & N_k,
		const PointCloudPair & point_cloud_pair,
		const std::map<int,arma::mat::fixed<3,3> > & M_pcs,
		const std::map<int,arma::vec::fixed<3> > & X_pcs);

	void add_subproblem_to_problem(
		std::vector<T>& coeffs,
		EigVec & N,
		const arma::mat & Lambda_k,
		const arma::vec & N_k,
		const PointCloudPair & point_cloud_pair);

	void apply_deviation(const EigVec & deviation);

	void solve_bundle_adjustment(const std::map<int,arma::mat::fixed<3,3> > & M_pcs,
		const std::map<int,arma::vec::fixed<3> > & X_pcs,bool apply_deviation = true);


	void create_pairs();

	void update_point_cloud_pairs();
	void update_point_clouds(std::map<int,arma::mat::fixed<3,3> > & M_pcs, 
		std::map<int,arma::vec::fixed<3> > & X_pcs,
		std::map<int,arma::mat::fixed<6,6> > & R_pcs,
		std::vector<arma::mat::fixed<3,3> > & BN_measured,
		const std::vector<arma::vec::fixed<3> > & mrps_LN);

	arma::mat * LN_t0;
	arma::vec * x_t0;
	arma::vec X;

	bool use_true_pairs = false;

	int anchor_pc_index = 0;
	int next_anchor_pc_index = 0;

	int h;
	int N_iter;
	int cluster_size = 4;

	std::string dir;
	double sigma_rho;

	Adjacency_List<int,double> graph;
	MatrixXd Pdense;

	std::vector<std::set<int> >edges_to_remove;
	std::vector<arma::vec::fixed<3> > * mrp_LN_ptr;
	std::vector<arma::mat::fixed<3,3> > * BN_measured_ptr;


};






#endif