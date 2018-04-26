#ifndef HEADER_BUNDLE_ADJUSTER
#define HEADER_BUNDLE_ADJUSTER
#include <armadillo>
#include <memory>
#include "PC.hpp"

class BundleAdjuster {

public:
	BundleAdjuster(std::vector< std::shared_ptr<PC> > * all_registered_pc_);

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

	void find_point_cloud_pairs();
	void save_connectivity_matrix() const;

	void assemble_subproblem(arma::mat & Lambda_k,arma::vec & N_k,const PointCloudPair & point_cloud_pair);

	void add_subproblem_to_problem(arma::sp_mat & Lambda,arma::vec & N,const arma::mat & Lambda_k,const arma::vec & N_k,
		const PointCloudPair & point_cloud_pair);

	void apply_deviation(const arma::vec & dX);


	void solve_bundle_adjustment();


	void update_point_cloud_pairs();

	void find_good_pairs(const std::vector< PointCloudPair > & all_point_cloud_pairs);




};






#endif