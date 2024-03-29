#include "BundleAdjuster.hpp"
#include <armadillo>
#include "IterativeClosestPointToPlane.hpp"
#include "boost/progress.hpp"
#include <PointCloud.hpp>
#include <PointNormal.hpp>
#include <set>
#include <PointCloudIO.hpp>

#define BUNDLE_ADJUSTER_DEBUG 1
#define IOFLAGS_bundle_adjuster 0

BundleAdjuster::BundleAdjuster(
	double sigma_rho,
	std::vector< PointCloud<PointNormal > >  * all_registered_pc_, 
	int N_iter,
	int h,
	arma::mat * LN_t0,
	arma::vec * x_t0,
	std::string dir,
	std::vector<arma::vec::fixed<3> > * mrps_LN,
	std::vector<arma::mat::fixed<3,3> > * BN_measured){

	this -> sigma_rho = sigma_rho;
	this -> all_registered_pc = all_registered_pc_;
	this -> LN_t0 = LN_t0;
	this -> x_t0 = x_t0;
	this -> N_iter = N_iter;
	this -> h = h;
	this -> dir = dir;
	this -> mrp_LN_ptr = mrps_LN;
	this -> BN_measured_ptr = BN_measured;

}

BundleAdjuster::BundleAdjuster(double sigma_rho,
	std::vector< PointCloud<PointNormal > >  * all_registered_pc_, 
	arma::mat * LN_t0,
	arma::vec * x_t0,
	std::string dir){

	this -> all_registered_pc = all_registered_pc_;
	this -> LN_t0 = LN_t0;
	this -> x_t0 = x_t0;
	this -> dir = dir;
	this -> sigma_rho = sigma_rho;

}



void BundleAdjuster::set_use_true_pairs(bool use_true_pairs){
	this -> use_true_pairs = use_true_pairs;
}



void BundleAdjuster::run(
	std::map<int,arma::mat::fixed<3,3> > & M_pcs,
	std::map<int,arma::vec::fixed<3> > & X_pcs,
	std::map<int,arma::mat::fixed<6,6> > & R_pcs,
	std::vector<arma::mat::fixed<3,3> > & BN_measured,
	const std::vector<arma::vec::fixed<3> > & mrps_LN){

	if (this -> all_registered_pc -> size() == 0){
		std::cout << " - Nothing to do here\n";
		return;
	}

	// There are Q - 1 bundle adjusted PC + one anchor PC
	int Q = this -> all_registered_pc -> size() - this -> anchor_pc_index;

	this -> X = arma::zeros<arma::vec>(6 * (Q - 1));



	if (this -> N_iter > 0){
		this -> create_pairs();






		// solve the bundle adjustment problem
		this -> solve_bundle_adjustment(M_pcs,X_pcs);

	}	


	std::cout << "- Updating point clouds ... " << std::endl;
	this -> update_point_clouds(M_pcs,X_pcs,R_pcs,BN_measured,mrps_LN);




	
	if (this -> anchor_pc_index !=  this -> next_anchor_pc_index){
		
		std::cout << "- Updating anchor_pc_index from " << this -> anchor_pc_index << " to " << this -> next_anchor_pc_index << std::endl;
		
		// std::cout << "- Saving local structure ... " << std::endl;
		this -> save_local_bundle();


		this -> previous_anchor_pc_index = this -> anchor_pc_index;
		this -> anchor_pc_index = this -> next_anchor_pc_index;

		
		// The new anchor pc is effectively replaced by the new local structure
		PointCloud<PointNormal> & destination_pc = this -> all_registered_pc -> back();

		for (auto iter = this -> all_registered_pc -> begin(); iter != (--this -> all_registered_pc -> end()); ++iter){

			for (int k = 0; k < iter -> size(); ++k){
				destination_pc . push_back(iter-> get_point(k));
			}
			iter -> clear();
			
		}

		destination_pc . build_kdtree(false);

	}

	std::cout << "- Leaving bundle adjustment" << std::endl;

}


void BundleAdjuster::solve_bundle_adjustment(
	const std::map<int,arma::mat::fixed<3,3> > & M_pcs,
	const std::map<int,arma::vec::fixed<3> > & X_pcs){
	

	int Q = this -> all_registered_pc -> size() - this -> anchor_pc_index;
	std::cout << "\t Number of considered point clouds (Q): " << Q << std::endl;

	// This allows to compute the ICP RMS residuals for each considered point-cloud pair before running the bundle adjuster
	bool has_converged = this -> update_point_cloud_pairs(false);

	for (int iter = 0 ; iter < this -> N_iter; ++iter){

		std::cout << "\tIteration: " << std::to_string(iter + 1) << " /" << std::to_string(N_iter) << std::endl;

		std::vector<T> coefficients;          
		
		EigVec Nmat(6 * (Q - 1)); 
		Nmat.setZero();
		SpMat Lambda(6 * (Q - 1), 6 * (Q - 1));
		
		std::vector<arma::mat> Lambda_k_vector;
		std::vector<arma::vec> N_k_vector;

		// The subproblem matrices are pre-allocated
		for (int k = 0; k < this -> point_cloud_pairs.size(); ++k){

			arma::mat Lambda_k;
			arma::vec N_k;

			assert(this -> point_cloud_pairs . at(k).D_k > this -> point_cloud_pairs . at(k).S_k);

			if (this -> point_cloud_pairs . at(k).D_k != this -> anchor_pc_index && this -> point_cloud_pairs . at(k).S_k != this -> anchor_pc_index){
				Lambda_k = arma::zeros<arma::mat>(12,12);
				N_k = arma::zeros<arma::vec>(12);
			}

			else{
				Lambda_k = arma::zeros<arma::mat>(6,6);
				N_k = arma::zeros<arma::vec>(6);
			}

			Lambda_k_vector.push_back(Lambda_k);
			N_k_vector.push_back(N_k);

		}

		// For each point-cloud pair
		#if !BUNDLE_ADJUSTER_DEBUG
		boost::progress_display progress(this -> point_cloud_pairs.size());
		#endif 

		#pragma omp parallel for
		for (int k = 0; k < this -> point_cloud_pairs.size(); ++k){
			
			// The Lambda_k and N_k specific to this point-cloud pair are computed
			this -> assemble_subproblem(Lambda_k_vector. at(k),N_k_vector. at(k),this -> point_cloud_pairs . at(k),M_pcs,X_pcs);
			#if !BUNDLE_ADJUSTER_DEBUG
			++progress;
			#endif

		}

		for (int k = 0; k < this -> point_cloud_pairs.size(); ++k){
			// They are added to the whole problem
			this -> add_subproblem_to_problem(coefficients,Nmat,Lambda_k_vector. at(k),N_k_vector. at(k),this -> point_cloud_pairs . at(k));
		}	

		
		std::cout << "\n- Solving for the deviation" << std::endl;

		// The deviation in all of the rigid transforms is computed
		Lambda.setFromTriplets(coefficients.begin(), coefficients.end());
		

		// Sparsity in information matrix
		std::cout << "- Lambda is filled at " << double(Lambda.nonZeros()) / (6 * (Q - 1) * 6 * (Q - 1)) * 100  <<  " %\n";

		// Switching to dense
		MatrixXd Lambda_dense(Lambda);

		// The deviation is computed
		EigVec deviation = Lambda_dense.colPivHouseholderQr().solve(Nmat);    

		// It is applied to all of the point clouds (minus the first one)
		std::cout << "\n- Applying the deviation" << std::endl;

		this -> apply_deviation(deviation);
		std::cout << "\n- Updating the point cloud pairs" << std::endl;

		// The point cloud pairs are updated: their residuals are updated
		// and the rigid transforms positioning them are also updated
		has_converged = this -> update_point_cloud_pairs(int(this -> N_iter) - 1 == iter );

		
		this -> create_pairs();

		if (has_converged){
			std::cout << "\n- All point-cloud pairs are satisfying. BA has converged \n";
			break;
		}

	}

	std::cout << "\n- Removing edges from graph \n";
	this -> remove_edges_from_graph();


}


void BundleAdjuster::create_pairs(){

	std::set<std::set<int> > pairs;
	this -> point_cloud_pairs.clear();
	std::set<int> vertices = this -> graph. get_vertices();

	// Need to pull the point cloud pairs from the bundle adjustment graph
	// Bundle adjustment only runs between point cloud #anchor_pc_index and the last registered pc

	for (int vertex : vertices){
		std::set<int> neighbors = this -> graph. getneighbors(vertex);
		for (auto neighbor : neighbors){
			std::set<int> pair = {vertex,neighbor};
			if (pair.size() == 2){
				pairs.insert(pair);
			}
		}
	}


	for (auto pair_set : pairs){
		BundleAdjuster::PointCloudPair pair;
		
		// S_k is always less than S_k because the set is ordered
		int S_k = *(pair_set.begin());
		int D_k = *(--pair_set.end());
		pair.S_k = S_k;
		pair.D_k = D_k;

		// If S_k is greater or equal than $anchor_pc_index then this point-cloud pair will 
		// be processed

		if (pair.S_k >= this -> anchor_pc_index)
			this -> point_cloud_pairs.push_back(pair);
	}



}

void BundleAdjuster::assemble_subproblem(arma::mat & Lambda_k,arma::vec & N_k,
	const PointCloudPair & point_cloud_pair,
	const std::map<int,arma::mat::fixed<3,3> > & M_pcs,
	const std::map<int,arma::vec::fixed<3> > & X_pcs){

	// The point-pairs in the prescribed point-cloud pair are formed (with h = 0, so we are using them all)
	std::vector<PointPair> point_pairs;

	// The point pairs must be computed using the current estimate of the point clouds' rigid transform
	int S_k = point_cloud_pair.S_k - this -> anchor_pc_index;
	int D_k = point_cloud_pair.D_k - this -> anchor_pc_index;



	arma::vec::fixed<3> x_S = arma::zeros<arma::vec>(3);
	arma::mat::fixed<3,3> dcm_S = arma::eye<arma::mat>(3,3);

	arma::vec::fixed<3> x_D = arma::zeros<arma::vec>(3);
	arma::mat::fixed<3,3> dcm_D = arma::eye<arma::mat>(3,3);

	if (point_cloud_pair.S_k != this -> anchor_pc_index){
		x_S = this -> X.subvec(6 * (S_k - 1) , 6 * (S_k - 1) + 2);
		dcm_S =  RBK::mrp_to_dcm(this -> X.subvec(6 * (S_k - 1) + 3, 6 * (S_k - 1) + 5));
	}

	if (point_cloud_pair.D_k != this -> anchor_pc_index){
		x_D = this -> X.subvec(6 * (D_k - 1) , 6 * (D_k - 1) + 2);
		dcm_D = RBK::mrp_to_dcm(this -> X.subvec(6 * (D_k - 1) + 3, 6 * (D_k - 1) + 5));
	}
	
	int active_h;

	if (!this -> use_true_pairs){


		
		active_h = this -> h;

		
		IterativeClosestPointToPlane::compute_pairs(
			this -> all_registered_pc -> at(point_cloud_pair.S_k),
			this -> all_registered_pc -> at(point_cloud_pair.D_k),
			point_pairs,
			active_h,
			dcm_S ,
			x_S,
			dcm_D ,
			x_D );
	}

	else{
		throw(std::runtime_error("Not implemented"));
		
	}

	
	arma::rowvec H_ki;

	if (point_cloud_pair.D_k != this -> anchor_pc_index && point_cloud_pair.S_k != this -> anchor_pc_index){
		H_ki = arma::zeros<arma::rowvec>(12);
	}
	else{
		H_ki = arma::zeros<arma::rowvec>(6);
	}


	// For all the point pairs that where formed
	for (unsigned int i = 0; i < point_pairs.size(); ++i){

		IterativeClosestPointToPlane icp;

		double y_ki = icp.compute_distance(
			this -> all_registered_pc -> at(point_cloud_pair.S_k),
			this -> all_registered_pc -> at(point_cloud_pair.D_k),
			point_pairs[i],
			dcm_S,
			x_S,
			dcm_D,
			x_D);
		

		const PointNormal & p_S = this -> all_registered_pc -> at(point_cloud_pair.S_k) . get_point(point_pairs[i].first);
		const PointNormal & p_D = this -> all_registered_pc -> at(point_cloud_pair.D_k) . get_point(point_pairs[i].second);

		const arma::vec::fixed<3> & S_i = p_S.get_point_coordinates();
		const arma::vec::fixed<3> & D_i = p_D.get_point_coordinates();


		const arma::vec::fixed<3> & n = p_D.get_normal_coordinates();

		if (point_cloud_pair.D_k != this -> anchor_pc_index && point_cloud_pair.S_k != this -> anchor_pc_index){

			H_ki.subvec(0,2) = (dcm_D * n).t();
			H_ki.subvec(3,5) = (4 * RBK::tilde(S_i - this -> shift_origin) * dcm_S.t() * dcm_D * n).t();
			H_ki.subvec(6,8) = -(dcm_D * n).t();
			H_ki.subvec(9,11) = (- 4 * RBK::tilde(D_i) * n +  4 * RBK::tilde(n) * dcm_D.t() * (dcm_S * (S_i - this -> shift_origin)
				+ x_S - dcm_D * (D_i - this -> shift_origin) - x_D)).t();

		}

		else if(point_cloud_pair.S_k != this -> anchor_pc_index) {

			throw(std::runtime_error("This should never happen"));
			H_ki.subvec(0,2) = (dcm_D * n).t();
			H_ki.subvec(3,5) = (4 * RBK::tilde(S_i - this -> shift_origin) * dcm_S.t() * dcm_D * n).t();
		}

		else{
			H_ki.subvec(0,2) = -(dcm_D * n).t();
			H_ki.subvec(3,5) = (- 4 * RBK::tilde(D_i - this -> shift_origin) * n + 4 * RBK::tilde(n) * dcm_D.t() * (dcm_S * (S_i - this -> shift_origin) 
				+ x_S - dcm_D * (D_i - this -> shift_origin) - x_D)).t();
		}

		// epsilon = y - Hx !!!
		H_ki = - H_ki;



		// Uncertainty on measurement
		arma::rowvec::fixed<3> mapping_vector = (dcm_S * (S_i - this -> shift_origin) + x_S - dcm_D * (D_i - this -> shift_origin) - x_D).t() * dcm_D ;
		
		arma::vec::fixed<3> e = {1,0,0};
		double sigma_angle = 0.3; //5.7 deg of uncertainty

		arma::mat::fixed<3,3> R_n = std::pow(sigma_angle,2) / 2 * (arma::eye<arma::mat>(3,3) - n * n.t());

		double sigma_y_squared = arma::dot(mapping_vector.t(),R_n * mapping_vector.t());


		sigma_y_squared += std::pow(this -> sigma_rho,2) * arma::dot(
			dcm_D * n,
			(dcm_S * M_pcs.at(point_cloud_pair.S_k) * e * e.t() * (dcm_S * M_pcs.at(point_cloud_pair.S_k)).t()
				+ dcm_D * M_pcs.at(point_cloud_pair.D_k) * e * e.t() * (dcm_D * M_pcs.at(point_cloud_pair.D_k)).t())
			* dcm_D * n);


		Lambda_k +=  H_ki.t() * H_ki / sigma_y_squared;
		N_k +=  H_ki.t() * y_ki / sigma_y_squared;

	}


}

bool BundleAdjuster::update_point_cloud_pairs(bool last_iter){

	double max_error = -1;
	int worst_Sk,worst_Dk;
	int sum_point_pairs_sizes = 0;

	arma::vec errors(this -> point_cloud_pairs.size());
	arma::vec pc_pair_sizes(this -> point_cloud_pairs.size());


	for (int k = 0; k < this -> point_cloud_pairs.size(); ++k){
		
		const BundleAdjuster::PointCloudPair & point_cloud_pair = this -> point_cloud_pairs[k];



		int S_k = point_cloud_pair.S_k - this -> anchor_pc_index;
		int D_k = point_cloud_pair.D_k - this -> anchor_pc_index;


		arma::vec::fixed<3> x_S = arma::zeros<arma::vec>(3);
		arma::mat::fixed<3,3> dcm_S = arma::eye<arma::mat>(3,3);

		arma::vec::fixed<3> x_D = arma::zeros<arma::vec>(3);
		arma::mat::fixed<3,3> dcm_D = arma::eye<arma::mat>(3,3);


		if (point_cloud_pair.S_k != this -> anchor_pc_index){
			x_S = this -> X.subvec(6 * (S_k - 1) , 6 * (S_k - 1) + 2);
			dcm_S =  RBK::mrp_to_dcm(this -> X.subvec(6 * (S_k - 1) + 3, 6 * (S_k - 1) + 5));
		}

		if (point_cloud_pair.D_k != this -> anchor_pc_index){
			x_D = this -> X.subvec(6 * (D_k - 1) , 6 * (D_k - 1) + 2);
			dcm_D = RBK::mrp_to_dcm(this -> X.subvec(6 * (D_k - 1) + 3, 6 * (D_k - 1) + 5));
		}

		assert(point_cloud_pair.D_k != this -> anchor_pc_index);


		std::vector<PointPair> point_pairs;

		if (!this -> use_true_pairs){
			

			IterativeClosestPointToPlane::compute_pairs(
				this -> all_registered_pc -> at(point_cloud_pair.S_k),
				this -> all_registered_pc -> at(point_cloud_pair.D_k),
				point_pairs,
				this -> h,
				dcm_S ,
				x_S,
				dcm_D ,
				x_D );
		}

		else{

			throw(std::runtime_error("This should never happen"));

		}

		IterativeClosestPointToPlane icp;

		double error = std::abs(icp.compute_residuals(
			this -> all_registered_pc -> at(point_cloud_pair.S_k),
			this -> all_registered_pc -> at(point_cloud_pair.D_k),
			point_pairs,
			dcm_S ,
			x_S,
			{},
			dcm_D ,
			x_D));

		// errors(k) = error * point_pairs.size();
		sum_point_pairs_sizes += point_pairs.size();
		pc_pair_sizes(k) = point_pairs.size();

		errors(k) = error ;
		

		double p = std::log2(
			this -> all_registered_pc -> at(this -> point_cloud_pairs[k].S_k).size());
		
		int N_pairs = (int)(std::pow(2, p - this -> h));


		this -> point_cloud_pairs[k].error = error;
		this -> point_cloud_pairs[k].N_accepted_pairs = point_pairs.size();
		this -> point_cloud_pairs[k].N_pairs = N_pairs;

	}

	// errors /= (sum_point_pairs_sizes / this -> point_cloud_pairs.size());

	for (int k = 0; k < this -> point_cloud_pairs.size(); ++k){

		if (errors(k) > max_error){
			max_error = errors(k);
			worst_Dk = this -> point_cloud_pairs[k].D_k;
			worst_Sk = this -> point_cloud_pairs[k].S_k;
		}

		std::cout << " -- h == " << this -> h << " , (" << this -> point_cloud_pairs[k].S_k << "||"  <<  this -> all_registered_pc -> at(this -> point_cloud_pairs[k].S_k) . size() << ", " << this -> point_cloud_pairs[k].D_k << "||"  <<  this -> all_registered_pc -> at(this -> point_cloud_pairs[k].D_k) . size() <<  ") : " << errors(k) << " | " << pc_pair_sizes(k) << " point pairs" << std::endl;

	}

	if (this -> point_cloud_pairs.size() < 2) return false;

	arma::gmm_diag model_residuals;
	std::set<unsigned int> acceptable_clusters;
	arma::urowvec residuals_gaus_ids;

	int N_clusters_max = int(this -> point_cloud_pairs.size()) - 1;

	for (int N_clusters = 1; N_clusters <= N_clusters_max; ++N_clusters){

		// Init GMM 
		std::cout << "\tUsing " << N_clusters << " mixtures\n";
		acceptable_clusters.clear();

		// Training GMM
		model_residuals.learn(errors.t(), N_clusters, arma::maha_dist, arma::random_subset, 10, 10, 1e-10, false);
		residuals_gaus_ids = model_residuals.assign( errors.t(), arma::prob_dist);

		// GMM learned parameters
		arma::urowvec hist = arma::hist(residuals_gaus_ids,arma::regspace<arma::urowvec>(0,N_clusters - 1));
		model_residuals.means.print("\tResiduals GMM means: ");
		arma::sqrt(model_residuals.dcovs).print("\tResiduals GMM standard deviations: ");
		arma::rowvec(model_residuals.means - 3 * arma::sqrt(model_residuals.dcovs)).print("\tResiduals GMM means minus 3 standard deviations: ");
		hist.print("\tPopulation of each cluster: ");


		std::cout << "\tCluster assignments: " << std::endl;

		for (unsigned int k = 0; k < this -> point_cloud_pairs.size(); ++k){
			std::cout << "\t -- (" << this -> point_cloud_pairs[k].S_k << " , " << this -> point_cloud_pairs[k].D_k <<  ") : " << residuals_gaus_ids(k) << " \n";
		}

		if ((model_residuals.means - 3 * arma::sqrt(model_residuals.dcovs)).min() > 0){

			// The acceptable clusters are stored
			arma::urowvec most_populated_clusters = arma::find(hist == hist.max()).t();
			double largest_acceptable_error = std::max(1.2 * arma::min(model_residuals.means(most_populated_clusters)),this -> sigma_rho);
			std::cout<< "\t\tClustering achieved. Maximum acceptable cluster mean error: " << largest_acceptable_error << std::endl;
			
			for (unsigned int p = 0; p < N_clusters; ++p){
				if (model_residuals.means(p) <= largest_acceptable_error){
					acceptable_clusters.insert(p);
				}
			}

			break;
		}

	}


	std::cout << "-- Maximum point-cloud pair ICP error at (" << worst_Sk << " , " << worst_Dk <<  ") : " << max_error << std::endl;

	if(last_iter){
		this -> edges_to_remove.clear();

		for (unsigned int k = 0; k < this -> point_cloud_pairs.size(); ++k){

			if (acceptable_clusters.find(residuals_gaus_ids(k)) == acceptable_clusters.end() ){

				std::cout << "-- Bad edge ("  << this -> point_cloud_pairs[k].S_k << " , " << this -> point_cloud_pairs[k].D_k <<   ")\n";

				if (this -> anchor_pc_index !=  this -> next_anchor_pc_index){

					if (this -> point_cloud_pairs[k].D_k <= this -> next_anchor_pc_index){
						std::cout << "--- Cancelling creation of local structure since a bad edge ("  << this -> point_cloud_pairs[k].S_k << " , " << this -> point_cloud_pairs[k].D_k <<   ") was present\n";
						this -> next_anchor_pc_index = this -> anchor_pc_index;
					}
				}

				std::set<int> edge_to_remove;
				edge_to_remove.insert(this -> point_cloud_pairs[k].D_k);
				edge_to_remove.insert(this -> point_cloud_pairs[k].S_k);

				if (this -> can_remove_edge(edge_to_remove)){
					this -> edges_to_remove.push_back(edge_to_remove);

				}
			}
		}

	// If no edge needs to be removed, BA has converged
		if (this -> edges_to_remove.size() == 0){
			return true;
		}
		else{
			return false;
		}
	}
	else{
		return false;
	}


}

void BundleAdjuster::add_subproblem_to_problem(std::vector<T>& coeffs,
	EigVec & N,
	const arma::mat & Lambda_k,
	const arma::vec & N_k,
	const PointCloudPair & point_cloud_pair){

	int S_k = point_cloud_pair.S_k - this -> anchor_pc_index;
	int D_k = point_cloud_pair.D_k - this -> anchor_pc_index;

	
	if (D_k != 0 && S_k != 0){

		// S_k substate
		for(unsigned int i = 0; i < 6; ++i){
			for(unsigned int j = 0; j < 6; ++j){
				coeffs.push_back(T(6 * (S_k - 1) + i, 6 * (S_k -1) + j,Lambda_k(i,j)));
			}
			N(6 * (S_k -1 ) + i) += N_k(i);
		}
		
		// D_k substate
		for(unsigned int i = 0; i < 6; ++i){
			for(unsigned int j = 0; j < 6; ++j){
				coeffs.push_back(T(6 * (D_k - 1) + i, 6 * (D_k - 1) + j,Lambda_k(i + 6,j + 6)));
			}
			N(6 * (D_k -1) + i) += N_k(i + 6);
		}

		// Cross-correlations
		for(unsigned int i = 0; i < 6; ++i){
			for(unsigned int j = 0; j < 6; ++j){
				coeffs.push_back(T(6 * (S_k-1)  + i, 6 * (D_k -1) + j,Lambda_k(i,j + 6)));
				coeffs.push_back(T(6 * (D_k -1) + i, 6 * (S_k -1) + j,Lambda_k(i + 6,j)));
			}
		}

	}

	else if (S_k != 0){

		throw(std::runtime_error("BundleAdjuster::add_subproblem_to_problem: This should never happen"));

		// S_k substate
		for(unsigned int i = 0; i < 6; ++i){
			for(unsigned int j = 0; j < 6; ++j){
				coeffs.push_back(T(6 * (S_k-1)  + i, 6 * (S_k -1) + j,Lambda_k(i,j)));
			}
			N(6 * (S_k -1)  + i) += N_k(i);
		}

	}

	else {

		// D_k substate
		for(unsigned int i = 0; i < 6; ++i){
			for(unsigned int j = 0; j < 6; ++j){
				coeffs.push_back(T(6 * (D_k-1)  + i, 6 * (D_k -1) + j,Lambda_k(i,j)));
			}
			N(6 * (D_k -1) + i) += N_k(i);
		}
	}

}


void BundleAdjuster::apply_deviation(const EigVec & deviation){

	std::cout << "\t Deviations: \n";

	for (unsigned int i = 1 + this -> anchor_pc_index ; i < this -> all_registered_pc -> size(); ++i){
		
		int x_index = 6 * (i - 1 - this -> anchor_pc_index);
		int mrp_index = x_index + 3 ;

		std::cout << "\t\t pc # " << i << arma::vec::fixed<6>({
			deviation(x_index),
			deviation(x_index + 1),
			deviation(x_index + 2),
			deviation(mrp_index),
			deviation(mrp_index + 1),
			deviation(mrp_index + 2)}).t();
	}



	boost::progress_display progress(this -> all_registered_pc -> size() - this -> anchor_pc_index);
	++progress;
	#pragma omp parallel for
	for (unsigned int i = 1 + this -> anchor_pc_index ; i < this -> all_registered_pc -> size(); ++i){

		int x_index = 6 * (i - 1 - this -> anchor_pc_index);
		int mrp_index = x_index + 3 ;

		arma::vec::fixed<3> dx  = {
			deviation(x_index),
			deviation(x_index + 1),
			deviation(x_index + 2)
		};

		// The mrp used in the partials 
		// instantiates
		// [NS_bar]
		// but the solved for d_mrp
		// corresponds to 
		// [SS_bar]
		// so need to apply 
		// [S_barS] = dcm_to_mrp(-d_mrp)
		// as in [NS_bar] = [NS_bar] * [S_barS]

		arma::vec::fixed<3> d_mrp  = {
			deviation(mrp_index),
			deviation(mrp_index + 1),
			deviation(mrp_index + 2)
		};

		arma::mat::fixed<3,3> SS_bar = RBK::mrp_to_dcm(d_mrp);
		arma::mat::fixed<3,3> NS_bar = RBK::mrp_to_dcm(this -> X.subvec(mrp_index, mrp_index + 2));

		this -> X.subvec(x_index , x_index + 2) += dx ;
		this -> X.subvec(mrp_index, mrp_index + 2) = RBK::dcm_to_mrp(NS_bar * SS_bar.t());

		++progress;

	}

}


void BundleAdjuster::update_point_clouds(std::map<int,arma::mat::fixed<3,3> > & M_pcs, 
	std::map<int,arma::vec::fixed<3> > & X_pcs,
	std::map<int,arma::mat::fixed<6,6> > & R_pcs,
	std::vector<arma::mat::fixed<3,3> > & BN_measured,
	const std::vector<arma::vec::fixed<3> > & mrps_LN){

	boost::progress_display progress(this -> all_registered_pc -> size() - this -> anchor_pc_index );
	++progress;
	
	#pragma omp parallel for
	for (unsigned int i = 1 + this -> anchor_pc_index ; i < this -> all_registered_pc -> size(); ++i){

		int x_index = 6 * (i - 1 - this -> anchor_pc_index);
		int mrp_index = x_index + 3 ;

		const arma::vec::fixed<3> & x = this -> X.subvec(x_index , x_index + 2);
		const arma::vec::fixed<3> & mrp = this -> X.subvec(mrp_index, mrp_index + 2);

		arma::mat::fixed<3,3> NS_bar = RBK::mrp_to_dcm(mrp);
		
		this -> all_registered_pc -> at(i) . transform(NS_bar, x);
		this -> all_registered_pc -> at(i) . build_kdtree(false);

		// The rigid transforms are fixed
		M_pcs[i] = NS_bar * M_pcs[i];
		X_pcs[i] = NS_bar * X_pcs[i] + x;


		// The small body attitude is fixed
		// M_pc(k) is [LB](t_0) * [BL](t_k) = [LN](t_0)[NB](t_0) * [BN](t_k) * [NL](t_k);
		BN_measured[i] = BN_measured[0] *  RBK::mrp_to_dcm( mrps_LN[0]).t() * M_pcs[i] * RBK::mrp_to_dcm(mrps_LN[i]);

		++progress;

	}

}

void BundleAdjuster::save_local_bundle(){

	PointCloud<PointNormal> new_structure_pc;
	
	for (int k = this -> anchor_pc_index + 1; k <= this -> next_anchor_pc_index; ++k){
		
		for (int j = 0; j <  this -> all_registered_pc -> at(k) . size(); ++j){
			new_structure_pc.push_back( this -> all_registered_pc -> at(k) . get_point(j));
		}
		
	}

	PointCloudIO<PointNormal>::save_to_obj(
		new_structure_pc,
		this -> dir + "/local_bundle_" + std::to_string(this -> next_anchor_pc_index) + ".obj",
		this -> LN_t0 -> t(), 
		*this -> x_t0);

	std::cout << "\t Created local structure local_bundle_" + std::to_string(this -> next_anchor_pc_index) << " with " << new_structure_pc.size() << " points\n";
	

}


std::map<double,int> BundleAdjuster::find_overlap_with_pc(int pc_global_index,int start_index,int end_index,
	bool prune_overlaps) const{

	std::vector<PointPair> point_pairs;
	std::map<double,int> overlaps;

	std::vector<int> pcs_to_check;
	int active_h = 5;
	int len = std::abs(end_index - start_index) + 1;
	
	for (int i = 0; i < len; ++i){
		if (start_index < end_index){
			pcs_to_check.push_back(start_index + i);

		}
		else{
			pcs_to_check.push_back(start_index - i);
		}
	}

	for (auto other_pc_index : pcs_to_check){

		if (other_pc_index == pc_global_index) continue;


		// Should compute the los here and do not attempt overlap check
		// if they are more than 90 degrees apart
		std::set<int> current_pc_pair = {pc_global_index,other_pc_index};

		// We need the [BL] dcms to transform the line of sights into the body-fixed frame
		arma::vec::fixed<3> los_pc_global_index = {1,0,0};
		arma::vec::fixed<3> los_other_pc_index = {1,0,0};


		// Some point cloud indices may actually be pointing at a point cloud bundle
		// We must loop over each point cloud in the bundle to see if one makes sense or not
		double angle = std::numeric_limits<double>::infinity();


		if (other_pc_index == this -> anchor_pc_index){
			// The other pc index points at a point cloud bundle. It is necessary
			// to compare the los associated with pc_global_index to the los associated 
			// to each of the bundle's point clouds
			std::cout << " Checking " << other_pc_index << " since it concludes a bundle\n" << std::endl;

			for (int k = this -> previous_anchor_pc_index; k <= this -> anchor_pc_index ; ++k ){

				arma::mat::fixed<3,3> LB_pc_global_index = RBK::mrp_to_dcm(mrp_LN_ptr -> at(pc_global_index)) * BN_measured_ptr -> at(pc_global_index).t();
				arma::mat::fixed<3,3> LB_pc_k = RBK::mrp_to_dcm(mrp_LN_ptr -> at(k)) * BN_measured_ptr -> at(k).t();

				angle = std::min(std::acos(arma::dot(LB_pc_global_index.t() * los_pc_global_index,
					LB_pc_k.t() * los_other_pc_index) ) * 180./arma::datum::pi,angle);
				std::cout << "\t Angle == " << std::acos(arma::dot(LB_pc_global_index.t() * los_pc_global_index,
					LB_pc_k.t() * los_other_pc_index) ) * 180./arma::datum::pi << " for pc # " << k <<  " in bundle \n" << std::endl;

			}

		}
		else{

			arma::mat::fixed<3,3> LB_pc_global_index = RBK::mrp_to_dcm(mrp_LN_ptr -> at(pc_global_index)) * BN_measured_ptr -> at(pc_global_index).t();
			arma::mat::fixed<3,3> LB_pc_other_index = RBK::mrp_to_dcm(mrp_LN_ptr -> at(other_pc_index)) * BN_measured_ptr -> at(other_pc_index).t();

			angle = std::acos(arma::dot(LB_pc_global_index.t() * los_pc_global_index,
				LB_pc_other_index.t() * los_other_pc_index) ) * 180./arma::datum::pi;

		}



		if (angle > 120){
			std::cout << " Skipping ( " << *current_pc_pair.begin() << " , "<< *(--current_pc_pair.end()) << " ) since los are " << angle <<  " degrees appart" << std::endl;
		}
		else{

			std::cout << " Investigating ( " << *current_pc_pair.begin() << " , "<< *(--current_pc_pair.end()) << " ) since los are " << angle <<  " degrees appart" << std::endl;

			double p = std::log2(this -> all_registered_pc -> at(pc_global_index) . size());

			int N_pairs = (int)(std::pow(2, p - active_h));

			try{

				IterativeClosestPointToPlane::compute_pairs(
					this -> all_registered_pc -> at(pc_global_index),
					this -> all_registered_pc -> at(other_pc_index),
					point_pairs,
					active_h);

				double prop = double(point_pairs.size()) / N_pairs * 100;

				std::cout << "\t ( " << *current_pc_pair.begin() << " , "<< *(--current_pc_pair.end()) << " ) : " << point_pairs.size() << " point pairs , " << prop << " (%) overlap"<< std::endl;

				if (prop > 60 || (bool)(std::abs(int(*current_pc_pair.begin()) - int(*(--current_pc_pair.end()))) == 1)){
					
					overlaps[prop] = other_pc_index;
					if (prune_overlaps && overlaps.size() > 5){
						return overlaps;
					}

				}

			}
			catch(ICPNoPairsException & e){
				std::cerr << e.what() << std::endl;

			}
			catch(std::logic_error & e){
				std::cerr << e.what() << std::endl;

			}
		}

	}

	return overlaps;

}



bool BundleAdjuster::update_overlap_graph(){

	if (!this -> graph.vertexexists(0)){
		std::cout << "\t Inserting anchor point cloud # 0  in graph\n";
		this -> graph.addvertex(0);
	}

	int new_pc_index = static_cast<int>(this -> all_registered_pc -> size()) - 1;

	std::cout << "\t Inserting point cloud # " << new_pc_index  << " in graph\n";
	this -> graph.addvertex(new_pc_index);

	auto overlap = this -> find_overlap_with_pc(new_pc_index,this -> anchor_pc_index,new_pc_index - 1,false);
	int max_closure_length = -1;

	for (auto it = overlap.begin(); it != overlap.end(); ++it){
		this -> graph.addedge(new_pc_index,it -> second,it -> first);

		max_closure_length = std::max(max_closure_length,std::abs(new_pc_index - it -> second));


		if (this -> overlap_with_anchor_cluster_from_outside(new_pc_index,it -> second)){
			
			// We have full loop closure. new_pc_index will define the new anchor index
			// after ba has been run
			this -> next_anchor_pc_index = new_pc_index;
		}

	}

	std::cout << "\t Graph has " << this -> graph.get_n_edges() << " unique edges. Longest closure: " <<  max_closure_length << "\n";
	// return true;

	if (max_closure_length > this -> cluster_size){
		// there is closure between new_pc_index and another point cluster more than this -> cluster_size away. run bundle adjustment
		return true;
	}
	else{
		return false;
	}

}

void BundleAdjuster::set_cluster_size(int size){
	this -> cluster_size = size;
}

void BundleAdjuster::remove_edges_from_graph(){


	for (auto & edge_to_remove : this -> edges_to_remove){
		std::cout << "\t Removing edge (" << *edge_to_remove.begin() << "," << *(--edge_to_remove.end()) << ") based on residuals\n";

		
		this -> graph.removeedge(*edge_to_remove.begin(),*(--edge_to_remove.end()));

	}

	// The graph is cleaned up by keeping up to N at each node
	for (int i = 0; i < this -> all_registered_pc -> size(); ++i){

		std::set<int> neighbors = this -> graph.getneighbors(i);

		
		// Keep edges between consecutive point clouds
		if (i > 0){
			neighbors.erase(i - 1);

		}
		if (i < static_cast<int>(this -> all_registered_pc -> size()) - 1){
			neighbors.erase(i + 1);
		}

		if (neighbors.size() == 0){
			std::cout << "\t pc # " << i << " has no non-consecutive neighbors\n";
			continue;
		}
		
		std::vector<std::set<int> > clusters;

		std::set<int> cluster;
		
		cluster.insert(*neighbors.begin());

		for (int neighbor : neighbors){

			if (std::abs(neighbor - *(cluster.begin())) <= this -> cluster_size){
				// If this neighbor and the one that started the cluster
				// are less than this -> cluster_size appart
				// then the neighbor belongs to the same cluster
				cluster.insert(neighbor);
			}
			else{
				// If not, then the neighbor is added to a new cluster. 
				// The previous cluster is saved as is
				// And a new cluster is created with the neighbor
				
				clusters.push_back(cluster);
				cluster.clear();

				cluster.insert(neighbor);

			}

		}

		if (cluster.size() != 0){
			clusters.push_back(cluster);
		}
		
		#if BUNDLE_ADJUSTER_DEBUG
		std::cout << "\tFor pc # " << i << ", formed clusters: \n";
		for (int k = 0; k < clusters.size(); ++k){
			std::cout << "\t\tCluster #" << k << " : (";
			for (auto index : clusters[k]){
				std::cout << index << ", ";
			}
			std::cout << ")" << std::endl;
		}
		#endif


		// Clusters now stores all the different clusters of neighbors.
		// only only one point cloud per cluster will be kept

		// Would be nice to have a receding memory cleanup of the graph. That is, 
		// only edges between consecutive point clouds are kept forever. Edges formed in 
		// the bundle adjustment phase will be kept if they are within 10 * this -> cluster size from
		// the current point cloud

		for (auto cluster_to_process : clusters){

			
			auto cluster_to_process_it = cluster_to_process.begin();

			
			// // The first index in the cluster is kept
			// ++cluster_to_process_it;

			// The rest of the indices in this cluster are discarded
			while(cluster_to_process_it != cluster_to_process.end()){


				if(std::abs(std::max(*cluster_to_process_it,i) - (int(this -> all_registered_pc -> size()) - 1) ) > 3 * this -> cluster_size){
					
					if (this ->  can_remove_edge(std::set<int>({i,*cluster_to_process_it}))){

						std::cout << "\t Removing edge (" << i << "," << *cluster_to_process_it << ") based on receding graph memory\n";
						this -> graph.removeedge(i,*cluster_to_process_it);
					}
				}
				else if (cluster_to_process_it != cluster_to_process.begin()){
					
					if (this ->  can_remove_edge(std::set<int>({i,*cluster_to_process_it}))){

						std::cout << "\t Removing edge (" << i << "," << *cluster_to_process_it << ") based on clustering\n";
						this -> graph.removeedge(i,*cluster_to_process_it);
					}
				}
				++cluster_to_process_it;
			}

		}

	}

}

void BundleAdjuster::set_h(int h){
	this -> h = h;
}


const PointCloud < PointNormal > & BundleAdjuster::get_anchor_pc() const{
	return this -> all_registered_pc -> at(this -> anchor_pc_index);
}


bool BundleAdjuster::overlap_with_anchor_cluster_from_outside(int new_pc_index,int pc_maybe_in_anchor_cluster) const{



	if ((bool)(std::abs(pc_maybe_in_anchor_cluster - this -> anchor_pc_index) <= this -> cluster_size)  
		&& (bool)(std::abs(new_pc_index - pc_maybe_in_anchor_cluster) > this -> cluster_size) ){
		
		std::cout << "\t Found closure with anchor index\n";

	return true;
}
else{
	return false;
}



}

bool BundleAdjuster::can_remove_edge(const std::set<int> & edge_to_remove) {

	assert(edge_to_remove.size() == 2);

	int p0 = *edge_to_remove.begin();
	int p1 = *(--edge_to_remove.end());

	auto first_pc_neighbors = this -> graph . getneighbors(p0);
	auto second_pc_neighbors = this -> graph . getneighbors(p1);

	// Only looking for other neighbors
	first_pc_neighbors.erase(p1);
	second_pc_neighbors.erase(p0);

	// Checking connectivity of other neighbors
	// For the edge to be removable, the neighbors can't all be after or before the considered points
	// So the two pcs must have at least two neighbors left

	if (first_pc_neighbors.size() < 2 || second_pc_neighbors.size() < 2){
		return false;
	}

	// If the two pcs have at least two neighbors, we need to check if one pc's neighbors are "around" it

	bool p0_surrounded = (p0 > *first_pc_neighbors.begin()) && (p0 < *(--first_pc_neighbors.end()));
	bool p1_surrounded = (p1 > *second_pc_neighbors.begin()) && (p1 < *(--second_pc_neighbors.end()));



	if (p0 == this -> anchor_pc_index){
		p0_surrounded = (p0 < *(--first_pc_neighbors.end()));
	}
	if (p1 == int(this -> graph.getnumv()) - 1){
		p1_surrounded = (p1 > *second_pc_neighbors.begin());
	}



	return (p0_surrounded && p1_surrounded);


}




