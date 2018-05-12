#include "BundleAdjuster.hpp"
#include <armadillo>
#include "ICP.hpp"
#include "boost/progress.hpp"
#include "DebugFlags.hpp"
#include "FlyOverMap.hpp"





BundleAdjuster::BundleAdjuster(
	int t0, 
	int tf,
	std::map<int,arma::mat> & M_pcs,
	std::map<int,arma::vec> & X_pcs,
	std::vector<arma::mat> & BN_estimated,
	std::vector< std::shared_ptr<PC> > * all_registered_pc_, 
	int N_iter,
	const arma::mat & LN_t0,
	const arma::vec & x_t0,
	const std::vector<arma::vec> & mrps_LN,
	bool save_connectivity,
	int & previous_closure_index){


	this -> all_registered_pc = all_registered_pc_;
	this -> LN_t0 = LN_t0;
	this -> x_t0 = x_t0;
	this -> N_iter = N_iter;


	for (int i = t0; i <= tf; ++i){
		this -> local_pc_index_to_global_pc_index.push_back(i);
	}

	// The connectivity between point clouds is inferred
	std::cout << "- Creating point cloud pairs" << std::endl;
	this -> create_pairs(previous_closure_index);

	if (this -> local_pc_index_to_global_pc_index.size() == 0){
		std::cout << " - Nothing to do here, no loop closure or already closed\n";
		return;
	}

	if (this -> N_iter > 0){
	// solve the bundle adjustment problem
		this -> solve_bundle_adjustment();
		std::cout << "- Solved bundle adjustment" << std::endl;
	}	


	std::cout << "- Updating point clouds ... " << std::endl;
	this -> update_point_clouds(M_pcs,
		X_pcs,
		BN_estimated,
		mrps_LN);
	

	// The connectivity matrix is saved
	if (save_connectivity){
		this -> save_connectivity();
	}
}







void BundleAdjuster::solve_bundle_adjustment(){

	int Q = this -> local_pc_index_to_global_pc_index. size();
	this -> X = arma::zeros<arma::vec>(6 * (Q - 1));

	// This allows to compute the ICP RMS residuals for each considered point-cloud pair before running the bundle adjuster
	this -> update_point_cloud_pairs();

	for (int iter = 0 ; iter < this -> N_iter; ++iter){

		std::cout << "Iteration: " << std::to_string(iter + 1) << " /" << std::to_string(N_iter) << std::endl;

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


			if (this -> point_cloud_pairs . at(k).D_k != 0 && this -> point_cloud_pairs . at(k).S_k != 0){
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
			this -> assemble_subproblem(Lambda_k_vector. at(k),N_k_vector. at(k),this -> point_cloud_pairs . at(k));
			#if !BUNDLE_ADJUSTER_DEBUG
			++progress;
			#endif

		}

		for (int k = 0; k < this -> point_cloud_pairs.size(); ++k){

			// They are added to the whole problem
			this -> add_subproblem_to_problem(coefficients,Nmat,Lambda_k_vector. at(k),N_k_vector. at(k),this -> point_cloud_pairs . at(k));

			#if BUNDLE_ADJUSTER_DEBUG
			std::cout << "Subproblem info matrix: " << std::endl;
			std::cout << Lambda_k << std::endl;
			std::cout << "Conditionning : " << arma::cond(Lambda_k) << std::endl;
			std::cout << "Subproblem normal matrix: " << std::endl;
			std::cout << N_k << std::endl;
			#endif 


		}	

		
		std::cout << "- Solving for the deviation" << std::endl;

		// The deviation in all of the rigid transforms is computed
		Lambda.setFromTriplets(coefficients.begin(), coefficients.end());
		
		// The cholesky decomposition of Lambda is computed
		Eigen::SimplicialCholesky<SpMat> chol(Lambda);  

		// The deviation is computed
		EigVec deviation = chol.solve(Nmat);    

		// It is applied to all of the point clouds (minus the first one)
		std::cout << "- Applying the deviation" << std::endl;

		this -> apply_deviation(deviation);
		std::cout << "\n- Updating the point pairs" << std::endl;

		// The point cloud pairs are updated: their residuals are updated
		// and the rigid transforms positionning them are also updated
		this -> update_point_cloud_pairs();

		#if BUNDLE_ADJUSTER_DEBUG
		std::cout << "Deviation: " << std::endl;
		std::cout << this -> dX << std::endl;
		#endif

	}


}

int BundleAdjuster::get_cutoff_index() const{
	return this -> closure_index;
}

void BundleAdjuster::create_pairs( int & previous_closure_index){

	std::vector<PointPair> point_pairs;
	std::set<std::set<int> > pairs;

	int ground_index = 0; 

	// Checking possible closure between current point cloud and first cloud
	for (int tf = local_pc_index_to_global_pc_index.size() - 1 ; tf > ground_index ; --tf){
		
		try{
			ICP::compute_pairs(point_pairs,
				this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[ground_index]),
				this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[tf]),
				this -> h);

			double p = std::log2(this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[ground_index]) -> get_size());

			int N_pairs = (int)(std::pow(2, p - this -> h));

			double prop = double(point_pairs.size()) / N_pairs * 100;

			std::cout << " ( " << this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[ground_index]) -> get_label() << " , "<<
			this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[tf]) -> get_label() << " ) : " << point_pairs.size() << " point pairs , " << prop << " (%) overlap"<< std::endl;

			if (prop > 70){
				std::cout << "Choosing " << " ( " << this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[ground_index]) -> get_label() << " , "<<

				this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[tf]) -> get_label() << " ) in loop closure" <<  std::endl;

				std::set<int> pair = {tf,ground_index};
				pairs.insert(pair);
				if (tf != previous_closure_index){
					this -> closure_index = tf;
					previous_closure_index = tf;
					break;
				}
				else{
					this -> local_pc_index_to_global_pc_index.clear();
					return;
				}
			}
		}
		catch(ICPNoPairsException & e){

		}

	}

	// Only the point cloud pairs that "close the loop" with the ground point cloud are kep
	std::vector<int> local_pc_index_to_global_pc_index_temp;
	for (int i = 0; i <= this -> closure_index; ++i){
		local_pc_index_to_global_pc_index_temp.push_back(this -> local_pc_index_to_global_pc_index[i]);
	}

	this -> local_pc_index_to_global_pc_index = local_pc_index_to_global_pc_index_temp;

	if (this -> local_pc_index_to_global_pc_index.size() == 0){
		return;
	}


	// The successive measurements are added
	for (int i = 0; i < this -> local_pc_index_to_global_pc_index.size() - 1; ++i){
		std::set<int> pair = {i,i+1};
		pairs.insert(pair);

	}


	#if BUNDLE_ADJUSTER_DEBUG
	std::cout << " -- Number of pairs: " << pairs.size() << std::endl;
	std::cout << " -- Storing pairs" << std::endl;
	#endif

	for (auto pair_iter = pairs.begin(); pair_iter != pairs.end(); ++pair_iter){

		std::set<int> pair_set = *pair_iter;

		int S_k = (*pair_set.begin());
		int D_k = (*std::next(pair_set.begin()));

		std::vector<PointPair> point_pairs;

		ICP::compute_pairs(point_pairs,
			this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[S_k]),
			this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[D_k]),this -> h);				
		double error = ICP::compute_rms_residuals(point_pairs);

		double p = std::log2(this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[S_k]) -> get_size());
		int N_pairs = (int)(std::pow(2, p - this -> h));

		BundleAdjuster::PointCloudPair pair;
		pair.S_k = S_k;
		pair.D_k = D_k;
		pair.error = error;
		pair.N_pairs = N_pairs;
		pair.N_accepted_pairs = point_pairs.size();
		this -> point_cloud_pairs.push_back(pair);


	}

}

void BundleAdjuster::assemble_subproblem(arma::mat & Lambda_k,arma::vec & N_k,const PointCloudPair & point_cloud_pair){

	// The point-pairs in the prescribed point-cloud pair are formed (with h = 0, so we are using them all)
	std::vector<PointPair> point_pairs;

	// The point pairs must be computed using the current estimate of the point clouds' rigid transform

	arma::vec x_S = arma::zeros<arma::vec>(3);
	arma::mat dcm_S = arma::eye<arma::mat>(3,3);

	arma::vec x_D = arma::zeros<arma::vec>(3);
	arma::mat dcm_D = arma::eye<arma::mat>(3,3);

	if (point_cloud_pair.S_k != 0){
		x_S = this -> X.subvec(6 * (point_cloud_pair.S_k - 1) , 6 * (point_cloud_pair.S_k - 1) + 2);
		dcm_S =  RBK::mrp_to_dcm(this -> X.subvec(6 * (point_cloud_pair.S_k - 1) + 3, 6 * (point_cloud_pair.S_k - 1) + 5));
	}

	if (point_cloud_pair.D_k != 0){
		x_D = this -> X.subvec(6 * (point_cloud_pair.D_k - 1) , 6 * (point_cloud_pair.D_k - 1) + 2);
		dcm_D = RBK::mrp_to_dcm(this -> X.subvec(6 * (point_cloud_pair.D_k - 1) + 3, 6 * (point_cloud_pair.D_k - 1) + 5));
	}
	
	ICP::compute_pairs(point_pairs,
		this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[point_cloud_pair.S_k]),
		this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[point_cloud_pair.D_k]),
		0,
		dcm_S,
		x_S,
		dcm_D,
		x_D);		


	#if BUNDLE_ADJUSTER_DEBUG
	std::cout << " - Subproblem : " << point_cloud_pair.S_k << " / " << point_cloud_pair.D_k << std::endl;
	std::cout << " - Number of pairs: " << point_pairs.size() << std::endl;
	std::cout << " - Residuals: " << ICP::compute_rms_residuals(point_pairs,dcm_S,x_S,dcm_D,x_D) << std::endl;
	#endif

	arma::rowvec H_ki;


	if (point_cloud_pair.D_k != 0 && point_cloud_pair.S_k != 0){
		H_ki = arma::zeros<arma::rowvec>(12);
	}
	else{
		H_ki = arma::zeros<arma::rowvec>(6);
	}


	#if BUNDLE_ADJUSTER_DEBUG
	std::cout << " - Looping over the point pairs\n";
	#endif

	// For all the point pairs that where formed
	for (unsigned int i = 0; i < point_pairs.size(); ++i){

		double y_ki = ICP::compute_normal_distance(point_pairs[i],dcm_S,x_S,dcm_D,x_D);
		arma::mat n = point_pairs[i].second -> get_normal();

		if (point_cloud_pair.D_k != 0 && point_cloud_pair.S_k != 0){

			H_ki.subvec(0,2) = n.t() * dcm_D.t();
			H_ki.subvec(3,5) = - 4 * n.t() * dcm_D.t() * dcm_S * RBK::tilde(point_pairs[i].first -> get_point());
			H_ki.subvec(6,8) = - n.t() * dcm_D.t();
			H_ki.subvec(9,11) = 4 * ( n.t() * RBK::tilde(point_pairs[i].second -> get_point()) 
				- (dcm_S * point_pairs[i].first -> get_point() + x_S - dcm_D * point_pairs[i].second -> get_point() - x_D).t() * dcm_D * RBK::tilde(n));
			// think I fixed a sign error
		}

		else if(point_cloud_pair.S_k != 0) {
			H_ki.subvec(0,2) = n.t() * dcm_D.t();
			H_ki.subvec(3,5) = - 4 * n.t() * dcm_D.t() * dcm_S * RBK::tilde(point_pairs[i].first -> get_point());

		}

		else{
			H_ki.subvec(0,2) = - n.t() * dcm_D.t();
			H_ki.subvec(3,5) = 4 * ( n.t() * RBK::tilde(point_pairs[i].second -> get_point()) 
				- (dcm_S * point_pairs[i].first -> get_point() + x_S - dcm_D * point_pairs[i].second -> get_point() - x_D).t() * dcm_D * RBK::tilde(n));

		}

		// epsilon = y - Hx !!!
		H_ki = - H_ki;

		Lambda_k += H_ki.t() * H_ki;
		N_k += H_ki.t() * y_ki;

	}

	#if BUNDLE_ADJUSTER_DEBUG
	std::cout << " - Done with this sub-problem\n";
	#endif

}

void BundleAdjuster::update_point_cloud_pairs(){

	double max_rms_error = -1;
	double max_mean_error = -1;

	double mean_rms_error = 0 ;
	int worst_Sk_rms,worst_Dk_rms;
	int worst_Sk_mean,worst_Dk_mean;


	for (int k = 0; k < this -> point_cloud_pairs.size(); ++k){
		
		std::vector<PointPair> point_pairs;

		PointCloudPair point_cloud_pair = this -> point_cloud_pairs[k];

		std::string label_S_k = this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[point_cloud_pair.S_k]) -> get_label();
		std::string label_D_k = this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[point_cloud_pair.D_k]) -> get_label();

		arma::vec x_S = arma::zeros<arma::vec>(3);
		arma::mat dcm_S = arma::eye<arma::mat>(3,3);

		arma::vec x_D = arma::zeros<arma::vec>(3);
		arma::mat dcm_D = arma::eye<arma::mat>(3,3);


		if (point_cloud_pair.S_k != 0){
			x_S = this -> X.subvec(6 * (point_cloud_pair.S_k - 1) , 6 * (point_cloud_pair.S_k - 1) + 2);
			dcm_S =  RBK::mrp_to_dcm(this -> X.subvec(6 * (point_cloud_pair.S_k - 1) + 3, 6 * (point_cloud_pair.S_k - 1) + 5));
		}

		if (point_cloud_pair.D_k != 0){
			x_D = this -> X.subvec(6 * (point_cloud_pair.D_k - 1) , 6 * (point_cloud_pair.D_k - 1) + 2);
			dcm_D = RBK::mrp_to_dcm(this -> X.subvec(6 * (point_cloud_pair.D_k - 1) + 3, 6 * (point_cloud_pair.D_k - 1) + 5));
		}

		ICP::compute_pairs(point_pairs,
			this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[point_cloud_pair.S_k]),
			this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[point_cloud_pair.D_k]),
			this -> h,
			dcm_S ,
			x_S,
			dcm_D ,
			x_D );

		double rms_error = ICP::compute_rms_residuals(point_pairs,
			dcm_S ,
			x_S,
			dcm_D ,
			x_D);


		double mean_error = std::abs(ICP::compute_mean_residuals(point_pairs,
			dcm_S ,
			x_S,
			dcm_D ,
			x_D));


		double p = std::log2(this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[this -> point_cloud_pairs[k].S_k]) -> get_size());
		int N_pairs = (int)(std::pow(2, p - this -> h));

		
		this -> point_cloud_pairs[k].error = rms_error;
		this -> point_cloud_pairs[k].N_accepted_pairs = point_pairs.size();
		this -> point_cloud_pairs[k].N_pairs = N_pairs;


		if (rms_error > max_rms_error){
			max_rms_error = rms_error;
			worst_Dk_rms = this -> point_cloud_pairs[k].D_k;
			worst_Sk_rms = this -> point_cloud_pairs[k].S_k;
		}

		if (mean_error > max_mean_error){
			max_mean_error = mean_error;
			worst_Dk_mean = this -> point_cloud_pairs[k].D_k;
			worst_Sk_mean = this -> point_cloud_pairs[k].S_k;
		}

		mean_rms_error += rms_error / this -> point_cloud_pairs.size();

		
		std::cout << " -- (" << label_S_k << " , " << label_D_k <<  ") : " << mean_error << " , " << rms_error << " | "<< point_pairs.size() << " point pairs" << std::endl;

	}

	std::cout << "-- Mean point-cloud pair ICP RMS error: " << mean_rms_error << std::endl;
	std::cout << "-- Maximum point-cloud pair ICP RMS error at (" << this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[worst_Sk_rms]) -> get_label() << " , " << this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[worst_Dk_rms]) -> get_label() <<  ") : " << max_rms_error << std::endl;
	std::cout << "-- Maximum point-cloud pair ICP mean error at (" << this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[worst_Sk_mean]) -> get_label() << " , " << this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[worst_Dk_mean]) -> get_label() <<  ") : " << max_mean_error << std::endl;


}

void BundleAdjuster::add_subproblem_to_problem(std::vector<T>& coeffs,
	EigVec & N,
	const arma::mat & Lambda_k,
	const arma::vec & N_k,
	const PointCloudPair & point_cloud_pair){

	int S_k = point_cloud_pair.S_k;
	int D_k = point_cloud_pair.D_k;

	
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
			N(6 *(D_k -1) + i) += N_k(i + 6);
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

	boost::progress_display progress(this -> local_pc_index_to_global_pc_index . size() - 1);

	#pragma omp parallel for
	for (unsigned int i = 1; i < this -> local_pc_index_to_global_pc_index . size(); ++i){

		int x_index = 6 * (i - 1);
		int mrp_index = 6 * (i - 1) + 3;

		arma::vec dx  = {deviation(x_index),deviation(x_index + 1),deviation(x_index + 2)};
		
		// The mrp used in the partials 
		// instantiates
		// [NS_bar]
		// but the solved for d_mrp
		// corresponds to 
		// [SS_bar]
		// so need to apply 
		// [S_barS] = dcm_to_mrp(-d_mrp)
		// as in [NS_bar] = [NS_bar] * [S_barS]
		arma::vec d_mrp  = {deviation(mrp_index),deviation(mrp_index + 1),deviation(mrp_index + 2)};
		
		arma::mat SS_bar = RBK::mrp_to_dcm(d_mrp);
		arma::mat NS_bar = RBK::mrp_to_dcm(this -> X.subvec(mrp_index, mrp_index + 2));

		this -> X.subvec(x_index , x_index + 2) += dx;
		this -> X.subvec(mrp_index, mrp_index + 2) = RBK::dcm_to_mrp(NS_bar * SS_bar.t());

		++progress;

	}

}


void BundleAdjuster::update_point_clouds(std::map<int,arma::mat> & M_pcs, 
	std::map<int,arma::vec> & X_pcs,
	std::vector<arma::mat> & BN_estimated,
	const std::vector<arma::vec> & mrps_LN){

	boost::progress_display progress(this -> local_pc_index_to_global_pc_index.size() - 1);

#pragma omp parallel for
	for (unsigned int i = 1; i < this -> local_pc_index_to_global_pc_index.size(); ++i){

		int x_index = 6 * (i - 1);
		int mrp_index = 6 * (i - 1) + 3;

		arma::vec x = this-> X.subvec(x_index , x_index + 2);
		arma::vec mrp = this -> X.subvec(mrp_index, mrp_index + 2);

		arma::mat NS_bar = RBK::mrp_to_dcm(mrp);
		int pc_global_index = this -> local_pc_index_to_global_pc_index[i];

		this -> all_registered_pc -> at(pc_global_index) -> transform(NS_bar, x);

		// The rigid transforms are fixed
		M_pcs[pc_global_index] = NS_bar * M_pcs[pc_global_index];
		X_pcs[pc_global_index] += x;

		// The small body attitude is fixed
		BN_estimated[pc_global_index] = RBK::mrp_to_dcm(mrps_LN[0]).t() * M_pcs[pc_global_index] * RBK::mrp_to_dcm(mrps_LN[pc_global_index]);

		++progress;

	}


}

void BundleAdjuster::save_connectivity() const{
	int M = this -> point_cloud_pairs. size();
	int Q = this -> local_pc_index_to_global_pc_index. size();


	arma::mat connectivity_matrix_res(Q,Q);
	arma::mat connectivity_matrix_overlap(Q,Q);
	arma::mat connectivity_matrix_N_pairs(Q,Q);


	connectivity_matrix_res.fill(-1);
	connectivity_matrix_overlap.fill(-1);

	for (int k = 0; k < M; ++k){
		auto point_cloud_pair = this -> point_cloud_pairs.at(k);



		connectivity_matrix_res(point_cloud_pair.S_k,point_cloud_pair.D_k) = point_cloud_pair.error;
		connectivity_matrix_res(point_cloud_pair.D_k,point_cloud_pair.S_k) = point_cloud_pair.error;

		connectivity_matrix_overlap(point_cloud_pair.S_k,point_cloud_pair.D_k) = double(point_cloud_pair.N_accepted_pairs) / double(point_cloud_pair.N_pairs);
		connectivity_matrix_overlap(point_cloud_pair.D_k,point_cloud_pair.S_k) = double(point_cloud_pair.N_accepted_pairs) / double(point_cloud_pair.N_pairs);

		connectivity_matrix_N_pairs(point_cloud_pair.S_k,point_cloud_pair.D_k) = point_cloud_pair.N_pairs;
		connectivity_matrix_N_pairs(point_cloud_pair.D_k,point_cloud_pair.S_k) = point_cloud_pair.N_pairs;

		this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[point_cloud_pair.S_k]) -> save("../output/pc/source_" + std::to_string(this -> local_pc_index_to_global_pc_index[point_cloud_pair.S_k]) + "_ba.obj",
			this -> LN_t0.t(),
			this -> x_t0);

	}

	connectivity_matrix_res.save("../output/connectivity_res.txt",arma::raw_ascii);
	connectivity_matrix_overlap.save("../output/connectivity_overlap.txt",arma::raw_ascii);
	connectivity_matrix_N_pairs.save("../output/connectivity_N_pairs.txt",arma::raw_ascii);


}

