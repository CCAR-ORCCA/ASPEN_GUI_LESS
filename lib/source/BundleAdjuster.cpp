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
	bool save_connectivity){


	this -> all_registered_pc = all_registered_pc_;
	this -> LN_t0 = LN_t0;
	this -> x_t0 = x_t0;
	this -> N_iter = N_iter;


	for (int i = t0; i <= tf; ++i){
		this -> local_pc_index_to_global_pc_index.push_back(i);
	}


	int Q = this -> local_pc_index_to_global_pc_index. size();

	this -> X = arma::zeros<arma::vec>(6 * (Q - 1));


	// The connectivity between point clouds is inferred
	std::cout << "- Creating point cloud pairs" << std::endl;
	this -> create_pairs();

	// This allows to compute the ICP RMS residuals for each considered point-cloud pair before running the bundle adjuster
	this -> update_point_cloud_pairs();

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
	// if (save_connectivity){
	this -> save_connectivity();
	// }
}








// BundleAdjuster::BundleAdjuster(
// 	std::vector<arma::mat> & M_pcs,
// 	std::vector<arma::vec> & X_pcs,
// 	std::vector< std::shared_ptr<PC> > * all_registered_pc_, 
// 	int N_iter,
// 	FlyOverMap * fly_over_map,
// 	arma::mat & longitude_latitude,
// 	const arma::mat & LN_t0,
// 	const arma::vec & x_t0,
// 	bool look_for_closure,
// 	bool save_connectivity){

// 	this -> fly_over_map = fly_over_map;
// 	this -> all_registered_pc = all_registered_pc_;
// 	this -> LN_t0 = LN_t0;
// 	this -> x_t0 = x_t0;
// 	this -> N_iter = N_iter;

// 	int Q = this -> all_registered_pc -> size();

// 	this -> X = arma::zeros<arma::vec>(6 * (Q - 1));

// 	for (unsigned int i = 0; i < this -> all_registered_pc -> size() -1 ; ++i){
// 		this -> rotation_increment.push_back(arma::eye<arma::mat>(3,3));
// 		this -> position_increment.push_back(arma::zeros<arma::vec>(3));
// 	}

// 	// The connectivity between point clouds is inferred
// 	std::cout << "- Creating point cloud pairs" << std::endl;
// 	this -> create_pairs(look_for_closure);

// 	// This allows to compute the ICP RMS residuals for each considered point-cloud pair before running the bundle adjuster
// 	this -> update_point_cloud_pairs();

// 	if (this -> N_iter > 0){
// 	// solve the bundle adjustment problem
// 		this -> solve_bundle_adjustment();
// 		std::cout << "- Solved bundle adjustment" << std::endl;
// 	}	


// 	std::cout << "- Updating point clouds ... " << std::endl;
// 	this -> update_point_clouds( M_pcs,X_pcs);
// 	std::cout << "\n- Updating flyover map ... " << std::endl;
// 	this -> update_flyover_map(longitude_latitude);

// 	// The connectivity matrix is saved
// 	if (save_connectivity){
// 		this -> save_connectivity();
// 	}
// }





void BundleAdjuster::update_flyover_map(arma::mat & longitude_latitude){

	// // Updating the pcs. The first one is fixed
	// for (int pc = 1; pc < this -> all_registered_pc -> size(); ++pc){


	// 	std::string label = this -> all_registered_pc -> at(pc) -> get_label();
	// 	arma::vec old_los = {0,0,0};
	// 	arma::vec new_los = {0,0,0};
	// 	arma::rowvec long_lat = longitude_latitude.row(std::stoi( label));
	// 	double old_longitude = long_lat(0);
	// 	double old_latitude = long_lat(1);

	// 	double new_longitude,new_latitude;

	// 	if (old_longitude > 0){
	// 		if (std::abs(old_longitude) <= 90){
	// 			old_los(0) = 1;
	// 			old_los(1) = std::tan(arma::datum::pi / 180 * old_longitude);

	// 		}
	// 		else{
	// 			old_los(0) = -1;
	// 			old_los(1) = -std::tan(arma::datum::pi / 180 * old_longitude);
	// 		}
	// 	}
	// 	else{
	// 		if (std::abs(old_longitude) <= 90){
	// 			old_los(0) = 1;
	// 			old_los(1) = std::tan(arma::datum::pi / 180 * old_longitude);
	// 		}
	// 		else{
	// 			old_los(0) = -1;
	// 			old_los(1) = -std::tan(arma::datum::pi / 180 * old_longitude);
	// 		}
	// 	}

	// 	old_los(2) = arma::norm(old_los.subvec(0,1)) * std::tan(arma::datum::pi / 180 * old_latitude );
	// 	old_los = arma::normalise(old_los);
	// 	new_los = this -> rotation_increment.at(pc - 1) * old_los;

	// 	new_longitude = 180. / arma::datum::pi * std::atan2(new_los(1),new_los(0));
	// 	new_latitude = 180. / arma::datum::pi * std::atan(new_los(2)/arma::norm(new_los.subvec(0,1)));

	// 	longitude_latitude(std::stoi(label),0) = new_longitude;
	// 	longitude_latitude(std::stoi(label),1) = new_latitude;
	// 	this -> fly_over_map -> update_label(std::stoi(label),new_longitude,new_latitude);
	// }

}



void BundleAdjuster::solve_bundle_adjustment(){

	int Q = this -> all_registered_pc -> size();

	this -> X = arma::zeros<arma::vec>(6 * (Q - 1));

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


void BundleAdjuster::create_pairs( bool look_for_closure){

	std::vector<PointPair> point_pairs;
	std::set<std::set<int> > pairs;
	int h = 5;

	int tf = local_pc_index_to_global_pc_index.back(); 

	// Checking possible closure between current point cloud and first cloud
	for (int closure_index = 0; closure_index < tf; ++closure_index){

		ICP::compute_pairs(point_pairs,
			this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[tf]),
			this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[closure_index]),
			h);

		int p = std::log2(this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[tf]) -> get_size());

		int N_pairs = (int)(std::pow(2, p - h));

		double prop = double(point_pairs.size()) / N_pairs * 100;

		std::cout << " ( " << this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[tf]) -> get_label() << " , "<<
		this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[closure_index]) -> get_label() << " ) : " << point_pairs.size() << " , " << prop << std::endl;
		
		if (prop > 70){
			std::cout << "Choosing " << " ( " << this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[tf]) -> get_label() << " , "<<
			this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[closure_index]) -> get_label() << " ) in loop closure" std::endl;

			std::set<int> pair = {tf,closure_index};
			pairs.insert(pair);
			break;

		}

	}

	// if (look_for_closure){
	// 	pairs = this -> fly_over_map -> get_flyovers();

	// 	std::cout << " -- Flyover pairs: \n";
	// 	for (auto iter_pair = pairs.begin(); iter_pair != pairs.end(); ++iter_pair){
	// 		std::set<int> pair = *iter_pair;
	// 		int S_k = *pair.begin();
	// 		int D_k = *std::next(pair.begin());
	// 		std::string label_S_k = this -> all_registered_pc -> at(S_k) -> get_label();
	// 		std::string label_D_k = this -> all_registered_pc -> at(D_k) -> get_label();


	// 		std::cout << "(" << label_S_k << "," << label_D_k << ")" << std::endl;
	// 	}
	// }

	
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

		int h = 4;
		std::vector<PointPair> point_pairs;

		ICP::compute_pairs(point_pairs,
			this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[S_k]),
			this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[D_k]),h);				
		double error = ICP::compute_rms_residuals(point_pairs);

		double p = std::log2(this -> all_registered_pc -> at(this -> local_pc_index_to_global_pc_index[S_k]) -> get_size());
		int N_pairs = (int)(std::pow(2, p - h));

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
		int h = 4;

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
			h,
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
		int N_pairs = (int)(std::pow(2, p - h));

		
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

		
		std::cout << " -- (" << label_S_k << " , " << label_D_k <<  ") : " << mean_error << " , " << rms_error << " | "<< point_pairs.size() << std::endl;

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

