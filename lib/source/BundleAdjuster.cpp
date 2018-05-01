#include "BundleAdjuster.hpp"
#include <armadillo>
#include "ICP.hpp"
#include "boost/progress.hpp"
#include "DebugFlags.hpp"


BundleAdjuster::BundleAdjuster(std::vector< std::shared_ptr<PC> > * all_registered_pc_, 
	int N_iter,
	arma::mat LN_t0,arma::vec x_t0,
	const arma::mat & longitude_latitude){

	this -> all_registered_pc = all_registered_pc_;
	this -> LN_t0 = LN_t0;
	this -> x_t0 = x_t0;
	this -> N_iter = N_iter;


	// The connectivity between point clouds is inferred
	std::cout << "- Creating point cloud pairs" << std::endl;
	this -> create_pairs(longitude_latitude);

	// This allows to compute the ICP RMS residuals for each considered point-cloud pair before running the bundle adjuster
	this -> update_point_cloud_pairs();

	if (this -> N_iter > 0){
	// solve the bundle adjustment problem
		this -> solve_bundle_adjustment();
		std::cout << "- Solved bundle adjustment" << std::endl;
	}

	// The connectivity matrix is saved
	this -> save_connectivity();
}

void BundleAdjuster::solve_bundle_adjustment(){

	int Q = this -> all_registered_pc -> size();


	for (int iter = 0 ; iter < this -> N_iter; ++iter){

		std::cout << "Iteration: " << std::to_string(iter + 1) << " /" << std::to_string(N_iter) << std::endl;

		std::vector<T> coefficients;          
		EigVec Nmat(6 * (Q - 1)); 
		Nmat.setZero();
		SpMat Lambda(6 * (Q - 1), 6 * (Q - 1));
		arma::vec dX(6 * (Q - 1));

		// For each point-cloud pair
		#if !BUNDLE_ADJUSTER_DEBUG
		boost::progress_display progress(this -> point_cloud_pairs.size());
		#endif 

		for (int k = 0; k < this -> point_cloud_pairs.size(); ++k){

			arma::mat Lambda_k;
			arma::vec N_k;


			if (this -> point_cloud_pairs . at(k).D_k != this -> ground_pc_index && this -> point_cloud_pairs . at(k).S_k != this -> ground_pc_index){
				Lambda_k = arma::zeros<arma::mat>(12,12);
				N_k = arma::zeros<arma::vec>(12);
			}
			else{
				Lambda_k = arma::zeros<arma::mat>(6,6);
				N_k = arma::zeros<arma::vec>(6);
			}

			// The Lambda_k and N_k specific to this point-cloud pair are computed
			this -> assemble_subproblem(Lambda_k,N_k,this -> point_cloud_pairs . at(k));

			// They are added to the whole problem
			this -> add_subproblem_to_problem(coefficients,Nmat,Lambda_k,N_k,this -> point_cloud_pairs . at(k));


			#if !BUNDLE_ADJUSTER_DEBUG
			++progress;
			#else

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

		// dX = arma::spsolve(Lambda,N,"superlu",settings);

		// Transitionning to Eigen
		// The cholesky decomposition of Lambda is computed
		Eigen::SimplicialCholesky<SpMat> chol(Lambda);  

		// The deviation is computed
		EigVec deviation = chol.solve(Nmat);    


		#pragma omp parallel for
		for (unsigned int i = 0; i < 6 * (Q-1); ++i){
			dX(i) = deviation(i);
		}


		// It is applied to all of the point clouds (minus the first one)
		std::cout << "- Applying the deviation" << std::endl;

		this -> apply_deviation(dX);
		std::cout << "\n- Updating the point pairs" << std::endl;

		// The point cloud pairs are updated: their residuals are update
		this -> update_point_cloud_pairs();

		#if BUNDLE_ADJUSTER_DEBUG
		std::cout << "Deviation: " << std::endl;
		std::cout << dX << std::endl;
		#endif

	}


}

void BundleAdjuster::ICP_pass(){

	arma::mat M_pc = arma::eye<arma::mat>(3,3);
	arma::vec X_pc = arma::zeros<arma::vec>(3);

	boost::progress_display progress(this -> all_registered_pc -> size() - 1);

	for (int i = 0; i < this -> all_registered_pc -> size() - 1; ++i){
		ICP icp_pc(this -> all_registered_pc -> at(i), this -> all_registered_pc -> at(i + 1), M_pc, X_pc);

		M_pc = icp_pc.get_M();
		X_pc = icp_pc.get_X();

		this -> all_registered_pc -> at(i + 1) -> transform(M_pc,X_pc);
		++progress;
	}

}


void BundleAdjuster::create_pairs(const arma::mat & longitude_latitude){


	int n_bins_longitude = 72;
	int n_bins_latitude = 36;

	double d_bin_longitude = 360./ n_bins_longitude;
	double d_bin_latitude = 180./ n_bins_latitude;

	std::vector< std::vector< std::vector< int > > > bins;

	#if BUNDLE_ADJUSTER_DEBUG
	std::cout << "-- Creating the empty bin matrix" << std::endl;
	#endif
	for (int i = 0; i < n_bins_latitude; ++i){

		std::vector< std::vector < int > > row;

		for (int j = 0; j < n_bins_longitude; ++j){
			std::vector < int > empty_vector;
			row.push_back(empty_vector);
		}
		bins.push_back(row);

	}

	#if BUNDLE_ADJUSTER_DEBUG
	std::cout << "-- Fetching point clouds in bins" << std::endl;
	#endif

	for (int i = 0; i < longitude_latitude.n_rows; ++i){
		int bin_longitude = int(longitude_latitude(i,0) / d_bin_longitude) + n_bins_longitude/2;
		int bin_latitude = int(longitude_latitude(i,1) / d_bin_latitude) + n_bins_latitude/2;
		bins[n_bins_latitude - bin_latitude][bin_longitude].push_back(i);
	}

	#if BUNDLE_ADJUSTER_DEBUG
	std::cout << "-- Forming pairs" << std::endl;
	#endif

	std::set<std::set<int> > pairs;

	for (int bin_latitude = 0; bin_latitude < n_bins_latitude; ++bin_latitude){
		for (int bin_longitude = 0; bin_longitude < n_bins_longitude; ++bin_longitude){
			
			std::vector < int > pc_in_bin = bins[n_bins_latitude - bin_latitude][bin_longitude];
			

			while(pc_in_bin.size() > 1){
				for (int pc_index = 0; pc_index < pc_in_bin.size() - 1; ++pc_index){
					std::set<int> new_pair;
					new_pair.insert(pc_in_bin[pc_in_bin.back()]);
					new_pair.insert(pc_in_bin[pc_index]);
					pairs.insert(new_pair);

				}
				pc_in_bin.pop_back();
			}

		}
	}
	#if BUNDLE_ADJUSTER_DEBUG

	std::cout << "-- Number of pairs: " << pairs.size() << std::endl;
	std::cout << "-- Storing pairs" << std::endl;
	#endif

	for (auto pair_iter = pairs.begin(); pair_iter != pairs.end(); ++pair_iter){

		int S_k =(*pair_iter-> begin());
		int D_k =(*std::next(pair_iter-> begin()));
		int h = 0;
		std::vector<PointPair> point_pairs;

		ICP::compute_pairs(point_pairs,this -> all_registered_pc -> at(S_k),this -> all_registered_pc -> at(D_k),h);				
		double error = ICP::compute_rms_residuals(point_pairs);

		double p = std::log2(this -> all_registered_pc -> at(S_k) -> get_size());
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

// void BundleAdjuster::find_point_cloud_pairs(){

// 	int M = this -> all_registered_pc -> size();

// 	#if BUNDLE_ADJUSTER_DEBUG

// 	std::cout << "Number of registered point clouds: " << M << std::endl;

// 	#endif

// 	std::vector<PointPair> point_pairs;

// 	double min_overlap = std::numeric_limits<double>::infinity();
// 	double max_error = - std::numeric_limits<double>::infinity();

// 	this -> ground_pc_index = 0;


// 	for (int i = 1; i < M ; ++i){
// 		int h = 5;

// 		ICP::compute_pairs(point_pairs,this -> all_registered_pc -> at(i),this -> all_registered_pc -> at(i-1),h);				
// 		double error = ICP::compute_rms_residuals(point_pairs);

// 		double p = std::log2(this -> all_registered_pc -> at(i) -> get_size());
// 		int N_pairs = (int)(std::pow(2, p - h));

// 		BundleAdjuster::PointCloudPair pair;
// 		pair.S_k = i;
// 		pair.D_k = i - 1;
// 		pair.error = error;
// 		pair.N_pairs = N_pairs;
// 		pair.N_accepted_pairs = point_pairs.size();

// 		this -> point_cloud_pairs.push_back(pair);

// 		if (this -> all_registered_pc -> at(i) -> get_size() > this -> all_registered_pc -> at(this -> ground_pc_index) -> get_size()){
// 			this -> ground_pc_index = i;
// 		}



// 	}

// 	std::cout << "- Using point cloud " << this -> ground_pc_index << " as ground point cloud\n";
// 	std::cout << "- Longitude/latitude pairing\n";
// 	boost::progress_display progress(M);

// 	for (int j = 0; j < M; ++j){

// 		// The ground pc index cannot be registered to itself
// 		// Neither can it be registered again to a pc it has already been paired with
// 		if (std::abs(j - this -> ground_pc_index) < 3){
// 			continue;
// 		}

// 		try{

// 			int h = 4;

// 			ICP::compute_pairs(point_pairs,this -> all_registered_pc -> at(this -> ground_pc_index),this -> all_registered_pc -> at(j),h);				
// 			double error = ICP::compute_rms_residuals(point_pairs);

// 			double p = std::log2(this -> all_registered_pc -> at(this -> ground_pc_index) -> get_size());
// 			int N_pairs = (int)(std::pow(2, p - h));

// 			BundleAdjuster::PointCloudPair pair;
// 			pair.S_k = this -> ground_pc_index;
// 			pair.D_k = j;
// 			pair.error = error;
// 			pair.N_pairs = N_pairs;
// 			pair.N_accepted_pairs = point_pairs.size();

// 				// Restricting loop closure to 
// 				// pairs featuring 0 as one of their point clouds
// 			if (double(point_pairs.size()) / double(N_pairs) > min_overlap ){
// 				#if BUNDLE_ADJUSTER_DEBUG
// 				std::cout << "Using pair " << this -> ground_pc_index << " / " << j << " for loop closure" << std::endl;
// 				#endif
// 				this -> point_cloud_pairs.push_back(pair);
// 				break;
// 			}


// 		}
// 		catch(ICPNoPairsException & e){

// 		}
// 		catch(ICPException & e){

// 		}
// 		++progress;

// 	}



// 	std::cout << "\nNumber of point cloud pairs: " << this -> point_cloud_pairs.size() << std::endl;


// }





void BundleAdjuster::assemble_subproblem(arma::mat & Lambda_k,arma::vec & N_k,const PointCloudPair & point_cloud_pair){

	// The point-pairs in the prescribed point-cloud pair are formed (with h = 0, so we are using them all)
	std::vector<PointPair> point_pairs;

	// The point pairs must be computed using the current estimate of the point clouds' rigid transform
	ICP::compute_pairs(point_pairs,this -> all_registered_pc -> at(point_cloud_pair.S_k),this -> all_registered_pc -> at(point_cloud_pair.D_k),0);		


	#if BUNDLE_ADJUSTER_DEBUG
	std::cout << " - Subproblem : " << point_cloud_pair.S_k << " / " << point_cloud_pair.D_k << std::endl;
	std::cout << " - Number of pairs: " << point_pairs.size() << std::endl;
	std::cout << " - Residuals: " << ICP::compute_rms_residuals(point_pairs) << std::endl;
	#endif

	arma::rowvec H_ki;


	if (point_cloud_pair.D_k != this -> ground_pc_index && point_cloud_pair.S_k != this -> ground_pc_index){
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

		double y_ki = ICP::compute_normal_distance(point_pairs[i]);
		arma::mat n = point_pairs[i].second -> get_normal();



		if (point_cloud_pair.D_k != this -> ground_pc_index && point_cloud_pair.S_k != this -> ground_pc_index){

			H_ki.subvec(0,2) = n.t();
			H_ki.subvec(3,5) = ICP::dGdSigma_multiplicative(arma::zeros<arma::vec>(3),point_pairs[i].first -> get_point(),n);
			H_ki.subvec(6,8) = - n.t();
			H_ki.subvec(9,11) = 4 * ( - n.t() * RBK::tilde(point_pairs[i].second -> get_point()) 
				+ (point_pairs[i].first -> get_point() - point_pairs[i].second -> get_point()).t() * RBK::tilde(n));

		}

		else if(point_cloud_pair.S_k != this -> ground_pc_index) {
			H_ki.subvec(0,2) = n.t();
			H_ki.subvec(3,5) = ICP::dGdSigma_multiplicative(arma::zeros<arma::vec>(3),point_pairs[i].first -> get_point(),n);

		}

		else{
			H_ki.subvec(0,2) = - n.t();
			H_ki.subvec(3,5) = 4 * ( - n.t() * RBK::tilde(point_pairs[i].second -> get_point()) 
				+ (point_pairs[i].first -> get_point() - point_pairs[i].second -> get_point()).t() * RBK::tilde(n));

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

		ICP::compute_pairs(point_pairs,
			this -> all_registered_pc -> at(this -> point_cloud_pairs[k].S_k),
			this -> all_registered_pc -> at(this -> point_cloud_pairs[k].D_k),
			h);

		double rms_error = ICP::compute_rms_residuals(point_pairs);
		double mean_error = std::abs(ICP::compute_mean_residuals(point_pairs));


		double p = std::log2(this -> all_registered_pc -> at(this -> point_cloud_pairs[k].S_k) -> get_size());
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
		

		std::cout << "(" << this -> point_cloud_pairs[k].S_k << " , " <<this -> point_cloud_pairs[k].D_k <<  ") : " << mean_error << " , " << rms_error << std::endl;


	}

	std::cout << "-- Mean point-cloud pair ICP RMS error: " << mean_rms_error << std::endl;
	std::cout << "-- Maximum point-cloud pair ICP RMS error at (" << worst_Sk_rms << " , " <<worst_Dk_rms <<  ") : " << max_rms_error << std::endl;
	std::cout << "-- Maximum point-cloud pair ICP mean error at (" << worst_Sk_mean << " , " <<worst_Dk_mean <<  ") : " << max_mean_error << std::endl;


}

void BundleAdjuster::add_subproblem_to_problem(std::vector<T>& coeffs,
	EigVec & N,
	const arma::mat & Lambda_k,
	const arma::vec & N_k,
	const PointCloudPair & point_cloud_pair){

	int S_k = point_cloud_pair.S_k;
	int D_k = point_cloud_pair.D_k;

	int offset_S_k = 0;
	int offset_D_k = 0;
	
	if (S_k > this -> ground_pc_index){
		offset_S_k = -6;
	}

	if (D_k > this -> ground_pc_index){
		offset_D_k = -6;
	}

	
	if (D_k != this -> ground_pc_index && S_k != this -> ground_pc_index){

		// S_k substate
		for(unsigned int i = 0; i < 6; ++i){
			for(unsigned int j = 0; j < 6; ++j){
				coeffs.push_back(T(6 * S_k + offset_S_k + i, 6 * S_k + offset_S_k + j,Lambda_k(i,j)));
			}
			N(6 * S_k + offset_S_k + i) += N_k(i);
		}
		
		// D_k substate
		for(unsigned int i = 0; i < 6; ++i){
			for(unsigned int j = 0; j < 6; ++j){
				coeffs.push_back(T(6 * D_k + offset_D_k + i, 6 * D_k + offset_D_k + j,Lambda_k(i + 6,j + 6)));
			}
			N(6 * D_k + offset_D_k + i) += N_k(i + 6);
		}



		// Cross-correlations
		for(unsigned int i = 0; i < 6; ++i){
			for(unsigned int j = 0; j < 6; ++j){
				coeffs.push_back(T(6 * S_k + offset_S_k + i, 6 * D_k + offset_D_k + j,Lambda_k(i,j + 6)));
				coeffs.push_back(T(6 * D_k + offset_D_k + i, 6 * S_k + offset_S_k + j,Lambda_k(i + 6,j)));

			}
		}

	}

	else if (S_k != this -> ground_pc_index){

		// S_k substate
		for(unsigned int i = 0; i < 6; ++i){
			for(unsigned int j = 0; j < 6; ++j){
				coeffs.push_back(T(6 * S_k + offset_S_k + i, 6 * S_k + offset_S_k + j,Lambda_k(i,j)));
			}
			N(6 * S_k + offset_S_k + i) += N_k(i);
		}


	}
	else {

		// D_k substate
		for(unsigned int i = 0; i < 6; ++i){
			for(unsigned int j = 0; j < 6; ++j){
				coeffs.push_back(T(6 * D_k + offset_D_k + i, 6 * D_k + offset_D_k + j,Lambda_k(i,j)));
			}
			N(6 * D_k + offset_D_k + i) += N_k(i);
		}
	}


}

void BundleAdjuster::apply_deviation(const arma::vec & dX){

	boost::progress_display progress(this -> all_registered_pc -> size());

	#pragma omp parallel for
	for (unsigned int i = 0; i < this -> all_registered_pc -> size(); ++i){


		if (i < this -> ground_pc_index){

			this -> all_registered_pc -> at(i) -> transform(
				RBK::mrp_to_dcm(dX.subvec(6 * i + 3, 6 * i + 5)), 
				dX.subvec(6 * i , 6 * i + 2));

		}
		else if (i > this -> ground_pc_index){

			this -> all_registered_pc -> at(i) -> transform(
				RBK::mrp_to_dcm(dX.subvec(6 * (i - 1) + 3, 6 * (i - 1) + 5)), 
				dX.subvec(6 * (i - 1) , 6 * (i - 1) + 2));

		}

		++progress;

	}

}


void BundleAdjuster::save_connectivity() const{
	int M = this -> point_cloud_pairs. size();
	int Q = this -> all_registered_pc -> size();


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

		this -> all_registered_pc -> at(point_cloud_pair.S_k) -> save("../output/pc/source_" + std::to_string(point_cloud_pair.S_k) + "_ba.obj",this -> LN_t0.t(),this -> x_t0);

	}

	connectivity_matrix_res.save("../output/connectivity_res.txt",arma::raw_ascii);
	connectivity_matrix_overlap.save("../output/connectivity_overlap.txt",arma::raw_ascii);
	connectivity_matrix_N_pairs.save("../output/connectivity_N_pairs.txt",arma::raw_ascii);


}

