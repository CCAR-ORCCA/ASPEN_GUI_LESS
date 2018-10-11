#include "BundleAdjuster.hpp"
#include <armadillo>
#include "IterativeClosestPointToPlane.hpp"
#include "boost/progress.hpp"
#include <PointCloud.hpp>
#include <PointNormal.hpp>
#include <set>
#include <PointCloudIO.hpp>

#define BUNDLE_ADJUSTER_DEBUG 1


BundleAdjuster::BundleAdjuster(
	int t0, 
	int tf,
	std::vector< std::shared_ptr<PointCloud<PointNormal > > > * all_registered_pc_, 
	int N_iter,
	int h,
	const arma::mat & LN_t0,
	const arma::vec & x_t0,
	std::string dir){

	this -> all_registered_pc = all_registered_pc_;
	this -> LN_t0 = LN_t0;
	this -> x_t0 = x_t0;
	this -> N_iter = N_iter;
	this -> h = h;
	this -> dir = dir;

	
	

}


void BundleAdjuster::set_use_true_pairs(bool use_true_pairs){
	this -> use_true_pairs = use_true_pairs;
}



void BundleAdjuster::run(
	std::map<int,arma::mat> & M_pcs,
	std::map<int,arma::vec> & X_pcs,
	std::vector<arma::mat> & BN_measured,
	const std::vector<arma::vec> & mrps_LN,
	bool save_connectivity,
	int & previous_closure_index){

	std::cout << "- Creating point cloud pairs" << std::endl;
	this -> create_pairs(previous_closure_index);
	this -> update_point_cloud_pairs();

	if (this -> all_registered_pc -> size() == 0){
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
		BN_measured,
		mrps_LN);
	

	// The connectivity matrix is saved
	if (save_connectivity){
		this -> save_connectivity();
		// The rigid transforms are saved after being adjusted
		for (int i = 1; i < M_pcs.size(); ++i){
			RBK::dcm_to_mrp(M_pcs[i]).save(this -> dir + "/sigma_tilde_after_ba_" + std::to_string(i) + ".txt",arma::raw_ascii);
			X_pcs[i].save(this -> dir + "/X_tilde_after_ba_" + std::to_string(i) + ".txt",arma::raw_ascii);

		}
	}
}


void BundleAdjuster::solve_bundle_adjustment(){

	int Q = this -> all_registered_pc -> size();
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
			this -> assemble_subproblem(Lambda_k_vector. at(k),
				N_k_vector. at(k),
				this -> point_cloud_pairs . at(k));
			#if !BUNDLE_ADJUSTER_DEBUG
			++progress;
			#endif

		}

		for (int k = 0; k < this -> point_cloud_pairs.size(); ++k){




			// They are added to the whole problem
			this -> add_subproblem_to_problem(coefficients,
				Nmat,Lambda_k_vector. at(k),
				N_k_vector. at(k),
				this -> point_cloud_pairs . at(k));


		}	

		
		std::cout << "\n- Solving for the deviation" << std::endl;

		// The deviation in all of the rigid transforms is computed
		Lambda.setFromTriplets(coefficients.begin(), coefficients.end());
		
		// The cholesky decomposition of Lambda is computed
		Eigen::SimplicialCholesky<SpMat> chol(Lambda);  

		// The deviation is computed
		EigVec deviation = chol.solve(Nmat);    

		// It is applied to all of the point clouds (minus the first one)
		std::cout << "\n- Applying the deviation" << std::endl;

		this -> apply_deviation(deviation);
		std::cout << "\n- Updating the point cloud pairs" << std::endl;

		// The point cloud pairs are updated: their residuals are updated
		// and the rigid transforms positioning them are also updated
		this -> update_point_cloud_pairs();

	}


}

int BundleAdjuster::get_cutoff_index() const{
	return this -> closure_index;
}

void BundleAdjuster::create_pairs(int & previous_closure_index){

	std::vector<PointPair> point_pairs;
	std::set<std::set<int> > pairs;

	int ground_index = 0; 
	this -> closure_index = this -> all_registered_pc -> size() - 1;

	int pc_matching_with_ground_index = this -> find_overlap_with_pc(
		ground_index,
		static_cast<int>(this -> all_registered_pc -> size() - 1),
		0);

	int pc_matching_with_closure_index = this -> find_overlap_with_pc(
		this -> closure_index,
		0,
		static_cast<int>(this -> all_registered_pc -> size() - 1));

	std::cout << "Choosing " << " ( " << ground_index << " , "<<
	pc_matching_with_ground_index << " ) in loop closure" <<  std::endl;
	
	std::cout << "Choosing " << " ( " << pc_matching_with_closure_index << " , "<<
	this -> closure_index << " ) in loop closure" <<  std::endl;

	std::set<int> pair_0 = {ground_index,pc_matching_with_ground_index};
	std::set<int> pair_1 = {pc_matching_with_closure_index,this -> closure_index};
	pairs.insert(pair_0);
	pairs.insert(pair_1);


	// The successive measurements are added
	for (int i = 0; i < this -> all_registered_pc -> size() - 1; ++i){
		std::set<int> pair = {i,i+1};
		pairs.insert(pair);
	}


	#if BUNDLE_ADJUSTER_DEBUG
	std::cout << " -- Number of pairs: " << pairs.size() << std::endl;
	std::cout << " -- Storing pairs" << std::endl;
	#endif

	for (auto pair_iter = pairs.begin(); pair_iter != pairs.end(); ++pair_iter){

		std::set<int> pair_set = *pair_iter;

		int S_k = *(pair_set.begin());
		int D_k = *(--pair_set.end());

		BundleAdjuster::PointCloudPair pair;
		pair.S_k = S_k;
		pair.D_k = D_k;
		this -> point_cloud_pairs.push_back(pair);

	}


}

void BundleAdjuster::assemble_subproblem(arma::mat & Lambda_k,arma::vec & N_k,const PointCloudPair & point_cloud_pair){

	// The point-pairs in the prescribed point-cloud pair are formed (with h = 0, so we are using them all)
	std::vector<PointPair> point_pairs;

	// The point pairs must be computed using the current estimate of the point clouds' rigid transform

	arma::vec::fixed<3> x_S = arma::zeros<arma::vec>(3);
	arma::mat::fixed<3,3> dcm_S = arma::eye<arma::mat>(3,3);

	arma::vec::fixed<3> x_D = arma::zeros<arma::vec>(3);
	arma::mat::fixed<3,3> dcm_D = arma::eye<arma::mat>(3,3);

	if (point_cloud_pair.S_k != 0){
		x_S = this -> X.subvec(6 * (point_cloud_pair.S_k - 1) , 6 * (point_cloud_pair.S_k - 1) + 2);
		dcm_S =  RBK::mrp_to_dcm(this -> X.subvec(6 * (point_cloud_pair.S_k - 1) + 3, 6 * (point_cloud_pair.S_k - 1) + 5));
	}

	if (point_cloud_pair.D_k != 0){
		x_D = this -> X.subvec(6 * (point_cloud_pair.D_k - 1) , 6 * (point_cloud_pair.D_k - 1) + 2);
		dcm_D = RBK::mrp_to_dcm(this -> X.subvec(6 * (point_cloud_pair.D_k - 1) + 3, 6 * (point_cloud_pair.D_k - 1) + 5));
	}
	
	if (!this -> use_true_pairs){
		
		IterativeClosestPointToPlane::compute_pairs(point_pairs,
			this -> all_registered_pc -> at(point_cloud_pair.S_k),
			this -> all_registered_pc -> at(point_cloud_pair.D_k),
			0,
			dcm_S,
			x_S,
			dcm_D,
			x_D);	
	}
	else{
		

		for (int i = 0; i < this -> all_registered_pc -> at(point_cloud_pair.S_k) -> size(); ++i){
			point_pairs.push_back(std::make_pair(i,i));
		}
	}	


	
	arma::rowvec H_ki;


	if (point_cloud_pair.D_k != 0 && point_cloud_pair.S_k != 0){
		H_ki = arma::zeros<arma::rowvec>(12);
	}
	else{
		H_ki = arma::zeros<arma::rowvec>(6);
	}


	// For all the point pairs that where formed
	for (unsigned int i = 0; i < point_pairs.size(); ++i){


		double y_ki = IterativeClosestPointToPlane::compute_distance(point_pairs[i],dcm_S,x_S,dcm_D,x_D,
			this -> all_registered_pc -> at(point_cloud_pair.S_k),
			this -> all_registered_pc -> at(point_cloud_pair.D_k));
		

		const PointNormal & p_S = this -> all_registered_pc -> at(point_cloud_pair.S_k) -> get_point(point_pairs[i].first);
		const PointNormal & p_D = this -> all_registered_pc -> at(point_cloud_pair.D_k) -> get_point(point_pairs[i].second);

		const arma::vec::fixed<3> & n = p_D.get_normal_coordinates();

		if (point_cloud_pair.D_k != 0 && point_cloud_pair.S_k != 0){

			H_ki.subvec(0,2) = n.t() * dcm_D.t();
			H_ki.subvec(3,5) = - 4 * n.t() * dcm_D.t() * dcm_S * RBK::tilde(p_S.get_point_coordinates());
			H_ki.subvec(6,8) = - n.t() * dcm_D.t();
			H_ki.subvec(9,11) = 4 * ( n.t() * RBK::tilde(p_D.get_point_coordinates()) 
				- (dcm_S * p_S.get_point_coordinates() + x_S - dcm_D * p_D.get_point_coordinates() - x_D).t() * dcm_D * RBK::tilde(n));
			// think I fixed a sign error
		}

		else if(point_cloud_pair.S_k != 0) {
			H_ki.subvec(0,2) = n.t() * dcm_D.t();
			H_ki.subvec(3,5) = - 4 * n.t() * dcm_D.t() * dcm_S * RBK::tilde(p_S.get_point_coordinates());
		}

		else{
			H_ki.subvec(0,2) = - n.t() * dcm_D.t();
			H_ki.subvec(3,5) = 4 * ( n.t() * RBK::tilde(p_D.get_point_coordinates()) 
				- (dcm_S * p_S.get_point_coordinates() + x_S - dcm_D * p_D.get_point_coordinates() - x_D).t() * dcm_D * RBK::tilde(n));
		}

		// epsilon = y - Hx !!!
		H_ki = - H_ki;

		Lambda_k +=  H_ki.t() * H_ki;
		N_k +=  H_ki.t() * y_ki;

	}


}

void BundleAdjuster::update_point_cloud_pairs(){

	double max_rms_error = -1;
	double max_mean_error = -1;

	double mean_rms_error = 0 ;
	int worst_Sk_rms,worst_Dk_rms;
	int worst_Sk_mean,worst_Dk_mean;


	for (int k = 0; k < this -> point_cloud_pairs.size(); ++k){
		

		const BundleAdjuster::PointCloudPair & point_cloud_pair = this -> point_cloud_pairs[k];


		arma::vec::fixed<3> x_S = arma::zeros<arma::vec>(3);
		arma::mat::fixed<3,3> dcm_S = arma::eye<arma::mat>(3,3);

		arma::vec::fixed<3> x_D = arma::zeros<arma::vec>(3);
		arma::mat::fixed<3,3> dcm_D = arma::eye<arma::mat>(3,3);


		if (point_cloud_pair.S_k != 0){
			x_S = this -> X.subvec(6 * (point_cloud_pair.S_k - 1) , 6 * (point_cloud_pair.S_k - 1) + 2);
			dcm_S =  RBK::mrp_to_dcm(this -> X.subvec(6 * (point_cloud_pair.S_k - 1) + 3, 6 * (point_cloud_pair.S_k - 1) + 5));
		}

		if (point_cloud_pair.D_k != 0){
			x_D = this -> X.subvec(6 * (point_cloud_pair.D_k - 1) , 6 * (point_cloud_pair.D_k - 1) + 2);
			dcm_D = RBK::mrp_to_dcm(this -> X.subvec(6 * (point_cloud_pair.D_k - 1) + 3, 6 * (point_cloud_pair.D_k - 1) + 5));
		}

		assert(point_cloud_pair.D_k != 0);


		std::vector<PointPair> point_pairs;

		#if BUNDLE_ADJUSTER_DEBUG
		std::cout << "Computing point pairs within point cloud pair  S_k = " << point_cloud_pair.S_k << " , D_k = " << point_cloud_pair.D_k << std::endl;
		#endif

		if (!this -> use_true_pairs){
			IterativeClosestPointToPlane::compute_pairs(point_pairs,
				this -> all_registered_pc -> at(point_cloud_pair.S_k),
				this -> all_registered_pc -> at(point_cloud_pair.D_k),
				this -> h,
				dcm_S ,
				x_S,
				dcm_D ,
				x_D );
		}
		else{

			

			for (int i = 0; i < this -> all_registered_pc -> at(point_cloud_pair.S_k) -> size(); ++i){
				point_pairs.push_back(std::make_pair(i,i));
			}
		}

		IterativeClosestPointToPlane icp;

		icp.set_pc_destination(this -> all_registered_pc -> at(point_cloud_pair.D_k));
		icp.set_pc_source(this -> all_registered_pc -> at(point_cloud_pair.S_k));
		icp.set_pairs(point_pairs);

		double rms_error = icp.compute_rms_residuals(point_pairs,
			dcm_S ,
			x_S,
			{},
			dcm_D ,
			x_D);


		double mean_error = std::abs(icp.compute_mean_residuals(point_pairs,
			dcm_S ,
			x_S,
			{},
			dcm_D ,
			x_D));


		double p = std::log2(this -> all_registered_pc -> at(this -> point_cloud_pairs[k].S_k) -> size());
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

		std::cout << " -- (" << point_cloud_pair.S_k << " , " << point_cloud_pair.D_k <<  ") : " << mean_error << " , " << rms_error << " | "<< point_pairs.size() << " point pairs" << std::endl;

	}

	std::cout << "-- Mean point-cloud pair ICP RMS error: " << mean_rms_error << std::endl;
	std::cout << "-- Maximum point-cloud pair ICP RMS error at (" << worst_Sk_rms << " , " << worst_Dk_rms <<  ") : " << max_rms_error << std::endl;
	std::cout << "-- Maximum point-cloud pair ICP mean error at (" << worst_Sk_mean << " , " << worst_Dk_mean <<  ") : " << max_mean_error << std::endl;

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


	boost::progress_display progress(this -> all_registered_pc -> size());
	++progress;
	#pragma omp parallel for
	for (unsigned int i = 1; i < this -> all_registered_pc -> size(); ++i){

		int x_index = 6 * (i - 1);
		int mrp_index = 6 * (i - 1) + 3;

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

		this -> X.subvec(x_index , x_index + 2) += dx;
		this -> X.subvec(mrp_index, mrp_index + 2) = RBK::dcm_to_mrp(NS_bar * SS_bar.t());

		++progress;

	}

}


void BundleAdjuster::update_point_clouds(std::map<int,arma::mat> & M_pcs, 
	std::map<int,arma::vec> & X_pcs,
	std::vector<arma::mat> & BN_measured,
	const std::vector<arma::vec> & mrps_LN){

	boost::progress_display progress(this -> all_registered_pc -> size());
	++progress;
	#pragma omp parallel for
	for (unsigned int i = 1; i < this -> all_registered_pc -> size(); ++i){

		int x_index = 6 * (i - 1);
		int mrp_index = 6 * (i - 1) + 3;

		const arma::vec::fixed<3> & x = this-> X.subvec(x_index , x_index + 2);
		const arma::vec::fixed<3> & mrp = this -> X.subvec(mrp_index, mrp_index + 2);

		arma::mat::fixed<3,3> NS_bar = RBK::mrp_to_dcm(mrp);
		
		
		this -> all_registered_pc -> at(i) -> transform(NS_bar, x);
		this -> all_registered_pc -> at(i) -> build_kdtree(false);

		// The rigid transforms are fixed
		M_pcs[i] = NS_bar * M_pcs[i];
		X_pcs[i] += x;


		// The small body attitude is fixed
		// M_pc(k) is [LB](t_0) * [BL](t_k) = [LN](t_0)[NB](t_0) * [BN](t_k) * [NL](t_k);
		BN_measured[i] = BN_measured[0] *  RBK::mrp_to_dcm( mrps_LN[0]).t() * M_pcs[i] * RBK::mrp_to_dcm(mrps_LN[i]);

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

		// connectivity_matrix_res(point_cloud_pair.S_k,point_cloud_pair.D_k) = point_cloud_pair.error;
		// connectivity_matrix_res(point_cloud_pair.D_k,point_cloud_pair.S_k) = point_cloud_pair.error;

		// connectivity_matrix_overlap(point_cloud_pair.S_k,point_cloud_pair.D_k) = double(point_cloud_pair.N_accepted_pairs) / double(point_cloud_pair.N_pairs);
		// connectivity_matrix_overlap(point_cloud_pair.D_k,point_cloud_pair.S_k) = double(point_cloud_pair.N_accepted_pairs) / double(point_cloud_pair.N_pairs);

		// connectivity_matrix_N_pairs(point_cloud_pair.S_k,point_cloud_pair.D_k) = point_cloud_pair.N_pairs;
		// connectivity_matrix_N_pairs(point_cloud_pair.D_k,point_cloud_pair.S_k) = point_cloud_pair.N_pairs;

		if (point_cloud_pair.D_k != 0){
			PointCloudIO<PointNormal>::save_to_obj(
				*this -> all_registered_pc -> at(point_cloud_pair.D_k),
				this -> dir + "/destination_" + std::to_string(point_cloud_pair.D_k) + "_ba.obj",
				this -> LN_t0.t(), 
				this -> x_t0);
		}

	}

	// connectivity_matrix_res.save(this -> dir + "/connectivity_res.txt",arma::raw_ascii);
	// connectivity_matrix_overlap.save(this -> dir + "/connectivity_overlap.txt",arma::raw_ascii);
	// connectivity_matrix_N_pairs.save(this -> dir + "/connectivity_N_pairs.txt",arma::raw_ascii);


}


int BundleAdjuster::find_overlap_with_pc(int pc_global_index,int start_index,int end_index) const{

	std::vector<PointPair> point_pairs;
	int step_sign = (end_index - start_index )/std::abs(end_index - start_index);
	assert(std::abs(step_sign) == 1);

	for (int i = 0; i < this -> all_registered_pc -> size(); ++i){


		int other_pc_index = start_index + i * step_sign;


		double prop;

		double p = std::log2(this -> all_registered_pc -> at(0) -> size());

		int N_pairs = (int)(std::pow(2, p - this -> h));

		try{

			IterativeClosestPointToPlane::compute_pairs(point_pairs,
				this -> all_registered_pc -> at(pc_global_index),
				this -> all_registered_pc -> at(other_pc_index),
				this -> h);

			prop = double(point_pairs.size()) / N_pairs * 100;
			std::set<int> current_pc_pair = {pc_global_index,other_pc_index};

			std::cout << " ( " << *current_pc_pair.begin() << " , "<< *(--current_pc_pair.end()) << " ) : " << point_pairs.size() << " point pairs , " << prop << " (%) overlap"<< std::endl;
			
			if (prop > 75){
				return other_pc_index;
			}


		}
		catch(ICPNoPairsException & e){

		}

	}

	return -1;




}

