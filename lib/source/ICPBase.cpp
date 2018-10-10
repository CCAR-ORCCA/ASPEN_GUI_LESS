#include <ICPBase.hpp>
#include <FeatureMatching.hpp>
#include <EstimationFeature.hpp>
#include <PointCloudIO.hpp>

#define ICP_DEBUG 1

ICPBase::ICPBase(){

}

ICPBase::ICPBase(
	std::shared_ptr<PC> pc_destination, 
	std::shared_ptr<PC> pc_source){
	
	this -> pc_destination = pc_destination;
	this -> pc_source = pc_source;

}

void ICPBase::register_pc(
	const double r_tol,
	const double s_tol,
	const arma::mat::fixed<3,3> & dcm_0,
	const arma::vec::fixed<3> & X_0){

	this -> r_tol = r_tol;
	this -> s_tol = s_tol;


	this -> register_pc(dcm_0,X_0);

}


arma::vec::fixed<3> ICPBase::get_x() const{
	return this -> x;
}

arma::mat::fixed<3,3> ICPBase::get_dcm() const{
	return RBK::mrp_to_dcm(this -> mrp);
}

arma::mat::fixed<6,6> ICPBase::get_R() const{
	return this -> R;
}

double ICPBase::get_J_res() const{
	return this -> J_res;
}


const std::vector<PointPair > & ICPBase::get_point_pairs() {
	return this -> point_pairs;
}

void ICPBase::set_use_true_pairs(bool use_true_pairs){
	this -> use_true_pairs= use_true_pairs;
}
void ICPBase::set_r_tol(double r_tol){
	this -> r_tol= r_tol;
}
void ICPBase::set_s_tol(double s_tol){
	this ->s_tol = s_tol;
}
void ICPBase::set_iterations_max(unsigned int iterations_max){
	this -> iterations_max = iterations_max;
}


bool ICPBase::get_use_true_pairs() const{
	return this -> use_true_pairs;
}
double ICPBase::get_r_tol() const{
	return this -> r_tol;
}
double ICPBase::get_s_tol() const{
	return this -> s_tol;
}
unsigned int ICPBase::get_iterations_max() const{
	return this -> iterations_max;
}

void ICPBase::set_save_rigid_transform(const arma::vec::fixed<3> & x_save,
	const arma::mat::fixed<3,3> & dcm_save){
	this -> x_save = x_save;
	this -> dcm_save = dcm_save;
}


void ICPBase::set_pc_source(std::shared_ptr<PC> pc_source){
	this -> pc_source = pc_source;
}

void ICPBase::set_pc_destination(std::shared_ptr<PC> pc_destination){
	this -> pc_destination = pc_destination;
}








void ICPBase::register_pc(
	const arma::mat::fixed<3,3> & dcm_0,
	const arma::vec::fixed<3> & X_0){

	double J  = std::numeric_limits<double>::infinity();
	double J_0  = std::numeric_limits<double>::infinity();
	double J_previous = std::numeric_limits<double>::infinity();


	int h = this -> maximum_h;

	bool next_h = false;

	arma::mat::fixed<6,6> info_mat;
	arma::vec::fixed<6> normal_mat;


	// The batch estimator is initialized
	this -> mrp = RBK::dcm_to_mrp(dcm_0);
	this -> x = X_0;

	if (this -> use_pca_prealignment){
		this -> pca_prealignment(this -> mrp ,this -> x);
	}

	if (this -> iterations_max == 0){
		return;
	}

	// If no pairs have been provided, they are recomputed
	if(this -> point_pairs.size() == 0){
		this -> compute_pairs(h,RBK::mrp_to_dcm(this -> mrp),this -> x);
		FeatureMatching<PointNormal>::save_matches(
			"pairs_h_"+std::to_string(h) + "_iter_"+ std::to_string(0) + ".txt",
			this -> point_pairs,
			*this -> pc_source,
			*this -> pc_destination);
		this -> hierarchical = true;
		
	}
	else{
		this -> hierarchical = false;
	}

	J_0 = this -> compute_mean_residuals(RBK::mrp_to_dcm(this -> mrp),this -> x);
	J = J_0;


	#if ICP_DEBUG
	std::cout << "\nStarting ICP with a-priori rigid transform : " << std::endl;
	std::cout << "\tMRP: \n" << this -> mrp.t();
	std::cout << "\tx: \n" << this -> x.t();
	std::cout << "\nInitial residuals : " << std::endl;
	std::cout << J_0 << std::endl;

	#endif

		// The ICP is iterated
	for (unsigned int iter = 0; iter < this -> iterations_max; ++iter) {

		/***************************************
		Going down to the lower hierarchical level
		****************************************/

		if ( next_h == true && this -> hierarchical) {
				// The pairs are formed only after a change in the hierchical search
			this -> compute_pairs(h,RBK::mrp_to_dcm(this -> mrp),this -> x);
			next_h = false;

				#if ICP_DEBUG
			FeatureMatching<PointNormal>::save_matches(
				"pairs_h_"+std::to_string(h) + "_iter_"+ std::to_string(iter) + ".txt",
				this -> point_pairs,
				*this -> pc_source,
				*this -> pc_destination);
				#endif
		}
		/**************************************
		***************************************
		**************************************/


		#if ICP_DEBUG
		std::cout << "ICP iteration " << iter + 1 << " / " << this -> iterations_max  << std::endl;
		if (this -> hierarchical)
			std::cout << "Hierchical level : " << std::to_string(h) << std::endl;
		std::cout << "Pairs: " << this -> point_pairs.size() << std::endl << std::endl;
		#endif


		/**************************************
		Minimizing the chosen ICP cost function
		***************************************/

			// The matrices of the LS problem are now accumulated
		info_mat.fill(0);
		normal_mat.fill(0);

			// #pragma omp parallel for reduction(+:info_mat), reduction(+:normal_mat) if (USE_OMP_ICP)
		for (unsigned int pair_index = 0; pair_index < this -> point_pairs.size(); ++pair_index) {
			arma::mat::fixed<6,6> info_mat_temp;
			arma::vec::fixed<6> normal_mat_temp;
			this -> build_matrices(pair_index, this -> mrp,this -> x,info_mat_temp,normal_mat_temp,1);
			info_mat += info_mat_temp;
			normal_mat += normal_mat_temp;


			

		}


			#if ICP_DEBUG
		std::cout << "\nInfo mat: " << std::endl;
		std::cout << info_mat << std::endl;
		std::cout << "\nNormal mat: " << std::endl;
		std::cout << normal_mat << std::endl;
			#endif

			// The state deviation [dmrp,dx] is solved for
		arma::vec dX = arma::solve(info_mat, normal_mat);
		arma::vec dx = dX.subvec(0,2);
		arma::vec dmrp = dX.subvec(3,5);

			// The state is updated
		this -> mrp = RBK::dcm_to_mrp(RBK::mrp_to_dcm(dmrp) * RBK::mrp_to_dcm(this -> mrp));
		this -> x +=  dx;

			// the mrp is switched to its shadow if need be
		this -> mrp = RBK::shadow_mrp(this -> mrp);

			// The postfit residuals are computed
		J = this -> compute_mean_residuals(RBK::mrp_to_dcm(this -> mrp),this -> x);

			#if ICP_DEBUG
		std::cout << "\nDeviation : " << std::endl;
		std::cout << dX << std::endl;
		std::cout << "\nResiduals: " << J << std::endl;
		std::cout << "MRP: \n" << this -> mrp << std::endl;
		std::cout << "x: \n" << this -> x << std::endl;
		std::cout << "Covariance :\n" << std::endl;
		std::cout << arma::inv(info_mat) << std::endl;
			#endif


		/************************
		************************
		************************/

		if(this -> check_convergence(iter,J,J_0,J_previous,h,next_h)){
			break;
		}

	}
	

	#if ICP_DEBUG
	std::cout << "Leaving ICPBase. Best transform: \n";
	std::cout << "\tmrp: " << this -> mrp.t();
	std::cout << "\tx: " << this -> x.t();
	std::cout << "\tResiduals: " << J << std::endl;
	FeatureMatching<PointNormal>::save_matches(
		"icp_pairs.txt",
		this -> point_pairs,
		*this -> pc_source,
		*this -> pc_destination,
		RBK::mrp_to_dcm(this -> mrp),
		this -> x);

	#endif



	try{
		this -> R = arma::inv(info_mat);
	}
	catch(std::runtime_error & e){
		std::cout << e.what();
	}
	this -> J_res = J ;

}

bool ICPBase::check_convergence(const int & iter,const double & J,const double & J_0, double & J_previous,int & h,bool & next_h){

	// Has converged
	if ( J / J_0 <= this -> r_tol || J == 0 ) {
		#if ICP_DEBUG
		std::cout << "Has converged\n";
		#endif
		return true;
	}
	// Has stalled
	else if ( (std::abs(J - J_previous) / J <= this -> s_tol && h > this -> minimum_h) && iter + 1 < this -> iterations_max) {
		h = h - 1;
		next_h = true;

		if (this -> hierarchical){
			J_previous = std::numeric_limits<double>::infinity();
		#if ICP_DEBUG
		std::cout << "Has stalled, going to the next level\n";
		#endif
		}
		else{
			return true;
		}

	}
	// Has not converged after all iterations have been used
	else if ((iter + 1 == this -> iterations_max || std::abs(J - J_previous) / J <= this -> s_tol ) && h == this -> minimum_h  ) {
		PointCloudIO<PointNormal>::save_to_obj(*this -> pc_source,
			"max_iter.obj",
			this -> dcm_save * RBK::mrp_to_dcm(this -> mrp),
			this -> dcm_save * this -> x + this -> x_save);
		#if ICP_DEBUG
		std::cout << "Has not converged\n";
		#endif
		return true;
	}
	// Has neither converged, stalled or used all iterations
	// The postfit residuals become the prefit residuals of the next iteration
	else {
		J_previous = J;
		#if ICP_DEBUG
		std::cout << "Next iteration\n";
		#endif
	}
	return false;

}


void ICPBase::set_pairs(const std::vector<PointPair> & point_pairs){
	this -> point_pairs = point_pairs;
}


// void ICPBase::register_pc_RANSAC(double fraction_inliers_used,
// 	double fraction_inliers_requested,
// 	unsigned int iter_ransac_max,
// 	const arma::mat::fixed<3,3> &  dcm_0,
// 	const arma::vec::fixed<3> &  X_0 ){

// 	double J_best_RANSAC = std::numeric_limits<double>::infinity();

// 	arma::mat::fixed<3,3> dcm_best_RANSAC;
// 	arma::vec::fixed<3>	x_best_RANSAC ;
// 	std::vector<PointPair> best_pairs_RANSAC,pairs_RANSAC;
// 	arma::vec best_pairs_RANSAC_weights;
// 	arma::mat::fixed<6,6> info_mat;
// 	arma::vec::fixed<6> normal_mat;



// 	if (this -> iterations_max == 0 || iter_ransac_max == 0){
// 		return;
// 	}

// 	#if ICP_DEBUG
// 	auto start = std::chrono::system_clock::now();

// 	std::cout << "Computing pc_destination descriptors...\n";
// 	#endif

// 	if (this -> use_FPFH){
// 		this -> pc_destination -> compute_feature_descriptors(PointDescriptor::Type::FPFHDescriptor,this -> keep_correlations,
// 			this -> N_bins,this -> neighborhood_radius,"pc_destination");
// 	}
// 	else{
// 		this -> pc_destination -> compute_feature_descriptors(PointDescriptor::Type::PFHDescriptor,this -> keep_correlations,
// 			this -> N_bins,this -> neighborhood_radius,"pc_destination");
// 	}

// 	#if ICP_DEBUG
// 	std::cout << "Computing pc_source descriptors...\n";
// 	#endif

// 	if (this -> use_FPFH){
// 		this -> pc_source -> compute_feature_descriptors(PointDescriptor::Type::FPFHDescriptor,this -> keep_correlations,
// 			this -> N_bins,this -> neighborhood_radius,"pc_source");
// 	}
// 	else{
// 		this -> pc_source -> compute_feature_descriptors(PointDescriptor::Type::PFHDescriptor,this -> keep_correlations,
// 			this -> N_bins,this -> neighborhood_radius,"pc_source");
// 	}

// 	#if ICP_DEBUG
// 	auto end = std::chrono::system_clock::now();
// 	std::chrono::duration<double> elapsed_seconds = end-start;
// 	std::cout << "Time elapsed computing features: " << elapsed_seconds.count()<< " (s)"<< std::endl;
// 	this -> pc_source -> save_point_descriptors("source_descriptors.txt");
// 	this -> pc_destination -> save_point_descriptors("destination_descriptors.txt");
// 	std::cout << "Matching descriptors...\n";
// 	start = std::chrono::system_clock::now();
// 	#endif

// 	auto all_matches = PC::find_pch_matches_kdtree(this -> pc_source,this -> pc_destination);

// 	int N_tentative_source_points = (int)(all_matches.size() * fraction_inliers_used);

// 	#if ICP_DEBUG
// 	end = std::chrono::system_clock::now();
// 	elapsed_seconds = end-start;
// 	std::cout << "Time elapsed matching features: " << elapsed_seconds.count()<< " (s)"<< std::endl;
// 	std::cout << "Total number of matches: "+ std::to_string(all_matches.size()) + " \n";
// 	ICPBase::save_pairs(all_matches,"all_pairs.txt",this -> pc_source,this -> pc_destination);
// 	std::cout << "Weighing the " << all_matches.size() << " pairs...\n";
// 	#endif

// 	// arma::vec weights = this -> weigh_ransac_pairs(all_matches,this -> neighborhood_radius);
// 	arma::vec weights = arma::ones<arma::vec>(all_matches.size());

// 	arma::uvec indices = arma::regspace<arma::uvec>(0,all_matches.size() - 1);

// 	#if ICP_DEBUG
// 	std::cout << "Consensus-based weights of the feature pairs:\n";
// 	std::cout  <<  std::endl << weights << std::endl;
// 	weights.save("all_pairs_weights.txt",arma::raw_ascii);
// 	#endif

// 	for (unsigned int iter_ransac = 0; iter_ransac < iter_ransac_max; ++iter_ransac){

// 		#if ICP_DEBUG
// 		std::cout <<  "RANSAC iteration " << iter_ransac + 1 << " / " << iter_ransac_max  << std::endl;
// 		#endif

// 		double J  = std::numeric_limits<double>::infinity();
// 		double J_0  = std::numeric_limits<double>::infinity();

// 		// The batch estimator is initialized
// 		arma::vec::fixed<3> mrp = RBK::dcm_to_mrp(dcm_0);
// 		arma::vec::fixed<3> x_temp = X_0;

// 		// Drawing random pairs from the matches
// 		this -> point_pairs.clear();
// 		pairs_RANSAC.clear();
// 		indices = arma::shuffle(indices);
// 		arma::vec active_weights = weights(indices.subvec(0,N_tentative_source_points - 1));

// 		#if ICP_DEBUG
// 		std::cout << "Active consensus-based weights of the feature pairs:\n";
// 		std::cout  <<  std::endl << active_weights << std::endl;
// 		#endif

// 		for (int k = 0; k < N_tentative_source_points; ++k){
// 			this -> point_pairs.push_back(all_matches[indices[k]]);
// 			pairs_RANSAC.push_back(all_matches[indices[k]]);
// 		}

// 		// The ICP is iterated
// 		for (unsigned int iter = 0; iter < this -> iterations_max; ++iter) {

// 		#if ICP_DEBUG
// 			std::cout << "\tICP iteration " << iter + 1 << " / " << this -> iterations_max  << std::endl;
// 		#endif

// 			if (iter == 0 ) {
// 				// The initial residuals are computed
// 				J_0 = this -> compute_rms_residuals(RBK::mrp_to_dcm(mrp),x_temp);
// 				J = J_0;
// 			}


// 			// The matrices of the LS problem are now accumulated
// 			info_mat.fill(0);
// 			normal_mat.fill(0);


// 			// #pragma omp parallel for reduction(+:info_mat), reduction(+:normal_mat) if (USE_OMP_ICP)
// 			for (unsigned int pair_index = 0; pair_index < this -> point_pairs.size(); ++pair_index) {

// 				arma::mat::fixed<6,6> info_mat_temp;
// 				arma::vec::fixed<6> normal_mat_temp;

// 				#if ICP_DEBUG
// 				std::cout << "Building matrix " << pair_index + 1 << " / " << this -> point_pairs.size() << std::endl;
// 				#endif

// 				this -> build_matrices(pair_index, mrp,x_temp,info_mat_temp,normal_mat_temp,active_weights(pair_index));

// 				info_mat += info_mat_temp;
// 				normal_mat += normal_mat_temp;

// 			}


// 			#if ICP_DEBUG
// 			std::cout << "\nInfo mat: " << std::endl;
// 			std::cout << info_mat << std::endl;
// 			std::cout << "\nNormal mat: " << std::endl;
// 			std::cout << normal_mat << std::endl;
// 			#endif

// 			// The state deviation [dmrp,dx] is solved for
// 			arma::vec dX = arma::solve(info_mat, normal_mat);
// 			arma::vec dx = dX.subvec(0,2);
// 			arma::vec dmrp = dX.subvec(3,5);



// 			// The state is updated
// 			mrp = RBK::dcm_to_mrp(RBK::mrp_to_dcm(dmrp) * RBK::mrp_to_dcm(mrp));

// 			x_temp = x_temp + dx;

// 			// the mrp is switched to its shadow if need be
// 			if (arma::norm(mrp) > 1) {
// 				mrp = - mrp / ( pow(arma::norm(mrp), 2));
// 			}

// 			// The postfit residuals are computed
// 			J = this -> compute_rms_residuals(RBK::mrp_to_dcm(mrp),x_temp);

// 		#if ICP_DEBUG
// 			std::cout << "\nDeviation : " << std::endl;
// 			std::cout << dX << std::endl;
// 			std::cout << "\nResiduals: " << J << std::endl;
// 			std::cout << "MRP: \n" << mrp << std::endl;
// 			std::cout << "x: \n" << x_temp << std::endl;
// 		#endif

// 		}

// 		// End of the ICP Loop, the surrogate model parameters have been fitted to the randomly sampled pair.
// 		// Let's see how well this model explains the rest of the data
// 		#if ICP_DEBUG
// 		std::cout << "Investigating surrogate model quality...\n";
// 		#endif
// 		int good_inlier_not_used_count = 0;

// 		for (int k = N_tentative_source_points; k < all_matches.size();  ++k){	

// 			auto point_pair = all_matches[indices[k]];
// 			double distance_to_potential_inlier = this -> compute_distance(point_pair, RBK::mrp_to_dcm(mrp),x_temp);



// 			#if ICP_DEBUG
// 			std::cout << "\tPair weight: " << weights[k] <<  std::endl;
// 			#endif


// 			if (distance_to_potential_inlier < this -> neighborhood_radius){
// 				this -> point_pairs.push_back(point_pair);
// 				++good_inlier_not_used_count;

// 			#if ICP_DEBUG
// 				std::cout << "\t\tFound inlier pair , distance =  " << distance_to_potential_inlier << " , needed " << this -> neighborhood_radius << std::endl;
// 			#endif


// 			}
// 			#if ICP_DEBUG
// 			else{
// 				std::cout << "\t\tOutlier pair , distance =  " << distance_to_potential_inlier << " , needed " << this -> neighborhood_radius << std::endl;
// 			}
// 			#endif




// 		}

// 		double fraction_inliers_found = ((double)(good_inlier_not_used_count + N_tentative_source_points) / all_matches.size());

// 		#if ICP_DEBUG
// 		std::cout << "Model has found " << 100 * fraction_inliers_found << " (%) of inliers total (need " << fraction_inliers_requested * 100 <<  " (%) to validate) \n";
// 		#endif

// 		// If good_inlier_not_used_count is greater than what is prescribed, we have found a good model
// 		if (fraction_inliers_found > fraction_inliers_requested){

// 			arma::mat::fixed<3,3> better_dcm = RBK::mrp_to_dcm(mrp);
// 			arma::vec::fixed<3> better_x = x_temp;


// 			double J_better = this -> compute_rms_residuals(pairs_RANSAC,better_dcm,better_x,active_weights);

// 			// If the good model we found surpasses the previous one, we keep it
// 			if (J_better < J_best_RANSAC){

// 				J_best_RANSAC = J_better;
// 				dcm_best_RANSAC = better_dcm;
// 				x_best_RANSAC = better_x;
// 				best_pairs_RANSAC = pairs_RANSAC;
// 				best_pairs_RANSAC_weights.clear();
// 				best_pairs_RANSAC_weights = active_weights; 

// 				#if ICP_DEBUG
// 				std::cout << "Found better model with J = " << J_best_RANSAC << " explaining " << this -> point_pairs.size() << " feature pairs using "+ std::to_string(N_tentative_source_points) +  " data points\n";
// 				#endif
// 			}
// 		}
// 	}

// 	#if ICP_DEBUG
// 	std::cout << "Leaving RANSAC. Best transform: \n";
// 	std::cout << "\tmrp: " << RBK::dcm_to_mrp(dcm_best_RANSAC).t();
// 	std::cout << "\tx: " << x_best_RANSAC.t();
// 	std::cout << "\tResiduals: " << J_best_RANSAC << std::endl;
// 	std::cout << "\tUsing a total of " << best_pairs_RANSAC.size() << " pairs\n";
// 	ICPBase::save_pairs(best_pairs_RANSAC,"ransac_pairs_aligned.txt",this -> pc_source,this -> pc_destination,dcm_best_RANSAC,x_best_RANSAC);
// 	ICPBase::save_pairs(best_pairs_RANSAC,"ransac_pairs.txt",this -> pc_source,this -> pc_destination);
// 	best_pairs_RANSAC_weights.save("best_pairs_RANSAC_weights.txt",arma::raw_ascii);
// 	#endif

// 	this -> x = x_best_RANSAC;
// 	this -> mrp = RBK::dcm_to_mrp(dcm_best_RANSAC);
// 	this -> J_res = J_best_RANSAC;

// }


// void ICPBase::register_pc_bf(unsigned int iter_bf_max,
// 	int N_possible_matches,int N_tentative_source_points,
// 	const arma::mat::fixed<3,3>  & dcm_0,
// 	const arma::vec::fixed<3> &  X_0 ){

// 	double J_best = std::numeric_limits<double>::infinity();
// 	arma::mat::fixed<3,3> dcm_best;
// 	arma::vec::fixed<3>	x_best;
// 	std::vector<PointPair> best_pairs;
// 	arma::mat::fixed<6,6> info_mat;
// 	arma::vec::fixed<6> normal_mat;



// 	if (this -> iterations_max == 0|| iter_bf_max == 0){
// 		return;
// 	}

// 	#if ICP_DEBUG
// 	auto start = std::chrono::system_clock::now();

// 	std::cout << "Computing pc_destination descriptors...\n";
// 	#endif

// 	if (this -> use_FPFH){
// 		this -> pc_destination -> compute_feature_descriptors(PointDescriptor::Type::FPFHDescriptor,this -> keep_correlations,
// 			this -> N_bins,this -> neighborhood_radius,"pc_destination");
// 	}
// 	else{
// 		this -> pc_destination -> compute_feature_descriptors(PointDescriptor::Type::PFHDescriptor,this -> keep_correlations,
// 			this -> N_bins,this -> neighborhood_radius,"pc_destination");
// 	}

// 	#if ICP_DEBUG
// 	std::cout << "Computing pc_source descriptors...\n";
// 	#endif

// 	if (this -> use_FPFH){
// 		this -> pc_source -> compute_feature_descriptors(PointDescriptor::Type::FPFHDescriptor,this -> keep_correlations,
// 			this -> N_bins,this -> neighborhood_radius,"pc_source");
// 	}
// 	else{
// 		this -> pc_source -> compute_feature_descriptors(PointDescriptor::Type::PFHDescriptor,this -> keep_correlations,
// 			this -> N_bins,this -> neighborhood_radius,"pc_source");
// 	}

// 	#if ICP_DEBUG
// 	auto end = std::chrono::system_clock::now();
// 	std::chrono::duration<double> elapsed_seconds = end-start;
// 	std::cout << "Time elapsed computing features: " << elapsed_seconds.count()<< " (s)"<< std::endl;
// 	this -> pc_source -> save_point_descriptors("source_descriptors.txt");
// 	this -> pc_destination -> save_point_descriptors("destination_descriptors.txt");
// 	std::cout << "Matching descriptors...\n";
// 	start = std::chrono::system_clock::now();
// 	#endif

// 	std::vector< int > active_source_points;
// 	std::map<int , std::vector<int > > possible_matches;

// 	PC::find_N_closest_pch_matches_kdtree(this -> pc_source,this -> pc_destination,N_possible_matches,
// 		active_source_points,possible_matches);

// 	#if ICP_DEBUG
// 	end = std::chrono::system_clock::now();
// 	elapsed_seconds = end-start;
// 	std::cout << "Time elapsed matching features: " << elapsed_seconds.count()<< " (s)"<< std::endl;
// 	#endif


// 	for (unsigned int iter_ransac = 0; iter_ransac < iter_bf_max; ++iter_ransac){

// 		#if ICP_DEBUG
// 		std::cout <<  "Brute-force iteration " << iter_ransac + 1 << " / " << iter_bf_max  << std::endl;
// 		#endif

// 		double J = std::numeric_limits<double>::infinity();
// 		double J_0 = std::numeric_limits<double>::infinity();

// 		// The batch estimator is initialized
// 		arma::vec::fixed<3> mrp = RBK::dcm_to_mrp(dcm_0);
// 		arma::vec::fixed<3> x_temp = X_0;

// 		// Drawing random pairs from the matches
// 		this -> point_pairs.clear();

// 		std::vector<int > tentative_source_points;

// 		// The following draws N_sample source points sufficiently separated from 
// 		// one another

// 		arma::ivec random_indices = arma::shuffle(arma::regspace<arma::ivec>(0,active_source_points.size() - 1));

// 		#if ICP_DEBUG
// 		std::cout << "\tDrawing " << N_tentative_source_points << " source points from the "  << active_source_points.size() <<  " active source points\n";
// 		#endif

// 		for (int i = 0; i < random_indices.size(); ++i){

// 			auto p_source = active_source_points[random_indices(i)];

// 			if(tentative_source_points.size() == 0){
// 				tentative_source_points.push_back(p_source);
// 			}
// 			else{
// 				bool insert = true;

// 				for (int k = 0; k < tentative_source_points.size(); ++k){
// 					if (arma::norm( this -> pc_source -> get_point_coordinates(p_source) - this -> pc_source -> get_point_coordinates(tentative_source_points.at(k))) < 3 * this -> neighborhood_radius){

// 						insert = false;

// 						break;
// 					}
// 				}

// 				if (insert){
// 					tentative_source_points.push_back(p_source);
// 				}
// 			}

// 			if (tentative_source_points.size() == N_tentative_source_points){
// 				break;
// 			}
// 		}

// 			#if ICP_DEBUG
// 		std::cout << "\tFound " << tentative_source_points.size() << " tentative source points\n";
// 			#endif


// 		// These points are then randomly matched with a destination points amongst 
// 		// those they are the closest to
// 		for (int k =0; k < tentative_source_points.size(); ++k){

// 			arma::ivec random_index = arma::randi<arma::ivec>(1,arma::distr_param(0,N_possible_matches -1));
// 			auto p_source = tentative_source_points[k];
// 			auto p_destination = possible_matches[p_source][random_index(0)];

// 			PointPair formed_pair = std::make_pair(k,p_destination);
// 			this -> point_pairs.push_back(formed_pair);
// 		}



// 		// The ICP is iterated
// 		for (unsigned int iter = 0; iter < this -> iterations_max; ++iter) {

// 		#if ICP_DEBUG
// 			std::cout << "\tICP iteration " << iter + 1 << " / " << this -> iterations_max  << std::endl;
// 		#endif

// 			if (iter == 0 ) {
// 				// The initial residuals are computed
// 				J_0 = this -> compute_rms_residuals(RBK::mrp_to_dcm(mrp),x_temp);
// 				J = J_0;
// 			}


// 			// The matrices of the LS problem are now accumulated
// 			info_mat.fill(0);
// 			normal_mat.fill(0);


// 			// #pragma omp parallel for reduction(+:info_mat), reduction(+:normal_mat) if (USE_OMP_ICP)
// 			for (unsigned int pair_index = 0; pair_index < this -> point_pairs.size(); ++pair_index) {

// 				arma::mat::fixed<6,6> info_mat_temp;
// 				arma::vec::fixed<6> normal_mat_temp;



// 				this -> build_matrices(pair_index, mrp,x_temp,info_mat_temp,normal_mat_temp,1.);

// 				info_mat += info_mat_temp;
// 				normal_mat += normal_mat_temp;

// 			}


// 			#if ICP_DEBUG
// 			std::cout << "\nInfo mat: " << std::endl;
// 			std::cout << info_mat << std::endl;
// 			std::cout << "\nNormal mat: " << std::endl;
// 			std::cout << normal_mat << std::endl;
// 			#endif

// 			// The state deviation [dmrp,dx] is solved for
// 			arma::vec dX = arma::solve(info_mat, normal_mat);
// 			arma::vec dx = dX.subvec(0,2);
// 			arma::vec dmrp = dX.subvec(3,5);

// 			// The state is updated
// 			mrp = RBK::dcm_to_mrp(RBK::mrp_to_dcm(dmrp) * RBK::mrp_to_dcm(mrp));

// 			x_temp = x_temp + dx;

// 			// the mrp is switched to its shadow if need be
// 			if (arma::norm(mrp) > 1) {
// 				mrp = - mrp / ( pow(arma::norm(mrp), 2));
// 			}

// 			// The postfit residuals are computed
// 			J = this -> compute_rms_residuals(RBK::mrp_to_dcm(mrp),x_temp);

// 		#if ICP_DEBUG
// 			std::cout << "\nDeviation : " << std::endl;
// 			std::cout << dX << std::endl;
// 			std::cout << "\nResiduals: " << J << std::endl;
// 			std::cout << "MRP: \n" << mrp << std::endl;
// 			std::cout << "x: \n" << x_temp << std::endl;
// 		#endif

// 		}


// 		arma::vec epsilon = this -> compute_y_vector(this -> point_pairs,RBK::mrp_to_dcm(mrp),x_temp);
// 		double J_better = this -> compute_Huber_loss(epsilon,10 * this -> neighborhood_radius);

// 		// If the good model we found surpasses the previous one, we keep it
// 		if (J_better < J_best){
// 			J_best = J_better;
// 			dcm_best = RBK::mrp_to_dcm(mrp);
// 			x_best = x_temp;
// 			best_pairs = this -> point_pairs;
// 				#if ICP_DEBUG
// 			std::cout << "Found better model with J = " << J_best << std::endl;
// 				#endif
// 		}
// 	}

// 	#if ICP_DEBUG
// 	std::cout << "Leaving Simplified RANSAC. Best transform: \n";
// 	std::cout << "\tmrp: " << RBK::dcm_to_mrp(dcm_best).t();
// 	std::cout << "\tx: " << x_best.t();
// 	std::cout << "\tResiduals: " << J_best << std::endl;
// 	std::cout << "\tUsing a total of " << best_pairs.size() << " pairs\n";
// 	ICPBase::save_pairs(best_pairs,"ransac_pairs_aligned.txt",this -> pc_source,this -> pc_destination,dcm_best,x_best);
// 	ICPBase::save_pairs(best_pairs,"ransac_pairs.txt",this -> pc_source,this -> pc_destination);
// 	#endif

// 	this -> x = x_best;
// 	this -> mrp = RBK::dcm_to_mrp(dcm_best);
// 	this -> J_res = J_best;

// }












// void ICPBase::save_pairs(const std::vector<PointPair> & pairs,
// 	std::string path,
// 	std::shared_ptr<PC> pc_source,
// 	std::shared_ptr<PC> pc_destination,
// 	const arma::mat::fixed<3,3> & dcm,
// 	const arma::vec::fixed<3> & x){

// 	arma::mat pairs_m(pairs.size(),7);

// 	for (int i = 0; i < pairs.size(); ++i){

// 		pairs_m.submat(i,0,i,2) = (dcm * pc_source -> get_point_coordinates( pairs[i].first)  + x).t();
// 		pairs_m.submat(i,3,i,5) = pc_destination -> get_point_coordinates( pairs[i].second) .t();
// 		pairs_m(i,6) = -1;

// 	}

// 	pairs_m.save(path,arma::raw_ascii);
// }

void ICPBase::pca_prealignment(arma::vec::fixed<3> & mrp,arma::vec::fixed<3> & x) const{

	arma::vec::fixed<3> center_destination = EstimationFeature<PointNormal,PointNormal>::compute_center(*this -> pc_destination);
	arma::vec::fixed<3> center_source = EstimationFeature<PointNormal,PointNormal>::compute_center(*this -> pc_source);

	arma::mat::fixed<3,3> E_source = EstimationFeature<PointNormal,PointNormal>::compute_principal_axes(*this -> pc_destination ,center_destination);
	arma::mat::fixed<3,3> E_destination = EstimationFeature<PointNormal,PointNormal>::compute_principal_axes(*this -> pc_source ,center_source);


	// The center of the source point cloud is aligned with that of the destination point cloud
	x = (center_destination- E_destination * E_source.t() * center_source);
	mrp = RBK::dcm_to_mrp(E_destination * E_source.t());

}	



bool ICPBase::get_use_pca_prealignment() const{
	return this -> use_pca_prealignment;
}
void ICPBase::set_use_pca_prealignment(bool use_pca_prealignment){
	this -> use_pca_prealignment = use_pca_prealignment;
}





void ICPBase::set_minimum_h(unsigned int min_h){
	this -> minimum_h = min_h;
}
unsigned int ICPBase::get_minimum_h() const{
	return this -> minimum_h;
}


void ICPBase::set_maximum_h(unsigned int  max_h){
	this -> maximum_h = max_h;
}
unsigned int ICPBase::get_maximum_h() const{
	return this -> maximum_h;
}

// double ICPBase::get_neighborhood_radius() const{
// 	return this -> neighborhood_radius;
// }

// void ICPBase::set_neighborhood_radius(double neighborhood_radius){
// 	this -> neighborhood_radius = neighborhood_radius;
// }

// arma::vec ICPBase::weigh_ransac_pairs(const std::vector<PointPair> & matched_pairs,double radius){

// 	arma::vec weights(matched_pairs.size());

// 	for (int i = 0; i < matched_pairs.size(); ++i){

// 		double likelihood = this -> compute_point_weight(this -> pc_source,matched_pairs[i].first,
// 			this -> pc_destination,3 * radius)
// 		* this -> compute_point_weight(this -> pc_destination, matched_pairs[i].second,
// 			this -> pc_source,3 * radius);

// 		if (likelihood > 0){
// 			// weights(i) = std::max(std::log(likelihood),0.);
// 			weights(i) = likelihood;

// 		}
// 		else{
// 			weights(i) = 0;
// 		}

// 		std::cout << "Weight for this pair: " << weights(i) << std::endl;

// 	}




// 	return weights;

// }

// double ICPBase::compute_point_weight(const std::shared_ptr<PC> & origin_pc, const int & origin_point,
// 	const std::shared_ptr<PC> & target_pc,
// 	const double & radius) const{



// 	const PointNormal p = origin_pc -> get_point(origin_point);

// 	auto origin_point_neighborhood = origin_pc -> get_nearest_neighbors_radius(p.get_point_coordinates(), radius);
// 	arma::vec::fixed<2> mean_angles = {0,0};
// 	arma::mat::fixed<2,2> covariance = arma::zeros<arma::mat>(2,2);
// 	std::vector<arma::vec> angles_distribution;

// 	int self_match = p.get_match() ;
// 	arma::vec self_pair_direction = arma::normalise(target_pc -> get_point_coordinates(self_match) - p.get_point_coordinates() );

// 	double self_alpha = std::atan2(self_pair_direction(1),self_pair_direction(0));
// 	double self_beta = std::atan2(self_pair_direction(2),arma::norm(self_pair_direction.subvec(0,1)));

// 	arma::vec self_angles = {self_alpha,self_beta};

// 	for (int j = 0; j < origin_point_neighborhood.size(); ++j){

// 		int match = origin_pc -> get_point(origin_point_neighborhood[j]).get_match();

// 		if (match < 0){
// 			arma::vec pair_direction = arma::normalise(target_pc -> get_point_coordinates(match) - origin_pc -> get_point_coordinates(origin_point_neighborhood[j] ) );

// 			double alpha = std::atan2(pair_direction(1),pair_direction(0));
// 			double beta = std::atan2(pair_direction(2),arma::norm(pair_direction.subvec(0,1)));

// 			arma::vec angles = {alpha,beta};
// 			mean_angles += angles;
// 			angles_distribution.push_back(angles);
// 		}

// 	}


// 	if (angles_distribution.size() < 4){

// 		#if ICP_DEBUG
// 		std::cout << "\tToo few active features found in neighborhood. Setting w = 0\n";
// 		#endif


// 		return 0;
// 	}
// 	else{
// 			// If there are more than 1 point in the neighborhood then a distribution of directions can be 
// 			// computed

// 		mean_angles = mean_angles / angles_distribution.size();

// 		for (int k = 0; k < angles_distribution.size(); ++k){
// 			covariance += 1./(angles_distribution.size() - 1) * (angles_distribution[k] - mean_angles) * (angles_distribution[k] - mean_angles).t();
// 		}
// 		double w = 1./std::sqrt(2 * arma::datum::pi * std::abs(arma::det(covariance))) * std::exp(
// 			-0.5 * arma::dot(self_angles - mean_angles,arma::inv(covariance)*(self_angles - mean_angles)));

// 		#if ICP_DEBUG
// 		std::cout << "\t" << angles_distribution.size() << " sets of angles were considered. w = " + std::to_string(w) + "\n";
// 		#endif


// 		return w;

// 	}




// }





double ICPBase::compute_rms_residuals(
	const std::vector<PointPair> & point_pairs,
	const arma::mat::fixed<3,3> & dcm_S ,
	const arma::vec::fixed<3> & x_S ,
	const arma::vec & weights,
	const arma::mat::fixed<3,3> & dcm_D ,
	const arma::vec::fixed<3> & x_D )  const{

	double J = 0;

	double mean = this -> compute_mean_residuals(point_pairs,dcm_S ,x_S ,weights ,dcm_D , x_D );

	if (weights.size() == 0){
	#pragma omp parallel for reduction(+:J) if (USE_OMP_ICP)
		for (unsigned int pair_index = 0; pair_index <point_pairs.size(); ++pair_index) {
			J += std::pow(this -> compute_distance(point_pairs[pair_index],  dcm_S,x_S,dcm_D,x_D) - mean,2);
		}
	}
	else{
		#pragma omp parallel for reduction(+:J) if (USE_OMP_ICP)
		for (unsigned int pair_index = 0; pair_index <point_pairs.size(); ++pair_index) {
			J += weights(pair_index) * std::pow(this -> compute_distance(point_pairs[pair_index],  dcm_S,x_S,dcm_D,x_D) - mean,2);
		}
	}
	return std::sqrt(J / (point_pairs.size()-1) );

}

double ICPBase::compute_mean_residuals(
	const std::vector<PointPair> & point_pairs,
	const arma::mat::fixed<3,3> & dcm_S ,
	const arma::vec::fixed<3> & x_S ,
	const arma::vec & weights,
	const arma::mat::fixed<3,3> & dcm_D ,
	const arma::vec::fixed<3> & x_D ) const{

	double J = 0;


	if (weights.size() == 0){
	#pragma omp parallel for reduction(+:J) if (USE_OMP_ICP)
		for (unsigned int pair_index = 0; pair_index <point_pairs.size(); ++pair_index) {

			J += this -> compute_distance(point_pairs[pair_index],  dcm_S,x_S,dcm_D,x_D)/ point_pairs.size();
		}
	}
	else{
		#pragma omp parallel for reduction(+:J) if (USE_OMP_ICP)
		for (unsigned int pair_index = 0; pair_index <point_pairs.size(); ++pair_index) {

			J += weights(pair_index) * this -> compute_distance(point_pairs[pair_index],  dcm_S,x_S,dcm_D,x_D)/ point_pairs.size();

		}
	}

	return J;
}



arma::vec ICPBase::compute_y_vector(const std::vector<PointPair> & point_pairs,
	const arma::mat::fixed<3,3> & dcm_S ,
	const arma::vec::fixed<3> & x_S) const {

	arma::vec y(point_pairs.size());

	for (unsigned int pair_index = 0; pair_index <point_pairs.size(); ++pair_index) {
		y(pair_index) = this -> compute_distance(point_pairs[pair_index],  dcm_S,x_S);
	}

	return y;

}


double ICPBase::compute_rms_residuals(
	const arma::mat::fixed<3,3> & dcm,
	const arma::vec::fixed<3> & x,
	const arma::vec & weights) {

	return this -> compute_rms_residuals(this -> point_pairs,dcm,x,weights);

}


double ICPBase::compute_mean_residuals(
	const arma::mat::fixed<3,3> & dcm,
	const arma::vec::fixed<3> & x,
	const arma::vec & weights) {
	return this -> compute_mean_residuals(this -> point_pairs,dcm,x,weights);

}


// double ICPBase::compute_Huber_loss(const arma::vec & y, double threshold){

// 	double hl = 0;

// 	for (int i = 0; i < y.size(); ++i){

// 		if (std::abs(y(i)) < threshold){
// 			hl += 0.5 * std::pow(y(i),2);

// 		}
// 		else{
// 			hl += threshold * (2 * std::abs(y(i))  - threshold);
// 		}

// 	}
// 	return std::sqrt(hl / y.size());

// }










