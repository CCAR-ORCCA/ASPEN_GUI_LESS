#include "ICPBase.hpp"

#define ICP_DEBUG 1

ICPBase::ICPBase(std::shared_ptr<PC> pc_destination, 
	std::shared_ptr<PC> pc_source){
	
	this -> pc_destination = pc_destination;
	this -> pc_source = pc_source;

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


std::vector<PointPair > * ICPBase::get_point_pairs() {
	return &this -> point_pairs;
}

void ICPBase::set_use_true_pairs(bool use_true_pairs){
	this -> use_true_pairs= use_true_pairs;
}
void ICPBase::set_rel_tol(double rel_tol){
	this -> rel_tol= rel_tol;
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
double ICPBase::get_rel_tol() const{
	return this -> rel_tol;
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





void ICPBase::register_pc(arma::mat::fixed<3,3> dcm_0,arma::vec::fixed<3> X_0){

	double J  = std::numeric_limits<double>::infinity();
	double J_0  = std::numeric_limits<double>::infinity();
	double J_previous = std::numeric_limits<double>::infinity();

	
	int h = this -> maximum_h;

	bool exit = false;
	bool next_h = true;

	arma::mat::fixed<6,6> info_mat;
	arma::vec::fixed<6> normal_mat;

	
	// The batch estimator is initialized
	this -> mrp = RBK::dcm_to_mrp(dcm_0);
	this -> x = X_0;

	if (this -> use_pca_prealignment){
		this -> pca_prealignment(this -> mrp ,this -> x);
	}
	
	if (this -> iterations_max == 0){
		exit = true;
	}

	#if ICP_DEBUG
	std::cout << "\nStarting ICP with a-priori rigid transform : " << std::endl;
	std::cout << "\tMRP: \n" << this -> mrp.t();
	std::cout << "\tx: \n" << this -> x.t();
	#endif

	while (h >= this -> minimum_h && exit == false) {

		// The ICP is iterated
		for (unsigned int iter = 0; iter < this -> iterations_max; ++iter) {
			
			#if ICP_DEBUG
			std::cout << "ICP iteration " << iter + 1 << " / " << this -> iterations_max  << std::endl;
			std::cout << "Hierchical level : " << std::to_string(h) << std::endl;
			#endif

			if ( next_h == true ) {
				// The pairs are formed only after a change in the hierchical search

				this -> compute_pairs(h,RBK::mrp_to_dcm(this -> mrp),this -> x);
				#if ICP_DEBUG
				ICPBase::save_pairs(this -> point_pairs,"pairs_h_"+std::to_string(h) + "_iter_"+ std::to_string(iter) + ".txt");
				#endif 

				next_h = false;
			}

			if (iter == 0 ) {
				// The initial residuals are computed
				J_0 = this -> compute_rms_residuals(RBK::mrp_to_dcm(this -> mrp),this -> x);
				J = J_0;
			}


			// The matrices of the LS problem are now accumulated
			info_mat.fill(0);
			normal_mat.fill(0);

			
			// #pragma omp parallel for reduction(+:info_mat), reduction(+:normal_mat) if (USE_OMP_ICP)
			for (unsigned int pair_index = 0; pair_index < this -> point_pairs.size(); ++pair_index) {
				
				arma::mat::fixed<6,6> info_mat_temp;
				arma::vec::fixed<6> normal_mat_temp;

				#if ICP_DEBUG
				std::cout << "Building matrix " << pair_index + 1 << " / " << this -> point_pairs.size() << std::endl;
				#endif

				this -> build_matrices(pair_index, this -> mrp,this -> x,info_mat_temp,normal_mat_temp);

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
			J = this -> compute_rms_residuals(RBK::mrp_to_dcm(this -> mrp),this -> x);

			#if ICP_DEBUG
			std::cout << "\nDeviation : " << std::endl;
			std::cout << dX << std::endl;
			std::cout << "\nResiduals: " << J << std::endl;
			std::cout << "MRP: \n" << this -> mrp << std::endl;
			std::cout << "x: \n" << this -> x << std::endl;
			std::cout << "Covariance :\n" << std::endl;
			std::cout << arma::inv(info_mat) << std::endl;
			#endif


			if ( J / J_0 <= this -> rel_tol || J == 0 || h == 0) {
				exit = true;

				break;
			}

			if ( (std::abs(J - J_previous) / J <= this -> s_tol && h > this -> minimum_h) || iter + 1 < this -> iterations_max) {
				h = h - 1;
				next_h = true;

				J_previous = std::numeric_limits<double>::infinity();

				break;
			}

			else if (iter + 1 == this -> iterations_max && h == this -> minimum_h) {
				this -> pc_source -> save("max_iter.obj",this -> dcm_save * RBK::mrp_to_dcm(this -> mrp),this -> dcm_save * this -> x + this -> x_save);
				exit = true;
				break;
			}

			// The postfit residuals become the prefit residuals of the next iteration
			J_previous = J;

		}
	}

	#if ICP_DEBUG
	std::cout << "Leaving ICPBase. Best transform: \n";
	std::cout << "\tmrp: " << this -> mrp.t();
	std::cout << "\tx: " << this -> x.t();
	std::cout << "\tResiduals: " << J << std::endl;
	#endif

	
	
	try{
		this -> R = arma::inv(info_mat);
	}
	catch(std::runtime_error & e){
		std::cout << e.what();
	}
	this -> J_res = J ;

}



void ICPBase::register_pc_RANSAC(double fraction_inliers_used,
	double fraction_inliers_requested,
	unsigned int iter_ransac_max,
	double acceptance_threshold_error, 
	arma::mat::fixed<3,3> dcm_0,
	arma::vec::fixed<3> X_0 ){

	double J_best_RANSAC = std::numeric_limits<double>::infinity();

	arma::mat::fixed<3,3> dcm_best_RANSAC;
	arma::vec::fixed<3>	x_best_RANSAC ;
	std::vector<PointPair> best_pairs_RANSAC,pairs_RANSAC;

	arma::mat::fixed<6,6> info_mat;
	arma::vec::fixed<6> normal_mat;



	if (this -> iterations_max == 0 || iter_ransac_max == 0){
		return;
	}

	#if ICP_DEBUG
	auto start = std::chrono::system_clock::now();

	std::cout << "Computing pc_destination descriptors...\n";
	#endif

	if (this -> use_FPFH){
		this -> pc_destination -> compute_feature_descriptors(PC::FeatureDescriptor::FPFHDescriptor,this -> keep_correlations,
			this -> N_bins,this -> neighborhood_radius);
	}
	else{
		this -> pc_destination -> compute_feature_descriptors(PC::FeatureDescriptor::PFHDescriptor,this -> keep_correlations,
			this -> N_bins,this -> neighborhood_radius);
	}

	#if ICP_DEBUG
	std::cout << "Computing pc_source descriptors...\n";
	#endif

	if (this -> use_FPFH){
		this -> pc_source -> compute_feature_descriptors(PC::FeatureDescriptor::FPFHDescriptor,this -> keep_correlations,
			this -> N_bins,this -> neighborhood_radius);
	}
	else{
		this -> pc_source -> compute_feature_descriptors(PC::FeatureDescriptor::PFHDescriptor,this -> keep_correlations,
			this -> N_bins,this -> neighborhood_radius);
	}





	#if ICP_DEBUG
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
	std::cout << "Time elapsed computing features: " << elapsed_seconds.count()<< " (s)"<< std::endl;
	this -> pc_source->save_point_descriptors("source_descriptors.txt");
	this -> pc_destination ->save_point_descriptors("destination_descriptors.txt");
	std::cout << "Matching descriptors...\n";
	start = std::chrono::system_clock::now();
	#endif

	auto all_matches = PC::find_pch_matches_kdtree(this -> pc_source,this -> pc_destination);
	
	int n_samples = (int)(all_matches.size() * fraction_inliers_used);

	#if ICP_DEBUG
	end = std::chrono::system_clock::now();
	elapsed_seconds = end-start;
	std::cout << "Time elapsed matching features: " << elapsed_seconds.count()<< " (s)"<< std::endl;
	std::cout << "Total number of matches: "+ std::to_string(all_matches.size()) + " \n";
	ICPBase::save_pairs(all_matches,"all_pairs.txt");
	#endif

	arma::ivec indices = arma::regspace<arma::ivec>(0,all_matches.size() - 1);
	
	for (unsigned int iter_ransac = 0; iter_ransac < iter_ransac_max; ++iter_ransac){

		#if ICP_DEBUG
		std::cout <<  "RANSAC iteration " << iter_ransac + 1 << " / " << iter_ransac_max  << std::endl;
		#endif

		double J  = std::numeric_limits<double>::infinity();
		double J_0  = std::numeric_limits<double>::infinity();
		
		// The batch estimator is initialized
		arma::vec::fixed<3> mrp = RBK::dcm_to_mrp(dcm_0);
		arma::vec::fixed<3> x_temp = X_0;

		// Drawing random pairs from the matches
		this -> point_pairs.clear();
		pairs_RANSAC.clear();
		indices = arma::shuffle(indices);

		for (int k = 0; k < n_samples; ++k){
			this -> point_pairs.push_back(all_matches[indices[k]]);
			pairs_RANSAC.push_back(all_matches[indices[k]]);
		}

		// The ICP is iterated
		for (unsigned int iter = 0; iter < this -> iterations_max; ++iter) {

		#if ICP_DEBUG
			std::cout << "\t ICP iteration " << iter + 1 << " / " << this -> iterations_max  << std::endl;
		#endif

			if (iter == 0 ) {
				// The initial residuals are computed
				J_0 = this -> compute_rms_residuals(RBK::mrp_to_dcm(mrp),x_temp);
				J = J_0;
			}


			// The matrices of the LS problem are now accumulated
			info_mat.fill(0);
			normal_mat.fill(0);


			// #pragma omp parallel for reduction(+:info_mat), reduction(+:normal_mat) if (USE_OMP_ICP)
			for (unsigned int pair_index = 0; pair_index < this -> point_pairs.size(); ++pair_index) {

				arma::mat::fixed<6,6> info_mat_temp;
				arma::vec::fixed<6> normal_mat_temp;

				#if ICP_DEBUG
				std::cout << "Building matrix " << pair_index + 1 << " / " << this -> point_pairs.size() << std::endl;
				#endif

				this -> build_matrices(pair_index, mrp,x_temp,info_mat_temp,normal_mat_temp);

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
			mrp = RBK::dcm_to_mrp(RBK::mrp_to_dcm(dmrp) * RBK::mrp_to_dcm(mrp));

			x_temp = x_temp + dx;

			// the mrp is switched to its shadow if need be
			if (arma::norm(mrp) > 1) {
				mrp = - mrp / ( pow(arma::norm(mrp), 2));
			}

			// The postfit residuals are computed
			J = this -> compute_rms_residuals(RBK::mrp_to_dcm(mrp),x_temp);

		#if ICP_DEBUG
			std::cout << "\nDeviation : " << std::endl;
			std::cout << dX << std::endl;
			std::cout << "\nResiduals: " << J << std::endl;
			std::cout << "MRP: \n" << mrp << std::endl;
			std::cout << "x: \n" << x_temp << std::endl;
		#endif

		}

		// End of the ICP Loop, the surrogate model parameters have been fitted to the randomly sampled pair.
		// Let's see how well this model explains the rest of the data

		int good_inlier_not_used_count = 0;
		
		for (int k = n_samples; k < all_matches.size();  ++k){	

			auto point_pair = all_matches[indices[k]];

			if ( this-> compute_distance(point_pair, RBK::mrp_to_dcm(mrp),x_temp) < acceptance_threshold_error){
				this -> point_pairs.push_back(point_pair);
				++good_inlier_not_used_count;
			}
		}

		double fraction_inliers_found = ((double)(good_inlier_not_used_count) / all_matches.size() + fraction_inliers_used);


		#if ICP_DEBUG
		std::cout << "Model has found " << 100 * fraction_inliers_found << " (%) of inliers total (need " << fraction_inliers_requested * 100 <<  "  (%) to validate) \n";
		#endif

		// If good_inlier_not_used_count is greater than what is prescribed, we have found a good model
		if (fraction_inliers_found > fraction_inliers_requested){
			
			arma::mat::fixed<3,3> better_dcm = RBK::mrp_to_dcm(mrp);
			arma::vec::fixed<3> better_x = x_temp;

			double J_better = this -> compute_rms_residuals(this -> point_pairs,better_dcm,better_x);

			// If the good model we found surpasses the previous one, we keep it
			if (J_better < J_best_RANSAC){

				J_best_RANSAC = J_better;
				dcm_best_RANSAC = better_dcm;
				x_best_RANSAC = better_x;
				best_pairs_RANSAC = pairs_RANSAC;
				
				#if ICP_DEBUG
				std::cout << "Found better model with J = " << J_best_RANSAC << " explaining " << this -> point_pairs.size() << " feature pairs using "+ std::to_string(n_samples) +  " data points\n";
				#endif
			}
		}
	}

	#if ICP_DEBUG
	std::cout << "Leaving RANSAC. Best transform: \n";
	std::cout << "\tmrp: " << RBK::dcm_to_mrp(dcm_best_RANSAC).t();
	std::cout << "\tx: " << x_best_RANSAC.t();
	std::cout << "\tResiduals: " << J_best_RANSAC << std::endl;
	ICPBase::save_pairs(best_pairs_RANSAC,"ransac_pairs_aligned.txt",dcm_best_RANSAC,x_best_RANSAC);
	ICPBase::save_pairs(best_pairs_RANSAC,"ransac_pairs.txt");
	#endif

	this -> x = x_best_RANSAC;
	this -> mrp = RBK::dcm_to_mrp(dcm_best_RANSAC);
	this -> J_res = J_best_RANSAC;

}

void ICPBase::save_pairs(std::vector<PointPair> pairs,std::string path,const arma::mat::fixed<3,3> & dcm,const arma::vec::fixed<3> & x){

	arma::mat pairs_m(pairs.size(),6);

	for (int i = 0; i < pairs.size(); ++i){

		pairs_m.submat(i,0,i,2) = (dcm * pairs[i].first -> get_point() + x).t();
		pairs_m.submat(i,3,i,5) = pairs[i].second -> get_point().t();

	}

	pairs_m.save(path,arma::raw_ascii);

}

void ICPBase::pca_prealignment(arma::vec::fixed<3> & mrp,arma::vec::fixed<3> & x) const{


	arma::mat::fixed<3,3> E_source = this -> pc_source -> get_principal_axes();
	arma::mat::fixed<3,3> E_destination = this -> pc_destination -> get_principal_axes();

	// The center of the source point cloud is aligned with that of the destination point cloud
	x = this -> pc_destination -> get_center() - E_destination * E_source.t() * this -> pc_source -> get_center();
	mrp = RBK::dcm_to_mrp(E_destination * E_source.t());

}	



bool ICPBase::get_use_pca_prealignment() const{
	return this -> use_pca_prealignment;
}
void ICPBase::set_use_pca_prealignment(bool use_pca_prealignment){
	this -> use_pca_prealignment = use_pca_prealignment;
}



bool ICPBase::get_use_FPFH() const{
	return this -> use_pca_prealignment;
}
void ICPBase::set_use_FPFH(bool use_FPFH){
	this -> use_FPFH = use_FPFH;
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

void ICPBase::set_keep_correlations( bool keep_correlations){
	this -> keep_correlations = keep_correlations;
}
bool ICPBase::get_keep_correlation()  const{
	return this -> keep_correlations;
}

unsigned int ICPBase::get_N_bins() const{
	return this -> N_bins;
}
void ICPBase::set_N_bins(unsigned int N_bins){
	this -> N_bins = N_bins;
}

double ICPBase::get_neighborhood_radius() const{
	return this -> neighborhood_radius;
}
void ICPBase::set_neighborhood_radius(double neighborhood_radius){
	this -> neighborhood_radius = neighborhood_radius;
}
