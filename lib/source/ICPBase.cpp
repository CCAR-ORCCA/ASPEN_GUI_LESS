#include <ICPBase.hpp>
#include <FeatureMatching.hpp>
#include <EstimationFeature.hpp>
#include <PointCloudIO.hpp>

#define ICP_DEBUG 0


#pragma omp declare reduction (+ : arma::vec::fixed<6> : omp_out += omp_in)\
initializer( omp_priv = arma::zeros<arma::vec>(6) )

#pragma omp declare reduction (+ : arma::mat::fixed<6,6> : omp_out += omp_in)\
initializer( omp_priv = arma::zeros<arma::mat>(6,6) )


ICPBase::ICPBase(){

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




void ICPBase::clear_point_pairs(){
	this -> point_pairs.clear();
}




void ICPBase::register_pc(
	const PointCloud<PointNormal > & pc_source,
	const PointCloud<PointNormal > & pc_destination,
	double los_noise_sd_baseline,
	const arma::mat::fixed<3,3> & M_pc_D,
	const arma::mat::fixed<3,3> & dcm_0,
	const arma::vec::fixed<3> & X_0){


	auto start = std::chrono::system_clock::now();

	double J  = std::numeric_limits<double>::infinity();
	double J_0  = std::numeric_limits<double>::infinity();
	double J_previous = std::numeric_limits<double>::infinity();

	int h = this -> maximum_h;

	bool next_h = false;

	arma::mat::fixed<6,6> info_mat;
	arma::vec::fixed<6> normal_mat;
	arma::vec residual_vector,sigma_vector;

	// The batch estimator is initialized
	this -> mrp = RBK::dcm_to_mrp(dcm_0);
	this -> x = X_0;

	

	if (this -> iterations_max > 0){


	// If no pairs have been provided, they are recomputed
		if(this -> point_pairs.size() == 0){
			this -> compute_pairs(pc_source,
				pc_destination,h,RBK::mrp_to_dcm(this -> mrp),this -> x);
			
			this -> hierarchical = true;

		}
		else{
			this -> hierarchical = false;
		}

		J_0 = this -> compute_residuals(pc_source,
			pc_destination,RBK::mrp_to_dcm(this -> mrp),this -> x);
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

			if ( (next_h == true && this -> hierarchical) || h == 0) {
				// The pairs are formed only after a change in the hierchical search

		#if ICP_DEBUG
				std::cout << "Updating pairs at h == " << h << std::endl;
		#endif

				this -> compute_pairs(pc_source,
					pc_destination,h,RBK::mrp_to_dcm(this -> mrp),this -> x);
				next_h = false;
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
			residual_vector.resize(this -> point_pairs.size());
			sigma_vector.resize(this -> point_pairs.size());

		#if !__APPLE__
		#pragma omp parallel for reduction(+:info_mat), reduction(+:normal_mat) if (USE_OMP_ICP)
		#endif
			for (unsigned int pair_index = 0; pair_index < this -> point_pairs.size(); ++pair_index) {
				arma::mat::fixed<6,6> info_mat_temp;
				arma::vec::fixed<6> normal_mat_temp;
				this -> build_matrices(

					pc_source,
					pc_destination,
					pair_index, 
					this -> mrp,
					this -> x,
					info_mat_temp,
					normal_mat_temp,
					residual_vector,
					sigma_vector,
					1,
					los_noise_sd_baseline,
					M_pc_D);


				normal_mat += normal_mat_temp;
				info_mat += info_mat_temp;

			}





			#if ICP_DEBUG
			std::cout << "\nInfo mat: " << std::endl;
			std::cout << info_mat << std::endl;
			std::cout << "\nNormal mat: " << std::endl;
			std::cout << normal_mat << std::endl;
			std::cout << "\nNormalized residuals: \n";
			std::cout << arma::stddev(residual_vector/sigma_vector) << std::endl;
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
			J = this -> compute_residuals(pc_source,
				pc_destination,RBK::mrp_to_dcm(this -> mrp),this -> x);

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
		

	#endif

		try{
			this -> R = arma::inv(info_mat);
		}
		catch(std::runtime_error & e){
			std::cout << e.what();
		}
		this -> J_res = J ;

	}


	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end-start;

	std::cout << "Time elapsed in ICP: " << elapsed_seconds.count()<< " (s)"<< std::endl;


}

bool ICPBase::check_convergence(const int & iter,const double & J,const double & J_0, double & J_previous,int & h,bool & next_h){

	// Has converged
	if ( (bool)(J / J_0 <= this -> r_tol || J == 0) &&  h > this -> minimum_h) {

		if (this -> hierarchical){
			h = h - 1;
			next_h = true;

		}
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


double ICPBase::compute_residuals(
	const PC & pc_source,
	const PC & pc_destination,
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

			J += std::pow(this -> compute_distance(pc_source,
				pc_destination,point_pairs[pair_index],  dcm_S,x_S,dcm_D,x_D),2)/ point_pairs.size();
		}
	}
	else{
		#pragma omp parallel for reduction(+:J) if (USE_OMP_ICP)
		for (unsigned int pair_index = 0; pair_index <point_pairs.size(); ++pair_index) {

			J += weights(pair_index) * std::pow(this -> compute_distance(pc_source,
				pc_destination,point_pairs[pair_index],  dcm_S,x_S,dcm_D,x_D),2)/ point_pairs.size();

		}
	}

	return std::sqrt(J);
}






double ICPBase::compute_residuals(
	const PC & pc_source,
	const PC & pc_destination,
	const arma::mat::fixed<3,3> & dcm,
	const arma::vec::fixed<3> & x,
	const arma::vec & weights) const {
	return this -> compute_residuals(pc_source,pc_destination,this -> point_pairs,dcm,x,weights);

}











