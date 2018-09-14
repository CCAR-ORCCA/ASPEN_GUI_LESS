#include "ICPBase.hpp"


ICPBase::ICPBase(std::shared_ptr<PC> pc_destination, std::shared_ptr<PC> pc_source){
	this -> pc_destination = pc_destination;
	this -> pc_source = pc_source;
}

arma::vec::fixed<3> ICPBase::get_x() const{
	return this -> x;
}

arma::mat::fixed<3,3> ICPBase::get_dcm() const{
	return this -> dcm;
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

	

	int h = 7;

	bool exit = false;
	bool next_h = true;

	arma::mat::fixed<6,6> info_mat;
	arma::vec::fixed<6> normal_mat;

	arma::vec::fixed<3> mrp,x_temp;

	if (this -> use_pca_prealignment){
		this -> pca_prealignment(mrp,x_temp);
	}
	else{
		// The batch estimator is initialized
		mrp = RBK::dcm_to_mrp(dcm_0);
		x_temp = X_0;
	}

	if (this -> iterations_max == 0){
		exit = true;
	}

	while (h >= this -> minimum_h && exit == false) {

		// The ICP is iterated
		for (unsigned int iter = 0; iter < this -> iterations_max; ++iter) {
			
			#if ICP_DEBUG
			std::cout << "Iter " << iter + 1 << " / " << this -> iterations_max  << std::endl;
			std::cout << "Hierchical level : " << std::to_string(h) << std::endl;

			#endif

			if ( next_h == true ) {
				// The pairs are formed only after a change in the hierchical search

				this -> compute_pairs(h,RBK::mrp_to_dcm(mrp),x);
				
				next_h = false;
			}

			if (iter == 0 ) {
				// The initial residuals are computed
				J_0 = this -> compute_rms_residuals(RBK::mrp_to_dcm(mrp),x_temp);
				J = J_0;
			}


			// The matrices of the LS problem are now accumulated
			info_mat.fill(0);
			normal_mat.fill(0);

			#if ICP_DEBUG
			std::cout << "Number of valid pairs: " <<  this -> point_pairs.size() << std::endl;
			#endif


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
			std::cout << "Covariance :\n" << std::endl;
			std::cout << arma::inv(info_mat) << std::endl;
			#endif


			if ( J / J_0 <= this -> rel_tol || J == 0) {
				exit = true;

				break;
			}

			if ( std::abs(J - J_previous) / J <= this -> s_tol && h > this -> minimum_h) {
				h = h - 1;
				next_h = true;

				J_previous = std::numeric_limits<double>::infinity();

				break;
			}

			else if (iter + 1 == this -> iterations_max) {
				this -> pc_source -> save("crash.obj",this -> dcm_save * RBK::mrp_to_dcm(mrp),this -> dcm_save * x_temp + this -> x_save);

				throw ICPException();
				break;
			}

			// The postfit residuals become the prefit residuals of the next iteration
			J_previous = J;

		}
	}

	#if ICP_DEBUG
	std::cout << "Leaving ICPBase\n";
	#endif

	this -> x = x_temp;
	this -> dcm = RBK::mrp_to_dcm(mrp);
	try{
		this -> R = arma::inv(info_mat);
	}
	catch(std::runtime_error & e){
		std::cout << e.what();
	}
	this -> J_res = J ;

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

void ICPBase::set_minimum_h(unsigned int min_h){
	this -> minimum_h = min_h;
}
unsigned int ICPBase::get_minimum_h() const{
	return this -> minimum_h;
}
