#include "ICPBase.hpp"



ICPBase::ICPBase(std::shared_ptr<PC> pc_destination, std::shared_ptr<PC> pc_source,
	bool verbose = true,
	const arma::mat::fixed<3,3> & M_save,
	const arma::vec::fixed<3> & X_save){
	this -> pc_destination = pc_destination;
	this -> pc_source = pc_source;
	this -> M_save = M_save;
	this -> X_save = X_save;
}

arma::vec::fixed<3> ICPBase::get_X() const{
	return this -> X;
}

arma::mat::fixed<3,3> ICPBase::get_M() const{
	return this -> M;
}

arma::mat::fixed<6,6> ICPBase::get_R() const{
	return this -> R;
}

double ICPBase::get_J_res() const{
	return this -> J_res;
}


std::vector<PointPair > * ICPBase::ICP::get_point_pairs() {
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




