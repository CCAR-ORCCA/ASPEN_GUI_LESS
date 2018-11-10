#include "Dynamics.hpp"
#define ESTIMATED_POINT_MASS_JAC_ATTITUDE_DXDT_INERTIAL_DEBUG 1
#define ESTIMATED_POINT_MASS_ATTITUDE_DXDT_INERTIAL 1



arma::vec Dynamics::point_mass_dxdt(double t, arma::vec X, Args * args) {

	arma::vec pos_inertial = X . subvec(0, 2);
	arma::vec acc_inertial = args -> get_dyn_analyses() -> point_mass_acceleration(pos_inertial , args -> get_mass());

	arma::vec dxdt = { X(3), X(4), X(5), acc_inertial(0), acc_inertial(1), acc_inertial(2)};

	return dxdt;

}


arma::vec Dynamics::point_mass_dxdt_odeint(double t, const arma::vec & x, const Args & args) {
	arma::vec pos_inertial = x . subvec(0, 2);
	arma::vec acc_inertial = args.get_dyn_analyses() -> point_mass_acceleration(pos_inertial , args . get_mass());

	arma::vec dxdt = {x(3), x(4), x(5), acc_inertial(0), acc_inertial(1), acc_inertial(2)};

	return dxdt;

}


arma::vec Dynamics::point_mass_mu_dxdt_odeint(double t, const arma::vec & x, const Args & args) {
	
	arma::vec r = x . rows(0, 2);

	arma::vec dxdt(7);
	dxdt.rows(0,2) = x.rows(3,5);
	dxdt.rows(3,5) = - x(6)/arma::dot(r,r) * arma::normalise(r);
	dxdt(6) = 0;

	return dxdt;

}

arma::mat Dynamics::point_mass_mu_jac_odeint(double t, const arma::vec & x, const Args & args) {

	arma::mat A = arma::zeros<arma::mat>(7,7);
	arma::vec r = x . subvec(0, 2);
	A.submat(0,3,2,5) = arma::eye<arma::mat>(3,3);
	A.submat(3,0,5,2) = x(6) / std::pow(arma::dot(r,r),3./2.) * (3 * r * r.t()/arma::dot(r,r) - arma::eye<arma::mat>(3,3));

	A.submat(3,6,5,6) = - arma::normalise(r)/ arma::dot(r,r);

	return A ;

}



arma::mat Dynamics::point_mass_jac_odeint(double t, const arma::vec & x, const Args & args) {

	arma::vec pos_inertial = x . subvec(0, 2);
	return args.get_dyn_analyses() -> point_mass_jacobian(pos_inertial , args . get_mass());

}

arma::mat Dynamics::gamma_OD(double dt){
	arma::mat gamma = arma::zeros<arma::mat>(6,3);
	gamma.submat(0,0,2,2) = 0.5 * dt * arma::eye<arma::mat>(3,3);
	gamma.submat(3,0,5,2) = arma::eye<arma::mat>(3,3);

	gamma = dt * gamma;
	return gamma;
}


arma::mat Dynamics::gamma_OD_augmented(double dt){
	arma::mat gamma = arma::zeros<arma::mat>(12,3);
	gamma.submat(0,0,2,2) = 0.5 * dt * arma::eye<arma::mat>(3,3);
	gamma.submat(3,0,5,2) = arma::eye<arma::mat>(3,3);

	gamma = dt * gamma;
	return gamma;
}



arma::vec Dynamics::point_mass_attitude_dxdt_body_frame(double t,const arma::vec & X, const Args & args) {

	arma::vec pos_body = X . subvec(0, 2);
	arma::vec vel_body = X . subvec(3, 5);

	arma::vec X_spacecraft = X . subvec(0, 5);

	arma::vec mrp_TN = X . subvec(6, 8);
	arma::vec omega_TN = X . subvec(9, 11);

	arma::vec X_small_body = X . subvec(6, 11);

	arma::vec acc_body_grav = args. get_dyn_analyses() -> point_mass_acceleration(pos_body , args. get_mass());
	arma::vec acc_body_frame = acc_body_grav - (2 * arma::cross(omega_TN, vel_body) + omega_TN * omega_TN.t() * pos_body - pos_body * omega_TN.t() * omega_TN);

	arma::vec dxdt = arma::zeros<arma::vec>(12);
	arma::vec dxdt_spacecraft = { X(3), X(4), X(5), acc_body_frame(0), acc_body_frame(1), acc_body_frame(2)};
	
	arma::vec dxdt_small_body = Dynamics::true_attitude_dxdt(t, X_small_body, args);
	
	dxdt.subvec(0,5) = dxdt_spacecraft;
	dxdt.subvec(6,11) = dxdt_small_body;

	return dxdt;

}


arma::vec Dynamics::estimated_point_mass_attitude_dxdt_body_frame(double t,const arma::vec & X, const Args & args) {

	arma::vec pos_body = X . subvec(0, 2);
	arma::vec vel_body = X . subvec(3, 5);

	arma::vec X_spacecraft = X . subvec(0, 5);

	arma::vec mrp_TN = X . subvec(6, 8);
	arma::vec omega_TN = X . subvec(9, 11);

	arma::vec X_small_body = X . subvec(6, 11);

	arma::vec acc_body_grav = args. get_dyn_analyses() -> point_mass_acceleration(pos_body , args. get_estimated_mass());
	arma::vec acc_body_frame = acc_body_grav - (2 * arma::cross(omega_TN, vel_body) + omega_TN * omega_TN.t() * pos_body - pos_body * omega_TN.t() * omega_TN);

	arma::vec dxdt = arma::zeros<arma::vec>(12);
	arma::vec dxdt_spacecraft = { X(3), X(4), X(5), acc_body_frame(0), acc_body_frame(1), acc_body_frame(2)};
	
	arma::vec dxdt_small_body = Dynamics::estimated_attitude_dxdt(t, X_small_body, args);
	
	dxdt.subvec(0,5) = dxdt_spacecraft;
	dxdt.subvec(6,11) = dxdt_small_body;

	return dxdt;

}


arma::vec Dynamics::harmonics_attitude_dxdt_body_frame(double t,const arma::vec & X, const Args & args) {

	arma::vec pos_body = X . subvec(0, 2);
	arma::vec vel_body = X . subvec(3, 5);

	arma::vec X_spacecraft = X . subvec(0, 5);

	arma::vec mrp_TN = X . subvec(6, 8);
	arma::vec omega_TN = X . subvec(9, 11);

	arma::vec X_small_body = X . subvec(6, 11);

	arma::vec acc_body_grav = args. get_dyn_analyses() -> spherical_harmo_acc(
		args.get_harmonics_degree(),
		args.get_ref_radius(),
		args.get_mu(),
		pos_body, 
		args.get_Cnm(),
		args.get_Snm());
	

	arma::vec acc_body_frame = acc_body_grav - (2 * arma::cross(omega_TN, vel_body) + omega_TN * omega_TN.t() * pos_body - pos_body * omega_TN.t() * omega_TN);

	arma::vec dxdt = arma::zeros<arma::vec>(12);
	arma::vec dxdt_spacecraft = { X(3), X(4), X(5), acc_body_frame(0), acc_body_frame(1), acc_body_frame(2)};


	arma::vec dxdt_small_body = Dynamics::true_attitude_dxdt(t, X_small_body, args);
	
	dxdt.subvec(0,5) = dxdt_spacecraft;
	dxdt.subvec(6,11) = dxdt_small_body;

	return dxdt;

}

arma::mat Dynamics::point_mass_jac_attitude_dxdt_body_frame(double t, const arma::vec & X, const Args & args){


	#if ESTIMATED_POINT_MASS_JAC_ATTITUDE_DXDT_INERTIAL_DEBUG
	std::cout << "in Dynamics::point_mass_jac_attitude_dxdt_body_frame\n";
	#endif 

	arma::mat A = arma::zeros<arma::mat>(12,12);

	arma::vec::fixed<3> pos_body = X . subvec(0, 2);
	arma::vec::fixed<3> vel_body = X . subvec(3, 5);

	arma::vec::fixed<3> mrp_TN = X . subvec(6, 8);
	arma::vec::fixed<3> omega_TN = X . subvec(9, 11);

	A.submat(0,0,5,5) += args.get_dyn_analyses() -> point_mass_jacobian(pos_body , args . get_mass());
	A.submat(3,0,5,2) += - omega_TN * omega_TN.t() + arma::eye<arma::mat>(3,3) * arma::dot(omega_TN,omega_TN);
	A.submat(3,3,5,5) = - 2 * RBK::tilde(omega_TN);


	#if ESTIMATED_POINT_MASS_JAC_ATTITUDE_DXDT_INERTIAL_DEBUG
	std::cout << "leaving Dynamics::point_mass_jac_attitude_dxdt_body_frame\n";
	#endif 
	return A;

}


arma::mat Dynamics::estimated_point_mass_jac_attitude_dxdt_body_frame(double t, const arma::vec & X, const Args & args){

	arma::mat A = arma::zeros<arma::mat>(12,12);

	arma::vec pos_body = X . subvec(0, 2);
	arma::vec vel_body = X . subvec(3, 5);

	arma::vec mrp_TN = X . subvec(6, 8);
	arma::vec omega_TN = X . subvec(9, 11);


	A.submat(0,0,5,5) += args.get_dyn_analyses() -> point_mass_jacobian(pos_body , args . get_estimated_mass());

	A.submat(3,0,5,2) += - omega_TN * omega_TN.t() + arma::eye<arma::mat>(3,3) * arma::dot(omega_TN,omega_TN);

	A.submat(3,3,5,5) = - 2 * RBK::tilde(omega_TN);


	return A;


}



arma::vec Dynamics::point_mass_attitude_dxdt_inertial(double t,const arma::vec & X, const Args & args) {

	arma::vec pos = X . subvec(0, 2);

	arma::vec X_small_body = X . subvec(6, 11);

	arma::vec acc_body_grav = args. get_dyn_analyses() -> point_mass_acceleration(pos , args. get_mass());

	arma::vec dxdt = arma::zeros<arma::vec>(12);
	arma::vec dxdt_spacecraft = { X(3), X(4), X(5), acc_body_grav(0), acc_body_grav(1), acc_body_grav(2)};
	
	dxdt.subvec(0,5) = dxdt_spacecraft;
	dxdt.subvec(6,11) = Dynamics::true_attitude_dxdt(t, X_small_body, args);

	return dxdt;

}

arma::vec Dynamics::harmonics_attitude_dxdt_inertial(double t,const arma::vec & X, const Args & args) {

	// Inertial position
	arma::vec pos = X . subvec(0, 2);

	arma::vec X_small_body = X . subvec(6, 11);

	// DCM BN
	arma::mat BN = RBK::mrp_to_dcm(X_small_body.subvec(0,3));

	// Body frame position
	pos = BN * pos;



	// Gravity acceleration expressed in the body frame
	arma::vec acc = args.get_sbgat_harmonics() -> GetAcceleration(pos);

	// Mapping it back to the inertial frame
	acc = BN.t() * acc;

	arma::vec dxdt = arma::zeros<arma::vec>(12);
	arma::vec dxdt_spacecraft = { X(3), X(4), X(5), acc(0), acc(1), acc(2)};
	
	dxdt.subvec(0,5) = dxdt_spacecraft;
	dxdt.subvec(6,11) = Dynamics::true_attitude_dxdt(t, X_small_body, args);

	return dxdt;

}


arma::vec Dynamics::estimated_point_mass_attitude_dxdt_inertial(double t,const arma::vec & X, const Args & args) {


	#if ESTIMATED_POINT_MASS_ATTITUDE_DXDT_INERTIAL_DEBUG
	std::cout << "in Dynamics::estimated_point_mass_attitude_dxdt_inertial\n";
	#endif 



	arma::vec pos = X . subvec(0, 2);

	arma::vec X_small_body = X . subvec(6, 11);

	arma::vec acc_body_grav = args. get_dyn_analyses() -> point_mass_acceleration(pos , args. get_estimated_mass());

	arma::vec dxdt = arma::zeros<arma::vec>(12);
	arma::vec dxdt_spacecraft = { X(3), X(4), X(5), acc_body_grav(0), acc_body_grav(1), acc_body_grav(2)};
	
	arma::vec dxdt_small_body = Dynamics::estimated_attitude_dxdt(t, X_small_body, args);
	
	dxdt.subvec(0,5) = dxdt_spacecraft;
	dxdt.subvec(6,11) = dxdt_small_body;


	#if ESTIMATED_POINT_MASS_ATTITUDE_DXDT_INERTIAL_DEBUG
	std::cout << "leaving Dynamics::estimated_point_mass_attitude_dxdt_inertial\n";
	#endif 


	return dxdt;

}



arma::mat Dynamics::point_mass_jac_attitude_dxdt_inertial(double t, const arma::vec & X, const Args & args){

	arma::mat A = arma::zeros<arma::mat>(12,12);

	arma::vec pos = X . subvec(0, 2);

	A.submat(0,0,5,5) += args.get_dyn_analyses() -> point_mass_jacobian(pos , args . get_mass());

	return A;

}


arma::mat Dynamics::estimated_point_mass_jac_attitude_dxdt_inertial(double t, const arma::vec & X, const Args & args){

	arma::mat A = arma::zeros<arma::mat>(12,12);

	arma::vec pos = X . subvec(0, 2);
	arma::vec attitude = X . subvec(6, 11);


	A.submat(0,0,5,5) += args.get_dyn_analyses() -> point_mass_jacobian(pos , args . get_estimated_mass());
	A.submat(6,6,11,11) += args.get_dyn_analyses() -> attitude_jacobian(attitude , args . get_estimated_inertia());


	return A;


}

arma::vec Dynamics::estimated_attitude_dxdt(double t, const arma::vec & X, const Args & args) {

	arma::vec dxdt = RBK::dXattitudedt(t, X , args . get_estimated_inertia());

	return dxdt;

}

arma::vec Dynamics::true_attitude_dxdt(double t, const arma::vec & X, const Args & args) {

	arma::vec dxdt = RBK::dXattitudedt(t, X , args . get_true_inertia());

	return dxdt;

}

arma::mat Dynamics::create_Q(double sigma_vel,double sigma_omeg){
	arma::mat Q = arma::zeros<arma::mat>(6,6);
	Q.submat(0,0,2,2) = std::pow(sigma_vel,2) * arma::eye<arma::mat>(3,3);
	Q.submat(3,3,5,5) = std::pow(sigma_omeg,2) * arma::eye<arma::mat>(3,3);
	return Q;
}
arma::mat Dynamics::create_Q(double sigma_vel){
	
	return std::pow(sigma_vel,2) * arma::eye<arma::mat>(3,3);
}

