#include "Dynamics.hpp"

#define POINT_MASS_JAC_ATTITUDE_DXDT_INERTIAL_DEBUG 0
#define POINT_MASS_ATTITUDE_DXDT_INERTIAL_DEBUG 0
#define HARMONICS_ATTITUDE_DXDT_INERTIAL_TRUTH_DEBUG 0
#define HARMONICS_ATTITUDE_DXDT_INERTIAL_ESTIMATE_DEBUG 0
#define HARMONICS_JAC_ATTITUDE_DXDT_INERTIAL_ESTIMATE_DEBUG 0
#define ATTITUDE_DXDT_INERTIAL_ESTIMATE_DEBUG 0
#define ATTITUDE_DXDT_ESTIMATE_DEBUG 0
#define ATTITUDE_JAC_DXDT_INERTIAL_ESTIMATE_DEBUG 0


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




arma::mat Dynamics::point_mass_gravity_gradient_matrix(double t,const arma::vec & X, const Args & args) {

	// X = {r,mu}

	const arma::vec::fixed<3> & point = X.subvec(0,2);
	const double & mu = X(3);

	arma::mat A = - mu * (
		arma::eye<arma::mat>(3,3) / std::pow(arma::norm(point),3) 
		- 3 * point * point.t() / std::pow(arma::norm(point),5));

	return A;

}


arma::vec Dynamics::point_mass_acceleration(double t,const arma::vec & X, const Args & args) {

	const arma::vec::fixed<3> & point = X.subvec(0,2);
	const double & mu = X(3);

	arma::vec acc = - mu / arma::dot(point, point) * arma::normalise(point);
	
	return acc;


}

arma::vec Dynamics::SRP_cannonball(double t,const arma::vec & X, const Args & args){

	double srp_flux = std::pow(args . get_distance_from_sun_AU(),2) * args . get_solar_constant();
	const arma::vec::fixed<3> & sun_to_spc_direction = args.get_sun_to_spc_direction();


	return (srp_flux / arma::datum::c_0) * X(0) * (args.get_area_to_mass_ratio()) * sun_to_spc_direction;

}

arma::mat Dynamics::SRP_cannonball_unit_C(double t,const arma::vec & X, const Args & args){
	return  Dynamics::SRP_cannonball(t,arma::ones<arma::vec>(1),args);

}

arma::mat Dynamics::point_mass_acceleration_unit_mu(double t,const arma::vec & X, const Args & args) {

	const arma::vec::fixed<3> & point = X.subvec(0,2);

	arma::mat acc = - 1. / arma::dot(point, point) * arma::normalise(point);
	
	return acc;

}


arma::mat Dynamics::identity_33(double t,const arma::vec & X, const Args & args) {
	return arma::eye<arma::mat>(3,3);
}





arma::vec Dynamics::point_mass_attitude_dxdt_inertial_truth(double t,const arma::vec & X, const Args & args) {

	throw(std::runtime_error("to reimplement"));
	arma::vec dxdt = arma::zeros<arma::vec>(12);

	// arma::vec::fixed<3> pos = X . subvec(0, 2);
	// arma::vec X_small_body = X . subvec(6, 11);
	// arma::vec acc_body_grav = Dynamics::point_mass_acceleration(pos , args. get_mass_truth());

	// arma::vec dxdt_spacecraft = { X(3), X(4), X(5), acc_body_grav(0), acc_body_grav(1), acc_body_grav(2)};
	
	// dxdt.subvec(0,5) = dxdt_spacecraft;
	// dxdt.subvec(6,11) = Dynamics::attitude_dxdt_truth(t, X_small_body, args);

	// return dxdt;

}


arma::vec Dynamics::attitude_dxdt_inertial_estimate(double t,const arma::vec & X, const Args & args) {


	#if ATTITUDE_DXDT_INERTIAL_ESTIMATE_DEBUG
	std::cout << "in Dynamics::attitude_dxdt_inertial_estimate\n";
	#endif

	arma::vec dxdt = Dynamics::attitude_dxdt_estimate(t, X, args);

	return dxdt;

}



arma::mat Dynamics::attitude_jac_dxdt_inertial_estimate(double t, const arma::vec & X, const Args & args){
	#if ATTITUDE_JAC_DXDT_INERTIAL_ESTIMATE_DEBUG
	std::cout << "in attitude_jac_dxdt_inertial_estimate\n";
	#endif
	arma::mat A = Dynamics::attitude_jacobian(X , args . get_inertia_estimate());
	
	#if ATTITUDE_JAC_DXDT_INERTIAL_ESTIMATE_DEBUG
	std::cout << "leaving attitude_jac_dxdt_inertial_estimate\n";
	#endif

	return A;

}

arma::vec Dynamics::harmonics_attitude_dxdt_inertial_truth(double t,const arma::vec & X, const Args & args) {


	#if HARMONICS_ATTITUDE_DXDT_INERTIAL_TRUTH_DEBUG
	std::cout << "in Dynamics::harmonics_attitude_dxdt_inertial_truth\n";
	#endif

	
	// Inertial position
	arma::vec::fixed<3> pos = X . subvec(0, 2);
	arma::vec::fixed<6> X_small_body = X . subvec(6, 11);

	#if HARMONICS_ATTITUDE_DXDT_INERTIAL_TRUTH_DEBUG
	std::cout << "extracting DCM\n";
	#endif
	// DCM BN
	arma::mat::fixed<3,3> BN = RBK::mrp_to_dcm(X_small_body.subvec(0,2));

	// Body frame position
	pos = BN * pos;

	#if HARMONICS_ATTITUDE_DXDT_INERTIAL_TRUTH_DEBUG
	std::cout << "getting acceleration\n";
	#endif

	// Gravity acceleration expressed in the body frame
	arma::vec::fixed<3> acc = args.get_sbgat_harmonics_truth() -> GetAcceleration(pos);

	// Mapping it back to the inertial frame
	acc = BN.t() * acc;

	arma::vec::fixed<13> dxdt = arma::zeros<arma::vec>(13);
	arma::vec::fixed<6> dxdt_spacecraft = { X(3), X(4), X(5), acc(0), acc(1), acc(2)};
	
	dxdt.subvec(0,5) = dxdt_spacecraft;
	dxdt.subvec(6,11) = Dynamics::attitude_dxdt_truth(t, X_small_body, args);


	#if HARMONICS_ATTITUDE_DXDT_INERTIAL_TRUTH_DEBUG
	std::cout << "leaving Dynamics::harmonics_attitude_dxdt_inertial_truth\n";
	#endif
	return dxdt;

}



arma::vec Dynamics::spherical_harmonics_acceleration_truth(double t,const arma::vec & X, const Args & args) {


	// Inertial position
	const arma::vec::fixed<3> & pos = X . subvec(0, 2);
	const arma::vec::fixed<3> & sigma_BN = X . subvec(3, 5);

	
	// DCM BN
	arma::mat::fixed<3,3> BN = RBK::mrp_to_dcm(sigma_BN);

	// Gravity acceleration expressed in the body frame
	arma::vec::fixed<3> acc = args.get_sbgat_harmonics_truth() -> GetAcceleration(BN * pos);

	// Mapping it back to the inertial frame
	
	return BN.t() * acc;;

}


arma::vec Dynamics::spherical_harmonics_acceleration_estimate(double t,const arma::vec & X, const Args & args) {


	// Inertial position
	const arma::vec::fixed<3> & pos = X . subvec(0, 2);
	const arma::vec::fixed<3> & sigma_BN = X . subvec(3, 5);

	
	// DCM BN
	arma::mat::fixed<3,3> BN = RBK::mrp_to_dcm(sigma_BN);

	// Gravity acceleration expressed in the body frame
	arma::vec::fixed<3> acc = args.get_sbgat_harmonics_estimate() -> GetAcceleration(BN * pos);

	// Mapping it back to the inertial frame
	
	return BN.t() * acc;;

}








arma::vec Dynamics::point_mass_attitude_dxdt_inertial_estimate(double t,const arma::vec & X, const Args & args) {

	throw(std::runtime_error("to reimplement"));
	arma::vec dxdt = arma::zeros<arma::vec>(12);

	// arma::vec::fixed<3> pos = X . subvec(0, 2);
	// arma::vec::fixed<6> X_small_body = X . subvec(6, 11);
	// arma::vec acc_body_grav = Dynamics::point_mass_acceleration(pos , args. get_mass_estimate());

	// arma::vec dxdt_spacecraft = { X(3), X(4), X(5), acc_body_grav(0), acc_body_grav(1), acc_body_grav(2)};
	
	// arma::vec dxdt_small_body = Dynamics::attitude_dxdt_estimate(t, X_small_body, args);
	
	// dxdt.subvec(0,5) = dxdt_spacecraft;
	// dxdt.subvec(6,11) = dxdt_small_body;



	return dxdt;

}


arma::vec Dynamics::velocity(double t,const arma::vec & X, const Args & args) {
	return X;
}


arma::mat Dynamics::point_mass_jac_attitude_dxdt_inertial_estimate(double t, const arma::vec & X, const Args & args){

	throw(std::runtime_error("to reimplement"));
	
	arma::mat A = arma::zeros<arma::mat>(12,12);

	// arma::vec::fixed<3> pos = X . subvec(0, 2);
	// arma::vec::fixed<6> attitude = X . subvec(6, 11);


	// A.submat(0,0,5,5) += Dynamics::point_mass_jacobian(pos , args . get_mass_estimate());
	
	// A.submat(6,6,11,11) += Dynamics::attitude_jacobian(attitude , args . get_inertia_estimate());


	return A;


}

arma::vec Dynamics::harmonics_attitude_dxdt_inertial_estimate(double t,const arma::vec & X, const Args & args){
	
	#if HARMONICS_ATTITUDE_DXDT_INERTIAL_ESTIMATE_DEBUG
	std::cout << "in Dynamics::harmonics_attitude_dxdt_inertial_estimate\n";
	#endif

	// Inertial position
	arma::vec::fixed<3> pos = X . subvec(0, 2);

	arma::vec::fixed<6> X_small_body = X . subvec(6, 11);

	// DCM BN
	arma::mat::fixed<3,3> BN = RBK::mrp_to_dcm(X_small_body.subvec(0,2));

	// Body frame position
	pos = BN * pos;

	// Gravity acceleration expressed in the body frame
	arma::vec::fixed<3> acc = X(12)/ (arma::datum::G * args.get_estimated_shape_model() -> get_volume()) * args.get_sbgat_harmonics_estimate() -> GetAcceleration(pos);

	// Mapping it back to the inertial frame
	acc = BN.t() * acc;

	arma::vec::fixed<13> dxdt = arma::zeros<arma::vec>(13);
	arma::vec::fixed<6> dxdt_spacecraft = { X(3), X(4), X(5), acc(0), acc(1), acc(2)};
	
	dxdt.subvec(0,5) = dxdt_spacecraft;
	dxdt.subvec(6,11) = Dynamics::attitude_dxdt_estimate(t, X_small_body, args);

	#if HARMONICS_ATTITUDE_DXDT_INERTIAL_ESTIMATE_DEBUG
	std::cout << "leaving Dynamics::harmonics_attitude_dxdt_inertial_estimate\n";
	#endif
	return dxdt;

}

arma::mat Dynamics::harmonics_jac_attitude_dxdt_inertial_estimate(double t,const arma::vec & X, const Args & args){

	#if HARMONICS_JAC_ATTITUDE_DXDT_INERTIAL_ESTIMATE_DEBUG
	std::cout << "in harmonics_jac_attitude_dxdt_inertial_estimate\n";
	#endif
	arma::mat A = arma::zeros<arma::mat>(13,13);

	arma::vec::fixed<3> pos = X . subvec(0, 2);
	arma::vec::fixed<6> attitude = X . subvec(6, 11);

	// DCM BN
	arma::mat::fixed<3,3> BN = RBK::mrp_to_dcm(attitude.subvec(0,2));

	// Body frame position
	arma::vec::fixed<3> pos_B = BN * pos;

	#if HARMONICS_JAC_ATTITUDE_DXDT_INERTIAL_ESTIMATE_DEBUG
	std::cout << "Getting spherical harmonics acceleration\n";
	std::cout << args.get_sbgat_harmonics_estimate() << std::endl;
	#endif

	arma::mat::fixed<3,3> gravity_gradient_mat;
	args.get_sbgat_harmonics_estimate() -> GetGravityGradientMatrix(pos_B,gravity_gradient_mat);
	
	gravity_gradient_mat *= X(12)/ (arma::datum::G * args.get_estimated_shape_model() -> get_volume());

	// Partial derivatives of the spacecraft state.

	// drdot/dr is zero

	// drdot/drdot
	A.submat(0,3,2,5) = arma::eye<arma::mat>(3,3);

	// drddot/dr
	A.submat(3,0,5,2) = BN.t() * gravity_gradient_mat * BN;

	// drddot/drdot is zero
	#if HARMONICS_JAC_ATTITUDE_DXDT_INERTIAL_ESTIMATE_DEBUG
	std::cout << "Getting partials\n";
	#endif
	
	// drddot/dsigma
	arma::vec::fixed<3> acc_body_grav = X(12)/ (arma::datum::G * args.get_estimated_shape_model() -> get_volume()) * args.get_sbgat_harmonics_estimate() -> GetAcceleration(pos_B);

	A.submat(3,6,5,8) = 4 * (BN.t() * gravity_gradient_mat * BN * RBK::tilde(pos) - RBK::tilde(BN.t() * acc_body_grav));


	// The small body is not affected by the spacecraft state
	A.submat(6,6,11,11) += Dynamics::attitude_jacobian(attitude , args . get_inertia_estimate());





	// drddot/dmu

	A.submat(3,12,5,12) = acc_body_grav / X(12);

	// omega_dot is not affected by a varying mu because 
	// the rhs of [I]omega_dot = ... is proportional to [I], hence mu





	#if HARMONICS_JAC_ATTITUDE_DXDT_INERTIAL_ESTIMATE_DEBUG
	std::cout << "leaving harmonics_jac_attitude_dxdt_inertial_estimate\n";
	#endif
	return A;

}


arma::vec::fixed<6> Dynamics::attitude_dxdt_estimate(double t, const arma::vec & X, const Args & args) {
	
	#if ATTITUDE_DXDT_ESTIMATE_DEBUG
	std::cout << "in attitude_dxdt_estimate\n";
	#endif

	arma::vec::fixed<6> dxdt = RBK::dXattitudedt(t, X , args . get_inertia_estimate());

	#if ATTITUDE_DXDT_ESTIMATE_DEBUG
	std::cout << "leaving attitude_dxdt_estimate\n";
	#endif

	return dxdt;

}

arma::vec::fixed<6> Dynamics::attitude_dxdt_truth(double t, const arma::vec & X, const Args & args) {

	arma::vec::fixed<6> dxdt = RBK::dXattitudedt(t, X , args . get_inertia_truth());

	return dxdt;

}


arma::vec Dynamics::dmrp_dt(double t, const arma::vec & X, const Args & args) {

	return RBK::dmrpdt(t, X);
}


arma::vec Dynamics::domega_dt_estimate(double t, const arma::vec & X, const Args & args) {

	return RBK::domegadt( t, X, args . get_inertia_estimate());
}

arma::vec Dynamics::domega_dt_truth(double t, const arma::vec & X, const Args & args) {

	return RBK::domegadt( t, X, args . get_inertia_truth());
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



arma::mat::fixed<6,6> Dynamics::attitude_jacobian(const arma::vec::fixed<6> & attitude_state ,const arma::mat & inertia) {


	#if ATTITUDE_JACOBIAN_DEBUG
	std::cout << "in Dynamics::attitude_jacobian\n";
	#endif

	arma::mat::fixed<6,6> A = arma::zeros<arma::mat>(6,6);
	const arma::vec::fixed<3> & sigma = attitude_state.subvec(0,2);
	const arma::vec::fixed<3> & omega = attitude_state.subvec(3,5);

	// dsigma_dot_dsigma
	A.submat(0,0,2,2) = (0.5 * (- omega * sigma.t() - RBK::tilde(omega) 
		+ arma::eye<arma::mat>(3,3)* arma::dot(sigma,omega) + sigma * omega.t()));
	// dsigma_dot_domega
	A.submat(0,3,2,5) = 1./4 * RBK::Bmat(sigma);
	// domega_dot_dsigma is zero 
	// domega_dot_domega
	A.submat(3,3,5,5) = arma::solve(inertia,- RBK::tilde(omega) * inertia + RBK::tilde(inertia * omega));


	#if ATTITUDE_JACOBIAN_DEBUG
	std::cout << "exiting Dynamics::attitude_jacobian\n";
	#endif
	return A;

}


arma::mat Dynamics::partial_mrp_dot_partial_mrp(double t, const arma::vec & X, const Args & args){

	const arma::vec::fixed<3> & sigma = X.subvec(0,2);
	const arma::vec::fixed<3> & omega = X.subvec(3,5);

	return (0.5 * (- omega * sigma.t() - RBK::tilde(omega) 
		+ arma::eye<arma::mat>(3,3)* arma::dot(sigma,omega) + sigma * omega.t()));

}


arma::mat Dynamics::partial_mrp_dot_partial_omega(double t, const arma::vec & X, const Args & args){

	return 1./4 * RBK::Bmat(X);

}


arma::mat Dynamics::partial_omega_dot_partial_omega_estimate(double t, const arma::vec & X, const Args & args){

	const arma::vec::fixed<3> & sigma = X.subvec(0,2);
	const arma::vec::fixed<3> & omega = X.subvec(3,5);
	const arma::mat::fixed<3,3> & inertia = args . get_inertia_estimate();

	return arma::solve(inertia,- RBK::tilde(omega) * inertia + RBK::tilde(inertia * omega));
}




