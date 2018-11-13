#include "Dynamics.hpp"

#define POINT_MASS_JAC_ATTITUDE_DXDT_INERTIAL_DEBUG 0
#define POINT_MASS_ATTITUDE_DXDT_INERTIAL_DEBUG 0
#define HARMONICS_ATTITUDE_DXDT_INERTIAL_ESTIMATE_DEBUG 1
#define HARMONICS_JAC_ATTITUDE_DXDT_INERTIAL_ESTIMATE_DEBUG 1
arma::vec::fixed<3> Dynamics::point_mass_acceleration(const arma::vec::fixed<3> & point , double mass) {

	arma::vec::fixed<3> acc = - mass * arma::datum::G / arma::dot(point, point) * arma::normalise(point);
	
	return acc;


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



arma::mat::fixed<6,6> Dynamics::point_mass_jacobian(const arma::vec::fixed<3> & point , double mass) {


	arma::mat::fixed<6,6> A = arma::zeros<arma::mat>(6,6);

	A.submat(0,3,2,5) = arma::eye<arma::mat>(3,3);
	A.submat(3,0,5,2) = - mass * arma::datum::G * (
		arma::eye<arma::mat>(3,3) / std::pow(arma::norm(point),3) 
		- 3 * point * point.t() / std::pow(arma::norm(point),5));

	return A;

}



arma::vec Dynamics::point_mass_attitude_dxdt_inertial_truth(double t,const arma::vec & X, const Args & args) {

	arma::vec::fixed<3> pos = X . subvec(0, 2);
	arma::vec X_small_body = X . subvec(6, 11);
	arma::vec acc_body_grav = Dynamics::point_mass_acceleration(pos , args. get_mass_truth());

	arma::vec dxdt = arma::zeros<arma::vec>(12);
	arma::vec dxdt_spacecraft = { X(3), X(4), X(5), acc_body_grav(0), acc_body_grav(1), acc_body_grav(2)};
	
	dxdt.subvec(0,5) = dxdt_spacecraft;
	dxdt.subvec(6,11) = Dynamics::attitude_dxdt_truth(t, X_small_body, args);

	return dxdt;

}



arma::vec Dynamics::harmonics_attitude_dxdt_inertial_truth(double t,const arma::vec & X, const Args & args) {

	// Inertial position
	arma::vec pos = X . subvec(0, 2);

	arma::vec X_small_body = X . subvec(6, 11);

	// DCM BN
	arma::mat BN = RBK::mrp_to_dcm(X_small_body.subvec(0,3));

	// Body frame position
	pos = BN * pos;


	// Gravity acceleration expressed in the body frame
	arma::vec acc = args.get_sbgat_harmonics_truth() -> GetAcceleration(pos);

	// Mapping it back to the inertial frame
	acc = BN.t() * acc;

	arma::vec dxdt = arma::zeros<arma::vec>(12);
	arma::vec dxdt_spacecraft = { X(3), X(4), X(5), acc(0), acc(1), acc(2)};
	
	dxdt.subvec(0,5) = dxdt_spacecraft;
	dxdt.subvec(6,11) = Dynamics::attitude_dxdt_truth(t, X_small_body, args);

	return dxdt;

}


arma::vec Dynamics::point_mass_attitude_dxdt_inertial_estimate(double t,const arma::vec & X, const Args & args) {

	

	arma::vec::fixed<3> pos = X . subvec(0, 2);
	arma::vec::fixed<6> X_small_body = X . subvec(6, 11);
	arma::vec acc_body_grav = Dynamics::point_mass_acceleration(pos , args. get_mass_estimate());

	arma::vec dxdt = arma::zeros<arma::vec>(12);
	arma::vec dxdt_spacecraft = { X(3), X(4), X(5), acc_body_grav(0), acc_body_grav(1), acc_body_grav(2)};
	
	arma::vec dxdt_small_body = Dynamics::attitude_dxdt_estimate(t, X_small_body, args);
	
	dxdt.subvec(0,5) = dxdt_spacecraft;
	dxdt.subvec(6,11) = dxdt_small_body;



	return dxdt;

}



arma::mat Dynamics::point_mass_jac_attitude_dxdt_inertial_estimate(double t, const arma::vec & X, const Args & args){


	
	arma::mat A = arma::zeros<arma::mat>(12,12);

	arma::vec::fixed<3> pos = X . subvec(0, 2);
	arma::vec::fixed<6> attitude = X . subvec(6, 11);


	A.submat(0,0,5,5) += Dynamics::point_mass_jacobian(pos , args . get_mass_estimate());
	
	A.submat(6,6,11,11) += Dynamics::attitude_jacobian(attitude , args . get_inertia_estimate());


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
	arma::mat::fixed<3,3> BN = RBK::mrp_to_dcm(X_small_body.subvec(0,3));

	// Body frame position
	pos = BN * pos;


	// Gravity acceleration expressed in the body frame
	arma::vec::fixed<3> acc = args.get_sbgat_harmonics_estimate() -> GetAcceleration(pos);

	// Mapping it back to the inertial frame
	acc = BN.t() * acc;

	arma::vec::fixed<12> dxdt = arma::zeros<arma::vec>(12);
	arma::vec::fixed<6> dxdt_spacecraft = { X(3), X(4), X(5), acc(0), acc(1), acc(2)};
	
	dxdt.subvec(0,5) = dxdt_spacecraft;
	dxdt.subvec(6,11) = Dynamics::attitude_dxdt_truth(t, X_small_body, args);

	#if HARMONICS_ATTITUDE_DXDT_INERTIAL_ESTIMATE_DEBUG
	std::cout << "leaving Dynamics::harmonics_attitude_dxdt_inertial_estimate\n";
	#endif
	return dxdt;

}

arma::mat Dynamics::harmonics_jac_attitude_dxdt_inertial_estimate(double t,const arma::vec & X, const Args & args){

	#if HARMONICS_JAC_ATTITUDE_DXDT_INERTIAL_ESTIMATE_DEBUG
	std::cout << "in harmonics_jac_attitude_dxdt_inertial_estimate\n";
	#endif
	arma::mat A = arma::zeros<arma::mat>(12,12);

	arma::vec::fixed<3> pos = X . subvec(0, 2);
	arma::vec::fixed<6> attitude = X . subvec(6, 11);

	// DCM BN
	arma::mat::fixed<3,3> BN = RBK::mrp_to_dcm(attitude.subvec(0,3));

	// Body frame position
	arma::vec::fixed<3> pos_B = BN * pos;

	#if HARMONICS_JAC_ATTITUDE_DXDT_INERTIAL_ESTIMATE_DEBUG
	std::cout << "Getting spherical harmonics acceleration\n";
	#endif
	arma::mat::fixed<3,3> gravity_gradient_mat;
	args.get_sbgat_harmonics_estimate() -> GetGravityGradientMatrix(pos_B,gravity_gradient_mat);
	



	// Partial derivatives of the spacecraft state.

	// drdot/dr is zero

	// drdot/drdot
	A.submat(0,3,2,5) = arma::eye<arma::mat>(3,3);

	// drddot/dr
	A.submat(3,0,5,2) = BN.t() * gravity_gradient_mat * BN;

	// drddot/drdot is zero

	// drddot/dsigma
	arma::vec::fixed<3> acc_body_grav = args.get_sbgat_harmonics_estimate() -> GetAcceleration(pos_B);

	A.submat(3,6,5,8) = 4 * (BN.t() * gravity_gradient_mat * BN * RBK::tilde(pos) - RBK::tilde(BN.t() * acc_body_grav));


	// The small body is not affected by the spacecraft state
	A.submat(6,6,11,11) += Dynamics::attitude_jacobian(attitude , args . get_inertia_estimate());

	#if HARMONICS_JAC_ATTITUDE_DXDT_INERTIAL_ESTIMATE_DEBUG
	std::cout << "leaving harmonics_jac_attitude_dxdt_inertial_estimate\n";
	#endif
	return A;


}


arma::vec Dynamics::attitude_dxdt_estimate(double t, const arma::vec & X, const Args & args) {

	arma::vec dxdt = RBK::dXattitudedt(t, X , args . get_inertia_estimate());

	return dxdt;

}

arma::vec Dynamics::attitude_dxdt_truth(double t, const arma::vec & X, const Args & args) {

	arma::vec dxdt = RBK::dXattitudedt(t, X , args . get_inertia_truth());

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


	arma::mat::fixed<6,6> A = arma::zeros<arma::mat>(6,6);
	arma::vec::fixed<3> sigma = attitude_state.subvec(0,2);
	arma::vec::fixed<3> omega = attitude_state.subvec(3,5);

	// dsigma_dot_dsigma
	A.submat(0,0,2,2) = (0.5 * (- omega * sigma.t() - RBK::tilde(omega) 
		+ arma::eye<arma::mat>(3,3)* arma::dot(sigma,omega) + sigma * omega.t()));
	// dsigma_dot_domega
	A.submat(0,3,2,5) = 1./4 * RBK::Bmat(sigma);
	// domega_dot_dsigma is zero 
	// domega_dot_domega
	A.submat(3,3,5,5) = arma::solve(inertia,- RBK::tilde(omega) * inertia + RBK::tilde(inertia * omega));


	return A;

}
