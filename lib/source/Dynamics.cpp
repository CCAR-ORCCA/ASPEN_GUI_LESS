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
		
		std::cout << "in Dynamics::SRP_cannonball_unit_C\n";
		std::cout << X(0) << std::endl;
		std::cout << Dynamics::SRP_cannonball(t,X,args).t() << std::endl;
		std::cout << Dynamics::SRP_cannonball(t,arma::ones<arma::vec>(1),args).t() << std::endl;
		throw;


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
	const double & mu = X(6);

	
	// DCM BN
	arma::mat::fixed<3,3> BN = RBK::mrp_to_dcm(sigma_BN);

	// Gravity acceleration expressed in the body frame
	arma::vec::fixed<3> acc = args.get_sbgat_harmonics_estimate() -> GetAcceleration(BN * pos);

	acc *= mu / (arma::datum::G * args.get_estimated_shape_model() -> get_volume());

	// Mapping it back to the inertial frame
	
	return BN.t() * acc;

}


arma::mat Dynamics::spherical_harmonics_acceleration_estimate_unit_mu(double t,const arma::vec & X, const Args & args) {


	// Inertial position
	const arma::vec::fixed<3> & pos = X . subvec(0, 2);
	const arma::vec::fixed<3> & sigma_BN = X . subvec(3, 5);

	
	// DCM BN
	arma::mat::fixed<3,3> BN = RBK::mrp_to_dcm(sigma_BN);

	// Gravity acceleration expressed in the body frame
	arma::vec::fixed<3> acc = args.get_sbgat_harmonics_estimate() -> GetAcceleration(BN * pos);
	acc *= 1. / (arma::datum::G * args.get_estimated_shape_model() -> get_volume());


	// Mapping it back to the inertial frame
	
	return BN.t() * acc;



}



arma::vec Dynamics::velocity(double t,const arma::vec & X, const Args & args) {
	return X;
}




arma::mat Dynamics::spherical_harmonics_gravity_gradient_matrix_estimate(double t,const arma::vec & X, const Args & args){

	const arma::vec::fixed<3> & pos = X . subvec(0, 2);
	const arma::vec::fixed<3> & mrp = X . subvec(3, 5);
	const double & mu = X(6);
	
	// DCM BN
	arma::mat::fixed<3,3> BN = RBK::mrp_to_dcm(mrp);

	// Body frame position
	arma::vec::fixed<3> pos_B = BN * pos;

	arma::mat::fixed<3,3> gravity_gradient_mat;
	args.get_sbgat_harmonics_estimate() -> GetGravityGradientMatrix(pos_B,gravity_gradient_mat);
	
	gravity_gradient_mat *= mu / (arma::datum::G * args.get_estimated_shape_model() -> get_volume());


	return BN.t() * gravity_gradient_mat * BN;

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




