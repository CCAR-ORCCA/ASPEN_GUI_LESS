#include "Dynamics.hpp"
#include "Ray.hpp"

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

arma::vec Dynamics::SRP_cannonball_truth(double t,const arma::vec & X, const Args & args){

	double au2meters = 149597870700;


	arma::vec::fixed<3> R = args . get_kep_state_small_body().convert_to_cart(t).get_position_vector(); // itokawa position vector w/r to the sun expressed in inertially-pointing barycentered frame
	double srp_flux =  args . get_solar_constant()/std::pow(arma::norm(R + X.subvec(0,2)) / au2meters,2);

	const arma::vec::fixed<3> & sigma_BN = X . subvec(3, 5);

	// DCM BN
	arma::mat::fixed<3,3> BN = RBK::mrp_to_dcm(sigma_BN);

	arma::vec::fixed<3> ray_direction_body_frame = - BN * arma::normalise(R + X.subvec(0,2));
	arma::vec::fixed<3> ray_origin_body_frame = BN * X.subvec(0,2);

	Ray ray(ray_origin_body_frame, ray_direction_body_frame) ;

	if (args . get_true_shape_model() -> ray_trace(&ray)){

		return arma::zeros<arma::vec>(3);
	}
	else{

		return srp_flux / arma::datum::c_0 * X(6) * args.get_area_to_mass_ratio() * arma::normalise(R + X.subvec(0,2));
	}
}


arma::vec Dynamics::SRP_cannonball_estimate(double t,const arma::vec & X, const Args & args){

	double au2meters = 149597870700;

	arma::vec::fixed<3> R = args . get_kep_state_small_body().convert_to_cart(t).get_position_vector(); // itokawa position vector w/r to the sun expressed in inertially-pointing barycentered frame
	double srp_flux =  args . get_solar_constant()/std::pow(arma::norm(R + X.subvec(0,2)) / au2meters,2);

	const arma::vec::fixed<3> & sigma_BN = X . subvec(3, 5);

	// DCM BN
	arma::mat::fixed<3,3> BN = RBK::mrp_to_dcm(sigma_BN);

	arma::vec::fixed<3> ray_direction_body_frame = - BN * arma::normalise(R + X.subvec(0,2));
	arma::vec::fixed<3> ray_origin_body_frame = BN * X.subvec(0,2);

	Ray ray(ray_origin_body_frame, ray_direction_body_frame) ;

	if (args . get_estimated_shape_model() -> ray_trace(&ray)){
		return arma::zeros<arma::vec>(3);
	}
	else{
		return srp_flux / arma::datum::c_0 * X(6) * args.get_area_to_mass_ratio() * arma::normalise(R + X.subvec(0,2));
	}
}

arma::mat Dynamics::SRP_cannonball_unit_C_truth(double t,const arma::vec & X, const Args & args){

	arma::vec X_input = {X(0),X(1),X(2),X(3),X(4),X(5),1};
	return  Dynamics::SRP_cannonball_truth(t,X_input,args);

}

arma::mat Dynamics::SRP_cannonball_unit_C_estimate(double t,const arma::vec & X, const Args & args){

	arma::vec X_input = {X(0),X(1),X(2),X(3),X(4),X(5),1};
	return  Dynamics::SRP_cannonball_estimate(t,X_input,args);

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

	return (0.5 * (- omega * sigma.t() 
		- RBK::tilde(omega) 
		+ arma::eye<arma::mat>(3,3)* arma::dot(sigma,omega) 
		+ sigma * omega.t()));

}


arma::mat Dynamics::partial_mrp_dot_partial_omega(double t, const arma::vec & X, const Args & args){

	return 1./4 * RBK::Bmat(X);

}


arma::mat Dynamics::partial_omega_dot_partial_omega_estimate(double t, const arma::vec & X, const Args & args){

	const arma::vec::fixed<3> & omega = X.subvec(3,5);
	const arma::mat::fixed<3,3> & inertia = args . get_inertia_estimate();

	return arma::solve(inertia,- RBK::tilde(omega) * inertia + RBK::tilde(inertia * omega));
}

arma::vec Dynamics::third_body_acceleration_itokawa_sun(double t, 
	const arma::vec & X,
	const Args & args){

	const arma::vec::fixed<3> & r = X.subvec(0,2);// spacecraft position in inertially-pointing barycentered frame

	arma::vec::fixed<3> R = args . get_kep_state_small_body().convert_to_cart(t).get_position_vector(); // itokawa position vector w/r to the sun expressed in inertially-pointing barycentered frame



	return  args . get_kep_state_small_body().get_mu() * (- (r + R) / std::pow(arma::norm((r + R)),3) + R / std::pow(arma::norm(R),3));


}



