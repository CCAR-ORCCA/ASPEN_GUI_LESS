#ifndef HEADER_DYNAMICS
#define HEADER_DYNAMICS

#include <armadillo>
#include "Args.hpp"

namespace Dynamics{

	arma::vec point_mass_mu_dxdt_odeint(double t, const arma::vec & x, const Args & args) ;
	arma::mat point_mass_mu_jac_odeint(double t, const arma::vec & x, const Args & args) ;



	/**
	Dummy interface 
	@param X (3 x 1) state {rx_dot,ry_dot,rz_dot,mu}
	@return {rx_dot,ry_dot,rz_dot}
	*/
	arma::vec velocity(double t,const arma::vec & X, const Args & args) ;


	/**
	Identity jacobian 3x3
	@param X (not used)
	@return identity matrix 3x3
	*/
	arma::mat identity_33(double t,const arma::vec & X, const Args & args) ;


	/**
	Evaluates the acceleration due to SRP at the provided point using a cannonball model assuming constant solar flux and no eclipses
	@param X (1) state {C == 1} (C ==  SRP constant == 1)
	@return point mass acceleration expressed in the inertial frame of reference
	*/
	arma::mat SRP_cannonball_unit_C(double t,const arma::vec & X, const Args & args);

	/**
	Evaluates the acceleration due to gravity at the provided point using a point mass model
	@param X (4 x 1) state {rx,ry,rz,mu}
	@return point mass acceleration expressed in the inertial frame of reference
	*/
	arma::vec point_mass_acceleration(double t,const arma::vec & X, const Args & args) ;


	/**
	Evaluates the acceleration due to SRP at the provided point using a cannonball model assuming constant solar flux and no eclipses
	@param X (1) state {C} (C ==  SRP constant)
	@return point mass acceleration expressed in the inertial frame of reference
	*/
	arma::vec SRP_cannonball(double t,const arma::vec & X, const Args & args) ;


	/**
	Evaluates the acceleration due to gravity at the provided point using a point mass model
	and mu == 1
	@param X (4 x 1) state {rx,ry,rz}
	@return point mass acceleration expressed in the inertial frame of reference
	*/
	arma::mat point_mass_acceleration_unit_mu(double t,const arma::vec & X, const Args & args);

	/**
	Evaluates the gravity gradient matrix at the provided point using a point mass model
	@param X (4 x 1) state {rx,ry,rz,mu}
	@return gravity gradient expressed in the inertial frame of reference
	*/
	arma::mat point_mass_gravity_gradient_matrix(double t,const arma::vec & X, const Args & args);

	
	/**
	Computes the acceleration caused by a spherical harmonics gravity model
	@param t time
	@param X {position_N_frame,mrp_BN} 6x1
	@param args structure of extra arguments. Used to extract args.get_sbgat_harmonics_truth()
	*/
	arma::vec spherical_harmonics_acceleration_truth(double t,const arma::vec & X, const Args & args) ;

	/**
	Computes the acceleration caused by a spherical harmonics gravity model
	@param t time
	@param X (7 x 1) state {rx,ry,rz,sigma_1,sigma_2,sigma_3,mu}. The position vector {rx,ry,rz} is expressed in the inertial frame.
	The mrp {sigma_1,sigma_2,sigma_3} govers the [BN] dcm (B == body frame in which the spherical harmonics where computed.)
	@param args structure of extra arguments. Used to extract args.get_sbgat_harmonics_estimate()
	*/
	arma::vec spherical_harmonics_acceleration_estimate(double t,const arma::vec & X, const Args & args) ;

	/**
	Computes the acceleration caused by a spherical harmonics gravity model under the assumption that mu == 1
	@param t time
	@param X (7 x 1) state {rx,ry,rz,sigma_1,sigma_2,sigma_3}. The position vector {rx,ry,rz} is expressed in the inertial frame.
	The mrp {sigma_1,sigma_2,sigma_3} govers the [BN] dcm (B == body frame in which the spherical harmonics where computed.)
	@param args structure of extra arguments. Used to extract args.get_sbgat_harmonics_estimate()
	*/
	arma::mat spherical_harmonics_acceleration_estimate_unit_mu(double t,const arma::vec & X, const Args & args) ;



	/**
	Evaluates the gravity gradient matrix at the provided point using a spherical harmonics gravity model
	@param t time
	@param X (7 x 1) state {rx,ry,rz,sigma_1,sigma_2,sigma_3,mu}. The position vector {rx,ry,rz} is expressed in the inertial frame.
	The mrp {sigma_1,sigma_2,sigma_3} govers the [BN] dcm (B == body frame in which the spherical harmonics where computed.)
	@return gravity gradient expressed in the inertial frame of reference
	*/
	arma::mat spherical_harmonics_gravity_gradient_matrix_estimate(double t,const arma::vec & X, const Args & args);

	/**
	Time derivative of a MRP set
	@param t time
	@param X {mrp_set,angular_velocity} (6x1)
	@param args structure of extra arguments (not used)
	*/
	arma::vec dmrp_dt(double t, const arma::vec & X, const Args & args) ;

	/**
	Computes the time derivative of the angular velocity using the estimated inertia
	@param t
	@param X {mrp_set,angular_velocity} (6x1)
	@param args structure of extra arguments. Uses args . get_inertia_estimate()
	*/
	arma::vec domega_dt_estimate(double t, const arma::vec & X, const Args & args) ;

	/**
	Computes the time derivative of the angular velocity using the true inertia
	@param t
	@param X {mrp_set,angular_velocity} (6x1)
	@param args structure of extra arguments. Uses args . get_inertia_truth()
	*/
	arma::vec domega_dt_truth(double t, const arma::vec & X, const Args & args) ;




	/**
	Computes the partial derivative of the time derivative of a MRP set with respect to the mrp set 
	@param t time
	@param X {mrp_set,angular_velocity} (6x1)
	@param args structure of extra arguments (not used)
	*/
	arma::mat partial_mrp_dot_partial_mrp(double t, const arma::vec & x, const Args & args);

	/**
	Computes the partial derivative of the time derivative of a MRP set with respect to the angular velocity 
	@param t time
	@param X {mrp_set} (3x1)
	@param args structure of extra arguments (not used)
	*/
	arma::mat partial_mrp_dot_partial_omega(double t, const arma::vec & X, const Args & args);

	/**
	Computes the partial derivative of the time derivative of the angular velocity set with respect to the angular velocity 
	@param t time
	@param X {mrp_set} (3x1)
	@param args structure of extra arguments. Uses args . get_inertia_estimate()
	*/
	arma::mat partial_omega_dot_partial_omega_estimate(double t, const arma::vec & X, const Args & args);


	arma::mat gamma_OD(double dt);
	arma::mat gamma_OD_augmented(double dt);

	arma::mat create_Q(double sigma_vel,double sigma_omeg);
	arma::mat create_Q(double sigma_vel);

}

#endif