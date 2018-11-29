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
	Evaluates the acceleration due to gravity at the provided point using a point mass model
	@param X (4 x 1) state {rx,ry,rz,mu}
	@return point mass acceleration expressed in the inertial frame of reference
	*/
	arma::vec point_mass_acceleration(double t,const arma::vec & X, const Args & args) ;


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
	Computes the jacobian of the dynamics of a rigid body undergoing torque free rotation
	@param attitude attitude set and associated angular velocity (6x1)
	@param inertia inertia tensor of rigid body
	@return jacobian of dynamics
	*/
	arma::mat::fixed<6,6> attitude_jacobian(const arma::vec::fixed<6> & attitude_state ,const arma::mat & inertia) ;


	arma::vec point_mass_attitude_dxdt_inertial_truth(double t, const arma::vec & X, const Args & args);
	arma::vec harmonics_attitude_dxdt_inertial_truth(double t,const arma::vec & X, const Args & args);

	arma::vec point_mass_attitude_dxdt_inertial_estimate(double t,const arma::vec & X, const Args & args);
	arma::mat point_mass_jac_attitude_dxdt_inertial_estimate(double t, const arma::vec & X, const Args & args);

	arma::vec harmonics_attitude_dxdt_inertial_estimate(double t,const arma::vec & X, const Args & args);
	arma::mat harmonics_jac_attitude_dxdt_inertial_estimate(double t,const arma::vec & X, const Args & args);

	arma::vec attitude_dxdt_inertial_estimate(double t,const arma::vec & X, const Args & args) ;
	arma::mat attitude_jac_dxdt_inertial_estimate(double t,const arma::vec & X, const Args & args) ;




	arma::mat gamma_OD(double dt);
	arma::mat gamma_OD_augmented(double dt);

	arma::vec debug(double t,const arma::vec & X, const Args & args) ;

	double energy_attitude(double t, arma::vec  X, Args * args);

	arma::vec::fixed<6> attitude_dxdt_truth(double t, const arma::vec & X, const Args & args) ;
	arma::vec::fixed<6> attitude_dxdt_estimate(double t, const arma::vec & X, const Args & args) ;

	arma::mat create_Q(double sigma_vel,double sigma_omeg);
	arma::mat create_Q(double sigma_vel);

}

#endif