#ifndef HEADER_DYNAMICS
#define HEADER_DYNAMICS

#include <armadillo>
#include "Args.hpp"

namespace Dynamics{

	arma::vec pgm_dxdt(double t, arma::vec X, Args * args);
	arma::vec pgm_dxdt_body_frame(double t, arma::vec X, Args * args);
	arma::vec point_mass_dxdt_body_frame(double t, arma::vec X, Args * args);
	arma::vec point_mass_dxdt(double t, arma::vec X, Args * args) ;


	arma::vec sigma_dot(double t, arma::vec X, Args * args);
	arma::vec validation_dxdt(double t, arma::vec  X, Args * args);

	arma::vec point_mass_dxdt_odeint(double t, const arma::vec & x, const Args & args) ;
	arma::mat point_mass_jac_odeint(double t, const arma::vec & x, const Args & args) ;

	arma::vec joint_sb_spacecraft_body_frame_dyn(double t, arma::vec  X, Args * args);

	arma::vec point_mass_attitude_dxdt_body_frame(double t, const arma::vec & X, const Args & args);
	arma::mat point_mass_jac_attitude_dxdt_body_frame(double t, const arma::vec & X, const Args & args);
	arma::vec harmonics_attitude_dxdt_body_frame(double t,const arma::vec & X, const Args & args);

	arma::vec point_mass_attitude_dxdt_inertial(double t, const arma::vec & X, const Args & args);
	arma::mat point_mass_jac_attitude_dxdt_inertial(double t, const arma::vec & X, const Args & args);
	arma::vec harmonics_attitude_dxdt_inertial(double t,const arma::vec & X, const Args & args);

	arma::vec estimated_point_mass_attitude_dxdt_body_frame(double t,const arma::vec & X, const Args & args);
	arma::mat estimated_point_mass_jac_attitude_dxdt_body_frame(double t, const arma::vec & X, const Args & args);

	arma::vec estimated_point_mass_attitude_dxdt_inertial(double t,const arma::vec & X, const Args & args);
	arma::mat estimated_point_mass_jac_attitude_dxdt_inertial(double t, const arma::vec & X, const Args & args);


	arma::mat gamma_OD(double dt);
	arma::mat gamma_OD_augmented(double dt);

	arma::vec debug(double t,const arma::vec & X, const Args & args) ;

	double energy_attitude(double t, arma::vec  X, Args * args);

	arma::vec estimated_attitude_dxdt(double t, const arma::vec & X, const Args & args) ;
	arma::vec true_attitude_dxdt(double t, const arma::vec & X, const Args & args) ;

	arma::mat create_Q(double sigma_vel,double sigma_omeg);
	arma::mat create_Q(double sigma_vel);

}










#endif