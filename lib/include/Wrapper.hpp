#ifndef HEADER_WRAPPER
#define HEADER_WRAPPER

#include "Args.hpp"
#include "DynamicAnalyses.hpp"

#include <RigidBodyKinematics.hpp>
#include <armadillo>


namespace Wrapper{


	arma::vec pgm_dxdt_wrapper(double t, arma::vec X, Args * args);
	arma::vec pgm_dxdt_wrapper_body_frame(double t, arma::vec X, Args * args);
	arma::vec point_mass_dxdt_wrapper_body_frame(double t, arma::vec X, Args * args);
	arma::vec point_mass_dxdt_wrapper(double t, arma::vec X, Args * args) ;

	arma::vec attitude_dxdt_wrapper(double t, arma::vec X, Args * args);

	arma::vec sigma_dot_wrapper(double t, arma::vec X, Args * args);
	arma::vec validation_dxdt_wrapper(double t, arma::vec  X, Args * args);

	arma::vec point_mass_dxdt_wrapper_odeint(double t, const arma::vec & x, const Args & args) ;
	arma::mat point_mass_jac_wrapper_odeint(double t, const arma::vec & x, const Args & args) ;

	arma::vec joint_sb_spacecraft_body_frame_dyn(double t, arma::vec  X, Args * args);

	arma::vec obs_long_lat(double t,const arma::vec & x, const Args & args);
	arma::mat obs_jac_long_lat(double t,const arma::vec & x, const Args & args);


	arma::vec obs_debug(double t,const arma::vec & x, const Args & args);
	arma::mat obs_jac_debug(double t,const arma::vec & x, const Args & args);

	
	arma::mat gamma_OD(double dt);


	double energy_attitude(double t, arma::vec  X, Args * args);
// double energy_orbit(double t, arma::vec X, Args * args);
// double energy_orbit_body_frame(double t, arma::vec X , Args * args);

	arma::vec event_function_mrp_omega(double t , arma::vec X, Args * args);
	arma::vec event_function_mrp(double t , arma::vec X, Args * args);
	arma::vec event_function_collision(double t , arma::vec X, Args * args);
	arma::vec event_function_collision_body_frame(double t, arma::vec X, Args * args);

}

#endif