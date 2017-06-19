#ifndef HEADER_WRAPPER
#define HEADER_WRAPPER

#include "Args.hpp"
#include "DynamicAnalyses.hpp"

#include <RigidBodyKinematics.hpp>
#include <armadillo>

arma::vec pgm_dxdt_wrapper(double t, arma::vec X, Args * args);
arma::vec pgm_dxdt_wrapper_body_frame(double t, arma::vec X, Args * args);

arma::vec attitude_dxdt_wrapper(double t, arma::vec X, Args * args);

double energy_attitude(double t, arma::vec  X, Args * args);
double energy_orbit(double t, arma::vec X, Args * args);
double energy_orbit_body_frame(double t, arma::vec X , Args * args);

arma::vec event_function_mrp(double t , arma::vec X, Args * args);
arma::vec event_function_collision(double t , arma::vec X, Args * args);
arma::vec event_function_collision_body_frame(double t, arma::vec X, Args * args);



#endif