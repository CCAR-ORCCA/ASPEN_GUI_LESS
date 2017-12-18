#ifndef HEADER_OBSERVATIONS
#define HEADER_OBSERVATIONS

#include <armadillo>
#include "Args.hpp"
#include "Lidar.hpp"
#include "FrameGraph.hpp"
#include "BatchFilter.hpp"


namespace Observations{

	arma::vec obs_long_lat(double t,const arma::vec & x, const Args & args);
	arma::mat obs_jac_long_lat(double t,const arma::vec & x, const Args & args);

	arma::vec obs_debug(double t,const arma::vec & x, const Args & args);
	arma::mat obs_jac_debug(double t,const arma::vec & x, const Args & args);

	arma::vec obs_pos_lidar(double t,const arma::vec & x,const Args & args);
	arma::vec obs_pos_computed(double t,const arma::vec & x,const Args & args);

	arma::mat obs_pos_jac(double t,const arma::vec & x,const Args & args);

	arma::vec obs_lidar_range_true(double t,const arma::vec & x, const Args & args);
	arma::vec obs_lidar_range_computed(double t,const arma::vec & x, const Args & args);
	arma::mat obs_lidar_range_jac(double t,const arma::vec & x, const Args & args);

}



#endif