#ifndef HEADER_INTERP
#define HEADER_INTERP

#include <armadillo>
#include <RigidBodyKinematics.hpp>

class Interpolator {
public:
	Interpolator(arma::vec * T, arma::mat * X);
	arma::vec interpolate(double t, bool is_attitude);

protected:
	arma::vec * T;
	arma::mat * X;
};

#endif