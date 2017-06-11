#ifndef HEADER_INTERP
#define HEADER_INTERP

#include <armadillo>

class Interpolator {
public:
	Interpolator(arma::vec * T, arma::mat * X);
	arma::vec interpolate(double t);

protected:
	arma::vec * T;
	arma::mat * X;
}


#endif