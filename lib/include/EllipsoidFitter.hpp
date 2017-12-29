#ifndef HEADER_ELLIPSOID_FIT
#define HEADER_ELLISPOID_FIT

#include "PC.hpp"
#include <armadillo>


class EllipsoidFitter{

public:

	EllipsoidFitter(PC * pc){
		this -> pc = pc;
	};
	arma::vec run(const arma::vec & X_bar,const arma::mat Pbar = arma::zeros<arma::mat>(6,6),
		unsigned int N_iter = 1,bool verbose = false) const;
	


protected:

	double G(const arma::vec & point, const arma::vec & X) const;
	arma::rowvec jacobian(const arma::vec & point, const arma::vec & X) const;


	PC * pc;

};










#endif