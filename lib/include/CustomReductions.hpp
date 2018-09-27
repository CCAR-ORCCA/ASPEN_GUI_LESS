#ifndef CUSTOM_REDUCTIONS_H
#define CUSTOM_REDUCTIONS_H
#include <omp.h>
#include <armadillo>

#pragma omp declare reduction (+ : arma::vec::fixed<6> : omp_out += omp_in)\
initializer( omp_priv = arma::zeros<arma::vec>(6) )

#pragma omp declare reduction (+ : arma::mat::fixed<3,3>  : omp_out += omp_in)\
initializer( omp_priv = arma::zeros<arma::mat>(3,3) )

#pragma omp declare reduction (+ : arma::mat::fixed<6,6> : omp_out += omp_in)\
initializer( omp_priv = arma::zeros<arma::mat>(6,6) )

#endif