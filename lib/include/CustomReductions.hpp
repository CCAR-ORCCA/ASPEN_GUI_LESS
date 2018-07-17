#ifndef CUSTOM_REDUCTIONS_H
#define CUSTOM_REDUCTIONS_H


#pragma omp declare reduction (+ : arma::vec::fixed<6> : omp_out += omp_in)\
initializer( omp_priv = omp_orig )

#pragma omp declare reduction (+ : arma::mat::fixed<3,3>  : omp_out += omp_in)\
initializer( omp_priv = omp_orig )

#pragma omp declare reduction (+ : arma::mat::fixed<6,6> : omp_out += omp_in)\
initializer( omp_priv = omp_orig )


#endif