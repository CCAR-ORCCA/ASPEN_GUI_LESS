#ifndef IOD_BOUNDS_HEADEr
#define IOD_BOUNDS_HEADEr

#include <armadillo>

#define A_MIN 750
#define A_MAX 1500

#define E_MIN 0.01
#define E_MAX 0.9999

#define I_MIN 0
#define I_MAX arma::datum::pi 

#define RAAN_MIN -arma::datum::pi 
#define RAAN_MAX arma::datum::pi 

#define OMEGA_MIN -arma::datum::pi 
#define OMEGA_MAX arma::datum::pi  

#define M0_MIN 0 
#define M0_MAX 2 * arma::datum::pi  

#define MU_MIN 1
#define MU_MAX 3 

#endif