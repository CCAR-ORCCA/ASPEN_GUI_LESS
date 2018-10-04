
#include <armadillo>
#include <Eigen/Dense>

#pragma omp declare reduction( + : arma::vec : omp_out += omp_in ) \
initializer( omp_priv = omp_orig )
// #pragma omp declare reduction (EigenPlus : Eigen::VectorXd: omp_out=omp_out+omp_in)\
//      initializer(omp_priv=omp_orig)

int main() {

	int N = 10000;
	int M = 100;
	double a = 0;



	// // Custom reduction

	// Eigen::VectorXd v_eigen = Eigen::VectorXd::Zero(M);
	
	// // Custom reduction, segfaults
	// #pragma omp parallel for reduction(EigenPlus:v_eigen)
	// for (int i = 0; i < N; ++i){
	// 	v_eigen = v_eigen + Eigen::VectorXd::Ones(M);
	// }

	// std::cout << v_eigen << std::endl;



	// Custom reduction, segfaults
	arma::vec v = arma::zeros<arma::vec>(M);

	#pragma omp parallel for reduction(+:v)
	for (int i = 0; i < N; ++i){
		v += arma::ones<arma::vec>(v.n_rows);
	}
	
	std::cout << v << std::endl;
	
	return 0;
}












