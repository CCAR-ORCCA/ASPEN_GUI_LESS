#include <armadillo>



int main(){




	arma::mat mat = arma::randn<arma::mat>(3,1000);

	arma::mat cov = arma::cov(mat.t());

	std::cout << cov << std::endl;
	


	return 0;
}