#include <armadillo>


void test(const arma::vec & v){
	std::cout << "in vec" << std::endl;

	std::cout<< v << std::endl;
	std::cout << v.subvec(0,20) << std::endl;
	std::cout << v.subvec(0,99) << std::endl;

}


int main(){

	arma::arma_rng::set_seed(0);

	arma::vec v = arma::randn<arma::vec>(100);

	std::cout << v << std::endl;
	std::cout << v.subvec(0,20) << std::endl;
	std::cout << v.subvec(0,99) << std::endl;

	
	test(v);



	return 0;
}