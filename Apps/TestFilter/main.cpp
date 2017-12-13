#include "BatchFilter.hpp"


arma::vec sinusoid(double t, arma::vec X , const Args & args){
	
	arma::vec Y = {X[0] * std::cos(X[1] * t)};
	return Y;
}

arma::mat H_sinusoid(double t, arma::vec X , const Args & args){
	
	arma::mat H = arma::zeros<arma::mat>(1,X.n_rows);

	H(0,0) = std::cos(X[1] * t);
	H(0,1) = - X[0] * t * std::sin(X[1] * t);

	return H;
}

int main(){

	Args args;
	BatchFilter filter(args);
	filter.set_observations_fun(sinusoid,H_sinusoid);

	std::vector<double> T_obs;

	unsigned int N = 2000;
	for (unsigned int i =0; i < N; ++i){
		T_obs.push_back( double(i)/(N - 1) * 4 * arma::datum::pi );
	}

	arma::vec X0_true = {1,0.5};
	arma::vec X_bar_0 = {0.9,0.6};
	arma::mat R = 0.0025 * arma::eye<arma::mat>(1,1);

	int iter = filter.run(20,X0_true,X_bar_0,T_obs,R);
	std::cout << "Converged in " << iter << " iterations\n";
	std::cout << filter.get_estimated_state_history()[0] << std::endl;

	filter.write_estimated_state("./X_hat.txt");
	filter.write_true_obs("./Y_true.txt");
	filter.write_T_obs(T_obs,"./T_obs.txt");
	filter.write_residuals("./residuals.txt");



	return 0;

}