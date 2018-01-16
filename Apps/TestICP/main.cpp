
#include "PC.hpp"
#include "ICP.hpp"
#include <armadillo>
#include "boost/progress.hpp"

int main() {



	arma::mat source_pc_mat;
	arma::mat destination_pc_mat;

	source_pc_mat.load("../source.txt");
	destination_pc_mat.load("../destination.txt");

	arma::inplace_trans(source_pc_mat);
	arma::inplace_trans(destination_pc_mat);


	arma::vec u = {1,0,0};

	std::shared_ptr<PC> source_pc = std::make_shared<PC>(PC(u,source_pc_mat));
	std::shared_ptr<PC> destination_pc = std::make_shared<PC>(PC(u,destination_pc_mat));
	source_pc -> save("../original_source.obj");
	destination_pc -> save("../destination.obj");





	unsigned int N_iter = 100;

	boost::progress_display progress(N_iter);

	auto start = std::chrono::system_clock::now();
	
	#pragma omp parallel for
	for (unsigned i = 0; i < N_iter ; ++i){
		++progress;

		ICP icp_pc(
			destination_pc, 
			source_pc, 
			arma::eye<arma::mat>(3,3), 
			arma::zeros<arma::vec>(3));
		

	}

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;


	std::cout << "\nTime elapsed: " << elapsed_seconds.count() << std::endl;

	// arma::mat M = icp_pc.get_M();
	// arma::vec X = icp_pc.get_X();


	// source_pc -> save("../registered_source.obj",M,X);

	// std::cout << "DCM: \n";
	// std::cout << M << std::endl;
	// std::cout << "X: \n";

	// std::cout << X << std::endl;




	return 0;
}












