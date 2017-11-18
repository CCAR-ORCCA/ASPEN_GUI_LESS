#include "Bezier.hpp"
#include "Facet.hpp"
#include "PC.hpp"
#include "ControlPoint.hpp"

#include <chrono>


int main(){

	arma::arma_rng::set_seed(0);

	std::vector<std::shared_ptr<ControlPoint> > vertices;

	unsigned int n = 2;
	unsigned int N = 100000;
	unsigned int N_points = (n + 1) * (n + 2) / 2;


	for (unsigned int i = 0; i < N_points; ++i){
		std::shared_ptr<ControlPoint> v = std::make_shared<ControlPoint>(ControlPoint());
		v -> set_coordinates(arma::randu<arma::vec>(3));
		vertices.push_back(v);
	}

	Bezier patch(vertices);
	arma::mat surface_points(3,N);

	
	for (unsigned int d = n; d < 35; ++d){

		// for (unsigned int i = 0; i < N; ++i){
		// 	arma::vec chi = arma::randu(2);
		// 	double u = chi(0);
		// 	double v = (1 - u) * chi(1);
		// 	surface_points.col(i) = patch.evaluate(u,v);
		// }

		// std::ofstream shape_file;
		// shape_file.open("patch_" +std::to_string(d) + ".obj");

		// for (unsigned int i = 0; i < N; ++i) {
		// 	shape_file << "v " << surface_points(0,i) << " " << surface_points(1,i) << " " << surface_points(2,i) << std::endl;
		// }

		std::ofstream control_file;
		control_file.open("control_" +std::to_string(d) + ".obj");

		for (unsigned int i = 0; i < patch.get_control_points() -> size(); ++i) {

			arma::vec C = patch.get_control_points() -> at(i) -> get_coordinates();
			control_file << "v " << C(0) << " " << C(1) << " " << C(2) << std::endl;
		}

		patch.elevate_n();
		std::cout << d << std::endl;
	}




	return 0;
}