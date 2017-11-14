#include "Bezier.hpp"
#include "Facet.hpp"
#include "PC.hpp"
#include "ControlPoint.hpp"
#include <chrono>
int main(){

	arma::arma_rng::set_seed(0);

	std::vector<std::shared_ptr<ControlPoint> > vertices;

	unsigned int n = 3;
	unsigned int N = 100000;
	unsigned int N_points = (n + 1) * (n + 2) / 2;


	for (unsigned int i = 0; i < N_points; ++i){
		std::shared_ptr<ControlPoint> v = std::make_shared<ControlPoint>(ControlPoint());
		v -> set_coordinates(arma::randu<arma::vec>(3));
		vertices.push_back(v);
	}

	Bezier patch(vertices);
	arma::mat surface_points(3,N);

	auto start = std::chrono::system_clock::now();
	for (unsigned int i = 0; i < N; ++i){
		arma::vec chi = arma::randu(2);
		double u = chi(0);
		double v = (1 - u) * chi(1);
		surface_points.col(i) = patch.evaluate(u,v);
	}

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

	std::ofstream shape_file;
	shape_file.open("../patch.obj");
	for (unsigned int i = 0; i < N; ++i) {
		shape_file << "v " << surface_points(0,i) << " " << surface_points(1,i) << " " << surface_points(2,i) << std::endl;
	}

	std::ofstream control_file;
	control_file.open("../control.obj");
	for (unsigned int i = 0; i < N_points; ++i) {

		arma::vec C = vertices[i] -> get_coordinates();
		control_file << "v " << C(0) << " " << C(1) << " " << C(2) << std::endl;
	}




	return 0;
}