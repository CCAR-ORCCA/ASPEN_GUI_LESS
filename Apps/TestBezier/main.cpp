#include "Bezier.hpp"
#include "Facet.hpp"
#include "PC.hpp"
#include "ControlPoint.hpp"

#include <chrono>


int main(){

	arma::arma_rng::set_seed(0);

	std::vector<std::shared_ptr<ControlPoint> > vertices;


	std::shared_ptr<ControlPoint> v0 = std::make_shared<ControlPoint>(ControlPoint());
	arma::vec nominal_coords0 = {0,0,0};
	v0 -> set_coordinates(nominal_coords0);
	vertices.push_back(v0);

	std::shared_ptr<ControlPoint> v1 = std::make_shared<ControlPoint>(ControlPoint());
	arma::vec nominal_coords1 = {1,0,0.5};
	v1 -> set_coordinates(nominal_coords1);
	vertices.push_back(v1);

	std::shared_ptr<ControlPoint> v2 = std::make_shared<ControlPoint>(ControlPoint());
	arma::vec nominal_coords2 = {0.0,1,0.3};
	v2 -> set_coordinates(nominal_coords2);
	vertices.push_back(v2);

	std::shared_ptr<ControlPoint> v3 = std::make_shared<ControlPoint>(ControlPoint());
	arma::vec nominal_coords3 = {2,0,0};
	v3 -> set_coordinates(nominal_coords3);
	vertices.push_back(v3);


	std::shared_ptr<ControlPoint> v4 = std::make_shared<ControlPoint>(ControlPoint());
	arma::vec nominal_coords4 = {1.5,0.5,1};
	v4 -> set_coordinates(nominal_coords4);
	vertices.push_back(v4);

	std::shared_ptr<ControlPoint> v5 = std::make_shared<ControlPoint>(ControlPoint());
	arma::vec nominal_coords5 = {0.5,2,0};
	v5 -> set_coordinates(nominal_coords5);
	vertices.push_back(v5);
	
	Bezier nominal_patch(vertices);

	Bezier patch(vertices);
	arma::vec chi = {0,0};
	arma::vec P_tilde = patch.evaluate(0.4,0.2) + patch.get_normal(0.4,0.2);
	arma::vec P_bar = patch.evaluate(chi(0),chi(1));
	arma::vec P_bar_old = P_bar;

	unsigned int N_iter = 30;

	arma::mat H = arma::zeros<arma::mat>(2,2);
	arma::vec Y = arma::zeros<arma::vec>(2);
	arma::vec dchi;
	arma::mat dbezier_dchi;


	auto start = std::chrono::system_clock::now();

	for (unsigned int i = 0; i < N_iter; ++i){


		dbezier_dchi = patch.partial_bezier(chi(0),chi(1));

		H.row(0) = dbezier_dchi.col(0).t() * dbezier_dchi - (P_tilde - P_bar).t() * patch.partial_bezier_du(chi(0),chi(1));
		H.row(1) = dbezier_dchi.col(1).t() * dbezier_dchi - (P_tilde - P_bar).t() * patch.partial_bezier_dv(chi(0),chi(1));

		Y(0) =  arma::dot(dbezier_dchi.col(0),P_tilde - P_bar);
		Y(1) =  arma::dot(dbezier_dchi.col(1),P_tilde - P_bar);

		dchi = arma::solve(H,Y);

		chi += dchi;
		P_bar = patch.evaluate(chi(0),chi(1));


		
		// std::ofstream P_bar_file;
		// P_bar_file.open("P_bar_" +std::to_string(i) + ".obj");
		// P_bar_file << "v " << P_bar(0) << " " << P_bar(1) << " " << P_bar(2) << std::endl;
		double error = arma::norm(arma::cross(patch.get_normal(chi(0),chi(1)),P_bar - P_tilde));
		if (error < 1e-6){
			std::cout << "Converged in " << i + 1 << " iterations\n";
			break;
		}

	}

    auto end = std::chrono::system_clock::now();
 
    std::chrono::duration<double> elapsed_seconds = end-start;
 
    std::cout << " Done  in " << elapsed_seconds.count() << " s\n";





	std::ofstream control_mesh_file;
	control_mesh_file.open("control_.obj");
	for (unsigned int i = 0; i < vertices.size(); ++i) {
		control_mesh_file << "v " << vertices[i] -> get_coordinates()(0) << " " << vertices[i] -> get_coordinates()(1) << " " << vertices[i] -> get_coordinates()(2) << std::endl;
	}

	std::ofstream P_tilde_file;
	P_tilde_file.open("P_tilde.obj");
	P_tilde_file << "v " << P_tilde(0) << " " << P_tilde(1) << " " << P_tilde(2) << std::endl;
	
	std::ofstream P_bar_old_file;
	P_bar_old_file.open("P_bar_old.obj");
	P_bar_old_file << "v " << P_bar_old(0) << " " << P_bar_old(1) << " " << P_bar_old(2) << std::endl;
	


	unsigned int D = 10;
	for (unsigned int i = 1; i < D + 1; ++i){
		std::cout << i << std::endl;
		patch.elevate_n();
	}
	patch.save("bezier_" + std::to_string(D)+ ".obj");

	





	return 0;
}