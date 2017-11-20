#include "Bezier.hpp"
#include "Facet.hpp"
#include "PC.hpp"
#include "ControlPoint.hpp"
#include "Ray.hpp"


#include <chrono>


int main(){

	arma::arma_rng::set_seed(0);

	std::vector<std::shared_ptr<ControlPoint> > vertices;

	unsigned int n = 2;
	unsigned int N = 100000;
	unsigned int N_points = (n + 1) * (n + 2) / 2;

	std::shared_ptr<ControlPoint> v0 = std::make_shared<ControlPoint>(ControlPoint());
	arma::vec coords0 = {0,0,0};
	v0 -> set_coordinates(coords0);
	vertices.push_back(v0);

	std::shared_ptr<ControlPoint> v1 = std::make_shared<ControlPoint>(ControlPoint());
	arma::vec coords1 = {1,0,0.5};
	v1 -> set_coordinates(coords1);
	vertices.push_back(v1);

	std::shared_ptr<ControlPoint> v2 = std::make_shared<ControlPoint>(ControlPoint());
	arma::vec coords2 = {0.0,1,0.3};
	v2 -> set_coordinates(coords2);
	vertices.push_back(v2);

	std::shared_ptr<ControlPoint> v3 = std::make_shared<ControlPoint>(ControlPoint());
	arma::vec coords3 = {2,0,0};
	v3 -> set_coordinates(coords3);
	vertices.push_back(v3);


	std::shared_ptr<ControlPoint> v4 = std::make_shared<ControlPoint>(ControlPoint());
	arma::vec coords4 = {1.5,0.5,1};
	v4 -> set_coordinates(coords4);
	vertices.push_back(v4);

	std::shared_ptr<ControlPoint> v5 = std::make_shared<ControlPoint>(ControlPoint());
	arma::vec coords5 = {0.5,2,0};
	v5 -> set_coordinates(coords5);
	vertices.push_back(v5);
	
	Bezier patch(vertices);

	int N_rays = 100000;
	std::vector<Ray> rays;
	for (int i = 0; i < N_rays; ++i){
		arma::vec pos = {0,0,5};
		arma::vec dir = patch.get_center() + 0.1 * arma::randn(3) - pos;
		dir = arma::normalise(dir);
		Ray ray(pos,dir);
		rays.push_back(ray);
	}


	auto start = std::chrono::system_clock::now();
	
	#pragma omp parallel for 
	for (unsigned int i = 0; i < rays.size(); ++ i){
		rays[i].single_patch_ray_casting(&patch);
	}


    auto end = std::chrono::system_clock::now();
 
    std::chrono::duration<double> elapsed_seconds = end-start;
 
    std::cout << " Done ray tracing in " << elapsed_seconds.count() << " s\n";


    std::ofstream ray_traced_file;
	ray_traced_file.open("ray_traced_" +std::to_string(n) + ".obj");
	arma::vec impact(3);
	for (unsigned int i = 0; i < N_rays; ++i) {

		if (rays[i].get_true_range() > 1e5){
			continue;
		}

		impact = rays[i].get_impact_point();
		ray_traced_file << "v " << impact(0) << " " << impact(1) << " " << impact(2) << std::endl;
	}



	return 0;
}