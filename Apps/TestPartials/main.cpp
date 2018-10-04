#include "Bezier.hpp"
#include "ControlPoint.hpp"
#include "Ray.hpp"
#include "ShapeModelBezier.hpp"



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

	double u = 0.2;
	double v = 0.3;

	arma::vec n_old = nominal_patch.get_normal_coordinates(u,v);

	auto indices = nominal_patch.get_local_indices(v5);
	arma::mat dndc = nominal_patch.partial_n_partial_Ck(u, v,std::get<0>(indices),std::get<1>(indices),2);

	arma::vec dC = {0.1, 0.01, -1};

	std::cout << "Old normal: \n" << n_old.t();
	v5 -> set_coordinates(v5 -> get_coordinates() + dC);

	arma::vec n = nominal_patch.get_normal_coordinates(u,v);
	std::cout << "New normal: \n" << n.t();

	std::cout << "Predicted new normal : " << std::endl;
	std::cout << (n_old + dndc * dC).t()<< std::endl;



	


	return 0;
}