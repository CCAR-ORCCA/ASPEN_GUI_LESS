#include "../include/Facet.hpp"
#include <memory>

Facet::Facet(std::vector<std::shared_ptr<ControlPoint > > control_points) : Element(control_points){
	
	this -> update();

}


std::set < Element *  > Facet::get_neighbors(bool all_neighbors) const {

	std::set< Element * > neighbors;

	std::shared_ptr<ControlPoint> V0 = this -> control_points[0];
	std::shared_ptr<ControlPoint> V1 = this -> control_points[1];
	std::shared_ptr<ControlPoint> V2 = this -> control_points[2];


	if (all_neighbors == true) {
		// Returns all facets sharing control_points with $this

		for (auto facet_it = V0 -> get_owning_elements().begin();
		        facet_it != V0 -> get_owning_elements().end(); ++facet_it) {
			neighbors.insert(*facet_it);
		}

		for (auto facet_it = V1 -> get_owning_elements().begin();
		        facet_it != V1 -> get_owning_elements().end(); ++facet_it) {
			neighbors.insert(*facet_it);
		}

		for (auto facet_it = V2 -> get_owning_elements().begin();
		        facet_it != V2 -> get_owning_elements().end(); ++facet_it) {
			neighbors.insert(*facet_it);
		}

	}

	else {
		// Returns facets sharing edges with $this
		std::set<Element * > neighboring_facets_e0 = V0 -> common_facets(V1);
		std::set<Element * > neighboring_facets_e1 = V1 -> common_facets(V2);
		std::set<Element * > neighboring_facets_e2 = V2 -> common_facets(V0);

		for (auto it = neighboring_facets_e0.begin(); it != neighboring_facets_e0.end(); ++it) {
			neighbors.insert(*it);
		}

		for (auto it = neighboring_facets_e1.begin(); it != neighboring_facets_e1.end(); ++it) {
			neighbors.insert(*it);
		}

		for (auto it = neighboring_facets_e2.begin(); it != neighboring_facets_e2.end(); ++it) {
			neighbors.insert(*it);
		}
	}
	return neighbors;

}


void Facet::compute_normal() {

	arma::vec P0 = this -> control_points[0] -> get_coordinates();
	arma::vec P1 = this -> control_points[1] -> get_coordinates();
	arma::vec P2 = this -> control_points[2] -> get_coordinates();
	this -> normal = arma::normalise(arma::cross(P1 - P0, P2 - P0));
}


std::shared_ptr<ControlPoint> Facet::vertex_not_on_edge(std::shared_ptr<ControlPoint> v0,
        std::shared_ptr<ControlPoint>v1) const {
	for (unsigned int i = 0; i < this -> control_points . size(); ++i) {

		if (this -> control_points [i]!= v0 && this -> control_points [i] != v1 ) {
			return this -> control_points [i];
		}
	}
	return nullptr;

}




void Facet::compute_center() {

	arma::vec center = arma::zeros(3);

	for (unsigned int vertex_index = 0; vertex_index < this -> control_points . size(); ++vertex_index) {

		center += this -> control_points [vertex_index] -> get_coordinates();

	}

	this -> center = center / this -> control_points . size();

}

void Facet::compute_area() {
	arma::vec P0 = this -> control_points[0] -> get_coordinates() ;
	arma::vec P1 = this -> control_points[1] -> get_coordinates() ;
	arma::vec P2 = this -> control_points[2] -> get_coordinates() ;
	this -> area = arma::norm( arma::cross(P1 - P0, P2 - P0)) / 2;
}
