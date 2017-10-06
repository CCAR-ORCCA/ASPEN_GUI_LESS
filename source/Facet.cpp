#include "../include/Facet.hpp"
#include <memory>

Facet::Facet(std::vector<std::shared_ptr<ControlPoint > > vertices) {
	this -> vertices = vertices;

	for (unsigned int vertex_index = 0; vertex_index < this -> vertices . size(); ++vertex_index) {
		this -> vertices[vertex_index]-> add_facet_ownership(this);
	}

	// Computing surface area
	this -> compute_area();

	// Computing normals
	this -> compute_normal();

	// Computing facet center
	this -> compute_facet_center();

}

void Facet::set_split_counter(unsigned int split_counter) {
	this -> split_counter = split_counter;
}

unsigned int Facet::get_hit_count() const {
	return this -> hit_count;
}


unsigned int Facet::get_split_count() const {
	return this -> split_counter;
}

void Facet::update() {
	this -> compute_normal();
	this -> compute_area();
	this -> compute_facet_center();
	
}

/**
Increases the hit counter by one
*/
void Facet::increase_hit_count() {
	this -> hit_count = 1 + this -> hit_count;
}



std::set < Facet * > Facet::get_neighbors(bool all_neighbors) const {

	std::set<Facet *> neighbors;

	std::shared_ptr<ControlPoint> V0 = this -> vertices[0];
	std::shared_ptr<ControlPoint> V1 = this -> vertices[1];
	std::shared_ptr<ControlPoint> V2 = this -> vertices[2];


	if (all_neighbors == true) {
		// Returns all facets sharing vertices with $this

		for (auto facet_it = V0 -> get_owning_facets().begin();
		        facet_it != V0 -> get_owning_facets().end(); ++facet_it) {
			neighbors.insert(*facet_it);
		}

		for (auto facet_it = V1 -> get_owning_facets().begin();
		        facet_it != V1 -> get_owning_facets().end(); ++facet_it) {
			neighbors.insert(*facet_it);
		}

		for (auto facet_it = V2 -> get_owning_facets().begin();
		        facet_it != V2 -> get_owning_facets().end(); ++facet_it) {
			neighbors.insert(*facet_it);
		}


	}

	else {
		// Returns facets sharing edges with $this
		std::set<Facet *> neighboring_facets_e0 = V0 -> common_facets(V1);
		std::set<Facet *> neighboring_facets_e1 = V1 -> common_facets(V2);
		std::set<Facet *> neighboring_facets_e2 = V2 -> common_facets(V0);

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

	arma::vec * P0 = this -> vertices[0] -> get_coordinates();
	arma::vec * P1 = this -> vertices[1] -> get_coordinates();
	arma::vec * P2 = this -> vertices[2] -> get_coordinates();
	this -> facet_normal = arma::normalise(arma::cross(*P1 - *P0, *P2 - *P0));
}


arma::vec * Facet::get_facet_normal()  {
	return (&this -> facet_normal);
}


std::shared_ptr<ControlPoint> Facet::vertex_not_on_edge(std::shared_ptr<ControlPoint> v0,
        std::shared_ptr<ControlPoint>v1) const {
	for (unsigned int i = 0; i < this -> vertices . size(); ++i) {

		if (this -> vertices [i]!= v0 && this -> vertices [i] != v1 ) {
			return this -> vertices [i];
		}
	}
	return nullptr;

}



arma::vec * Facet::get_facet_center()  {
	return (&this -> facet_center);

}

void Facet::compute_facet_center() {

	arma::vec facet_center = arma::zeros(3);

	for (unsigned int vertex_index = 0; vertex_index < this -> vertices . size(); ++vertex_index) {

		facet_center += *this -> vertices [vertex_index] -> get_coordinates();

	}

	this -> facet_center = facet_center / this -> vertices . size();

}

std::vector<std::shared_ptr<ControlPoint > >  * Facet::get_vertices() {
	return (&this -> vertices);
}


void Facet::compute_area() {
	arma::vec * P0 = this -> vertices[0] -> get_coordinates() ;
	arma::vec * P1 = this -> vertices[1] -> get_coordinates() ;
	arma::vec * P2 = this -> vertices[2] -> get_coordinates() ;
	this -> area = arma::norm( arma::cross(*P1 - *P0, *P2 - *P0)) / 2;
}

double Facet::get_area() const {
	return this -> area;
}