#include "../include/Facet.hpp"
#include <memory>

Facet::Facet(std::shared_ptr< std::vector<std::shared_ptr<Vertex > > >   vertices) {
	this -> vertices = vertices;

	for (unsigned int vertex_index = 0; vertex_index < this -> vertices -> size(); ++vertex_index) {
		this -> vertices -> at(vertex_index) -> add_facet_ownership(this);
	}

	// Allocating memory for the facet normal
	this -> facet_normal = std::make_shared<arma::vec>(arma::zeros(3));

	// Allocating memory for the facet dyad
	this -> facet_dyad = std::make_shared<arma::mat>(arma::zeros(3, 3));

	// Allocating memory for the facet center
	this -> facet_center = std::make_shared<arma::vec>(arma::zeros(3));;


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

void Facet::update(bool compute_dyad) {
	this -> compute_normal();
	this -> compute_area();
	this -> compute_facet_center();
	if (compute_dyad) {
		this -> compute_facet_dyad();
	}
}

/**
Increases the hit counter by one
*/
void Facet::increase_hit_count() {
	this -> hit_count = 1 + this -> hit_count;
}



bool Facet::has_good_edge_quality(double angle) {

	std::set < Facet * > neighbors = this -> get_neighbors(false);

	arma::vec * n = this -> facet_normal.get();

	for (auto & neighbor : neighbors) {

		if (arma::dot(*n, *neighbor -> get_facet_normal()) < - std::cos(angle)) {

			Vertex * V0 = nullptr;
			Vertex * V1 = nullptr;
			Vertex * V2 = nullptr;


			// The two vertices of $this lying on the edge are found
			for (unsigned int vertex_index = 0; vertex_index < 3; ++vertex_index) {

				if (this -> vertices -> at(vertex_index) -> is_owned_by(neighbor)) {
					if (V0 == nullptr) {
						V0 = this -> vertices -> at(vertex_index).get();
					}
					else {
						V1 = this -> vertices -> at(vertex_index).get();

					}
				}

			}


			// The other vertex in neighbor is found
			for (unsigned int vertex_index = 0; vertex_index < 3; ++vertex_index) {

				if (neighbor -> get_vertices() -> at(vertex_index).get() != V0 &&
				        neighbor -> get_vertices() -> at(vertex_index).get() != V1 )
				{
					V2 = neighbor -> get_vertices() -> at(vertex_index).get();
					break;
				}

			}


			arma::vec dir = arma::normalise(*n + *neighbor -> get_facet_normal());
			arma::vec delta_V0 = arma::dot(*V2 -> get_coordinates() - *V0 -> get_coordinates(),
			                               dir) * dir;
			arma::vec delta_V1 = arma::dot(*V2 -> get_coordinates() - *V1 -> get_coordinates(),
			                               dir) * dir;

			arma::vec delta = (delta_V0 + delta_V1) / 2;



			*V0 -> get_coordinates() = *V0 -> get_coordinates() + delta;
			*V1 -> get_coordinates() = *V1 -> get_coordinates() + delta;


			return false;

		}

	}

	return true;



}

bool Facet::has_good_surface_quality(double angle) const {


	if (this -> vertices -> size() != 3) {
		throw (std::runtime_error("this facet has " + std::to_string(this -> vertices -> size()) + " vertices"));
	}

	std::shared_ptr<Vertex> V0  = this -> vertices -> at(0);
	std::shared_ptr<Vertex> V1  = this -> vertices -> at(1);
	std::shared_ptr<Vertex> V2  = this -> vertices -> at(2);

	arma::vec * P0  = V0 -> get_coordinates();
	arma::vec * P1  = V1 -> get_coordinates();
	arma::vec * P2  = V2 -> get_coordinates();

	arma::vec angles = arma::vec(3);
	angles(0) = std::acos(arma::dot(arma::normalise(*P1 - *P0), arma::normalise( *P2 - *P0)));
	angles(1) = std::acos(arma::dot(arma::normalise(*P2 - *P1), arma::normalise( *P0 - *P1)));
	angles(2) = std::acos(arma::dot(arma::normalise(*P0 - *P2), arma::normalise( *P1 - *P2)));


	if (angles.min() < angle) {
		return false;
	}
	else {
		return true;
	}



}



std::set < Facet * > Facet::get_neighbors(bool all_neighbors) const {

	std::set<Facet *> neighbors;

	std::shared_ptr<Vertex> V0 = this -> vertices -> at(0);
	std::shared_ptr<Vertex> V1 = this -> vertices -> at(1);
	std::shared_ptr<Vertex> V2 = this -> vertices -> at(2);


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

	arma::vec * P0 = this -> vertices -> at(0) -> get_coordinates();
	arma::vec * P1 = this -> vertices -> at(1) -> get_coordinates();
	arma::vec * P2 = this -> vertices -> at(2) -> get_coordinates();
	*this -> facet_normal = arma::normalise(arma::cross(*P1 - *P0, *P2 - *P0));
}

void Facet::compute_facet_dyad() {

	*this -> facet_dyad = *this -> facet_normal * (*this -> facet_normal).t();
}

arma::vec * Facet::get_facet_normal()  {
	return this -> facet_normal.get();
}


arma::mat * Facet::get_facet_dyad()  {
	return this -> facet_dyad.get();
}

std::shared_ptr<Vertex> Facet::vertex_not_on_edge(std::shared_ptr<Vertex> v0,
        std::shared_ptr<Vertex>v1) const {
	for (unsigned int i = 0; i < this -> vertices -> size(); ++i) {

		if (this -> vertices -> at(i) != v0 && this -> vertices -> at(i) != v1 ) {
			return this -> vertices -> at(i);
		}
	}
	return nullptr;

}

void Facet::add_edge(Edge * edge) {
	this -> facet_edges.insert(edge);
}

void Facet::remove_edge(Edge * edge) {
	// Should throw exception if edge is not found
	this -> facet_edges.erase(this -> facet_edges.find(edge));
}


arma::vec * Facet::get_facet_center()  {
	return this -> facet_center.get();

}

void Facet::compute_facet_center() {

	arma::vec facet_center = arma::zeros(3);

	for (unsigned int vertex_index = 0; vertex_index < this -> vertices -> size(); ++vertex_index) {

		facet_center += *this -> vertices -> at(vertex_index) -> get_coordinates();

	}

	*this -> facet_center = facet_center / this -> vertices -> size();

}

std::vector<std::shared_ptr<Vertex > >  * Facet::get_vertices() {
	return this -> vertices.get();
}


void Facet::compute_area() {
	arma::vec * P0 = this -> vertices -> at(0) -> get_coordinates() ;
	arma::vec * P1 = this -> vertices -> at(1) -> get_coordinates() ;
	arma::vec * P2 = this -> vertices -> at(2) -> get_coordinates() ;
	this -> area = arma::norm( arma::cross(*P1 - *P0, *P2 - *P0)) / 2;
}

double Facet::get_area() const {
	return this -> area;
}