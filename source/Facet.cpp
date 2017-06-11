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

}


void Facet::set_split_counter(unsigned int split_counter) {
	this -> split_counter = split_counter;
}



unsigned int Facet::get_split_count() const {
	return this -> split_counter;
}

void Facet::update() {
	this -> compute_normal();
	this -> compute_area();
	this -> compute_facet_center();
	this -> compute_facet_dyad();
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


			// The shortest edge is made even shorter
			if (arma::norm(*V0 -> get_coordinates() - *V2 -> get_coordinates()) <
			        arma::norm(*V1 -> get_coordinates() - *V2 -> get_coordinates())) {

				*V2 -> get_coordinates() = *V0 -> get_coordinates() + 0.1 * ( *V2 -> get_coordinates() - *V0 -> get_coordinates());


			}
			else {
				*V2 -> get_coordinates() = *V1 -> get_coordinates() + 0.1 * ( *V2 -> get_coordinates() - *V1 -> get_coordinates());

			}
			return false;

		}

	}

	return true;



}

bool Facet::has_good_surface_quality(double angle) const {

	std::shared_ptr<Vertex> V0  = this -> vertices -> at(0);
	std::shared_ptr<Vertex> V1  = this -> vertices -> at(1);
	std::shared_ptr<Vertex> V2  = this -> vertices -> at(2);

	arma::vec * P0  = V0 -> get_coordinates();
	arma::vec * P1  = V1 -> get_coordinates();
	arma::vec * P2  = V2 -> get_coordinates();

	arma::vec sin_angles = arma::vec(3);
	sin_angles(0) = arma::norm(arma::cross(*P1 - *P0, *P2 - *P0) / ( arma::norm(*P1 - *P0) * arma::norm(*P2 - *P0) ));
	sin_angles(1) = arma::norm(arma::cross(*P2 - *P1, *P0 - *P1) / ( arma::norm(*P2 - *P1) * arma::norm(*P0 - *P1) ));
	sin_angles(2) = arma::norm(arma::cross(*P0 - *P2, *P1 - *P2) / ( arma::norm(*P0 - *P2) * arma::norm(*P1 - *P2) ));

	if (sin_angles.min() < std::sin(angle)) {
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

		for (unsigned int i = 0; i < V0 -> get_owning_facets().size(); ++i) {
			neighbors.insert(V0 -> get_owning_facets()[i]);
		}

		for (unsigned int i = 0; i < V1 -> get_owning_facets().size(); ++i) {
			neighbors.insert(V1 -> get_owning_facets()[i]);
		}

		for (unsigned int i = 0; i < V2 -> get_owning_facets().size(); ++i) {
			neighbors.insert(V2 -> get_owning_facets()[i]);
		}


	}

	else {
		// Returns facets sharing edges with $this
		std::vector<Facet *> neighboring_facets_e0 = V0 -> common_facets(V1);
		std::vector<Facet *> neighboring_facets_e1 = V1 -> common_facets(V2);
		std::vector<Facet *> neighboring_facets_e2 = V2 -> common_facets(V0);

		for (unsigned int i = 0; i <  neighboring_facets_e0.size(); ++i) {
			neighbors.insert(neighboring_facets_e0[i]);
		}

		for (unsigned int i = 0; i <  neighboring_facets_e1.size(); ++i) {
			neighbors.insert(neighboring_facets_e1[i]);
		}

		for (unsigned int i = 0; i <  neighboring_facets_e2.size(); ++i) {
			neighbors.insert(neighboring_facets_e2[i]);
		}
	}
	return neighbors;


}


void Facet::compute_normal() {

	arma::vec * P0 = this -> vertices -> at(0) -> get_coordinates();
	arma::vec * P1 = this -> vertices -> at(1) -> get_coordinates();
	arma::vec * P2 = this -> vertices -> at(2) -> get_coordinates();
	*this -> facet_normal = arma::cross(*P1 - *P0, *P2 - *P0) / arma::norm(arma::cross(*P1 - *P0, *P2 - *P0));
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