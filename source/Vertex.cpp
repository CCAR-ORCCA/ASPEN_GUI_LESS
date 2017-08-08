#include "../include/Vertex.hpp"

void Vertex::set_coordinates(std::shared_ptr<arma::vec> coordinates) {
	this -> coordinates = coordinates;
}


void Vertex::add_facet_ownership(Facet * facet) {

	this -> owning_facets.insert(facet);


}


void Vertex::remove_facet_ownership(Facet * facet) {

	this -> owning_facets.erase(facet);

}


std::set<Facet * > Vertex::get_owning_facets() const {
	return this -> owning_facets;
}



std::set<Facet *>  Vertex::common_facets(std::shared_ptr<Vertex> vertex) const {

	std::set<Facet *> common_facets;

	for (auto it = this -> owning_facets.begin();
	        it != this -> owning_facets.end(); ++it) {

		if (vertex -> is_owned_by(*it)) {
			common_facets.insert(*it);
		}

	}

	return common_facets;

}




bool Vertex::is_owned_by(Facet * facet) const {
	if (this -> owning_facets.find(facet) == this -> owning_facets.end()) {
		return false;

	}
	else {
		return true;

	}
}


arma::vec * Vertex::get_coordinates()  {
	return this -> coordinates.get();
}

unsigned int Vertex::get_number_of_owning_facets() const {
	return this -> owning_facets.size();
}
