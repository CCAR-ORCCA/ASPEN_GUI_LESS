#include "../include/ControlPoint.hpp"

void ControlPoint::set_coordinates(arma::vec coordinates) {
	this -> coordinates = coordinates;
}


void ControlPoint::add_ownership(Element *  el) {

	this -> owning_elements.insert(el);


}


void ControlPoint::remove_ownership(Element *  el) {

	this -> owning_elements.erase(el);

}

void ControlPoint::reset_ownership(){
	this -> owning_elements.clear();
}


std::set< Element *  > ControlPoint::get_owning_elements() const {
	return this -> owning_elements;
}



std::set< Element * >  ControlPoint::common_facets(std::shared_ptr<ControlPoint> vertex) const {

	std::set< Element *> common_facets;

	for (auto it = this -> owning_elements.begin();
	        it != this -> owning_elements.end(); ++it) {

		if (vertex -> is_owned_by(*it)) {
			common_facets.insert(*it);
		}

	}

	return common_facets;

}




bool ControlPoint::is_owned_by(Element * facet) const {
	if (this -> owning_elements.find(facet) == this -> owning_elements.end()) {
		return false;

	}
	else {
		return true;

	}
}


arma::vec ControlPoint::get_coordinates()  const {
	return this -> coordinates;
}

unsigned int ControlPoint::get_number_of_owning_elements() const {
	return this -> owning_elements.size();
}
