#include "../include/ControlPoint.hpp"
#include "Bezier.hpp"
#include "Facet.hpp"

void ControlPoint::set_coordinates(arma::vec coordinates) {
	this -> coordinates = coordinates;
}




void ControlPoint::add_ownership(Element *  el) {

	this -> owning_elements.insert(el);

}

void ControlPoint::set_mean_coordinates() {
	this -> mean_coordinates = this -> coordinates;
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

double * ControlPoint::get_coordinates_pointer(){
	return this -> coordinates.colptr(0);
}

arma::vec * ControlPoint::get_coordinates_pointer_arma(){
	return &this -> coordinates;

}


void ControlPoint::set_covariance(arma::mat P){
	this -> covariance = P;
}




arma::mat ControlPoint::get_covariance() const{
	return this -> covariance;
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


void ControlPoint::add_local_numbering(Element * element,const arma::uvec & local_indices){
	this -> local_numbering[element] = local_indices;
}



arma::uvec ControlPoint::get_local_numbering(Element * element) const{
	return this -> local_numbering.at(element);
}



bool ControlPoint::is_owned_by(Element * facet) const {
	if (this -> owning_elements.find(facet) == this -> owning_elements.end()) {
		return false;

	}
	else {
		return true;

	}
}


int  ControlPoint::get_global_index() const{
	return this -> global_index;
}


void ControlPoint::set_global_index(int index){
	this -> global_index = index;
}


arma::vec ControlPoint::get_coordinates()  const {
	return this -> coordinates;
}


arma::vec ControlPoint::get_mean_coordinates()  const {
	return this -> mean_coordinates;
}

unsigned int ControlPoint::get_number_of_owning_elements() const {
	return this -> owning_elements.size();
}



arma::vec ControlPoint::get_normal(bool bezier) const{

	arma::vec n = {0,0,0};

	for (auto it = this -> owning_elements.begin();
		it != this -> owning_elements.end(); ++it) {

		if (bezier){
			n += static_cast<Bezier *>((*it)) -> get_normal(1./3,1./3);

		}
		else{
			n += static_cast<Facet *>((*it)) -> get_normal();
		}

	}

	return arma::normalise(n);


}




