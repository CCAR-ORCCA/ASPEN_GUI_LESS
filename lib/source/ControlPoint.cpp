#include "ControlPoint.hpp"
#include "Bezier.hpp"
#include "Facet.hpp"
#include "ShapeModel.hpp"

ControlPoint::ControlPoint(ShapeModel * owning_shape){
	this -> owning_shape = owning_shape;
}


void ControlPoint::set_point_coordinates(arma::vec::fixed<3> & coordinates) {
	this -> coordinates = coordinates;
}


void ControlPoint::add_ownership(int el) {

	this -> owning_elements.insert(el);

}

void ControlPoint::set_owning_elements(std::set< int  > & owning_elements){
	this -> owning_elements = owning_elements;
}

void ControlPoint::remove_ownership(int el) {

	this -> owning_elements.erase(el);

}

void ControlPoint::reset_ownership(){
	this -> owning_elements.clear();
}


std::set< int  > ControlPoint::get_owning_elements() const {
	return this -> owning_elements;
}



void ControlPoint::set_covariance(arma::mat P){
	this -> covariance = P;
}


arma::mat ControlPoint::get_covariance() const{
	return this -> covariance;
}



std::set< int >  ControlPoint::common_facets(std::shared_ptr<ControlPoint> vertex) const {

	std::set< int> common_facets;

	for (auto it = this -> owning_elements.begin();
		it != this -> owning_elements.end(); ++it) {

		if (vertex -> is_owned_by(*it)) {
			common_facets.insert(*it);
		}

	}

	return common_facets;

}


void ControlPoint::add_local_numbering(int element,const arma::uvec & local_indices){
	this -> local_numbering[element] = local_indices;
}



arma::uvec ControlPoint::get_local_numbering(int element) const{
	return this -> local_numbering.at(element);
}



bool ControlPoint::is_owned_by(int facet) const {
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

const arma::vec::fixed<3> & ControlPoint::get_point_coordinates()  const {
	return this -> coordinates;
}


unsigned int ControlPoint::get_number_of_owning_elements() const {
	return this -> owning_elements.size();
}



arma::vec::fixed<3> ControlPoint::get_normal_coordinates(bool bezier) const{

	arma::vec n = {0,0,0};

	for (auto it = this -> owning_elements.begin(); it != this -> owning_elements.end(); ++it) {

		if (bezier){
			n += static_cast<Bezier *>(this -> owning_shape -> get_element((*it))).get_normal_coordinates(1./3,1./3);

		}
		else{
			n += static_cast<Facet *>(this -> owning_shape -> get_element((*it))).get_normal_coordinates();
		}
	}
	return arma::normalise(n);

}




