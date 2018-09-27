#include "PointNormal.hpp"




PointNormal::PointNormal(){

}


PointNormal::PointNormal(arma::vec point,int index) {
	this -> global_index = index;
	this -> point = point;
}

PointNormal::PointNormal(arma::vec point, arma::vec normal,int index) {
	this -> point = point;
	this -> normal = normal;
	this -> global_index = index;
}



const arma::vec & PointNormal::get_point_coordinates() const {
	return this -> point;
}

const arma::vec & PointNormal::get_normal_coordinates() const {
	return this -> normal;
}


void PointNormal::set_normal_coordinates(arma::vec normal) {
	this -> normal = normal;
}


void PointNormal::set_point_coordinates(arma::vec point) {
	this -> point = point;
}

double PointNormal::distance(const std::shared_ptr<PointNormal> & other_point) const {
	return arma::norm(this -> point - other_point -> get_point_coordinates());
}


double PointNormal::distance(PointNormal * other_point) const {
	return arma::norm(this -> point - other_point -> get_point_coordinates());
}


void PointNormal::decrement_inclusion_counter() {
	this -> inclusion_counter = this -> inclusion_counter - 1;
}

int PointNormal::get_inclusion_counter() const {
	return this -> inclusion_counter ;
}

void PointNormal::set_descriptor(const PointDescriptor & descriptor) {
	this -> descriptor = descriptor;
}


int PointNormal::get_global_index() const{
	return this -> global_index;
}
void PointNormal::set_global_index (int global_index){
	this -> global_index = global_index;
}


