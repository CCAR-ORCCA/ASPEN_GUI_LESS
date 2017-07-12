#include "PointNormal.hpp"


PointNormal::PointNormal(arma::vec point) {
	this -> point = point;
}

PointNormal::PointNormal(arma::vec point, int inclusion_counter) {
	this -> point = point;
	this -> inclusion_counter = inclusion_counter;
}


arma::vec * PointNormal::get_point() {
	return &this -> point;
}

arma::vec * PointNormal::get_normal() {
	return &this -> normal;
}


void PointNormal::set_normal(arma::vec normal) {
	this -> normal = normal;
}

double PointNormal::distance(std::shared_ptr<PointNormal> other_point) const {
	return arma::norm(this -> point - *other_point -> get_point());
}

void PointNormal::decrement_inclusion_counter() {
	this -> inclusion_counter = this -> inclusion_counter - 1;
}


int PointNormal::get_inclusion_counter() const {
	return this -> inclusion_counter ;
}
