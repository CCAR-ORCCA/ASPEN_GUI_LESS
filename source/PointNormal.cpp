#include "PointNormal.hpp"


PointNormal::PointNormal(arma::vec point) {
	this -> point = point;
}

arma::vec * PointNormal::get_point() {
	return &this -> point;
}

double PointNormal::distance(std::shared_ptr<PointNormal> other_point) const{
	return arma::norm(this -> point - *other_point -> get_point());
}
