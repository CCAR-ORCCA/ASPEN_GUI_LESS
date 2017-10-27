#include "DynamicAnalyses.hpp"

DynamicAnalyses::DynamicAnalyses(ShapeModelTri * shape_model) {
	this -> shape_model = shape_model;
}

arma::vec DynamicAnalyses::point_mass_acceleration(arma::vec & point , double mass) const {

	arma::vec acc = - mass * arma::datum::G / arma::dot(point, point) * arma::normalise(point);
	return acc;


}



