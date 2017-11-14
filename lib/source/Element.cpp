#include "Element.hpp"


Element::Element(std::vector<std::shared_ptr<ControlPoint > > control_points) {
	this -> control_points = control_points;
}


std::vector<std::shared_ptr<ControlPoint > >  * Element::get_control_points() {
	return (&this -> control_points);
}


double Element::get_area() const {
	return this -> area;
}


arma::vec Element::get_center()  const{
	return this -> center;

}

arma::vec  Element::get_normal() const  {
	return  this -> normal;
}


void Element::update() {
	this -> compute_normal();
	this -> compute_area();
	this -> compute_center();
}
