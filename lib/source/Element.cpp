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


std::shared_ptr<arma::vec> Element::get_dX_bar_ptr() const{
	return this -> dX_bar_ptr;
}

void Element::initialize_dX_bar(){
	unsigned int N = this -> control_points.size();
	this -> dX_bar_ptr = std::make_shared<arma::vec>(arma::zeros<arma::vec>(3 * N));
}


std::shared_ptr<arma::mat> Element::get_info_mat_ptr() const{
	return this -> info_mat_ptr;
}


void Element::update() {
	this -> compute_normal();
	this -> compute_area();
	this -> compute_center();
}


void Element::initialize_info_mat(){
	unsigned int N = this -> control_points.size();
	this -> info_mat_ptr = std::make_shared<arma::mat>(arma::zeros<arma::mat>(3 * N,3 * N ));
	*this -> info_mat_ptr += arma::eye<arma::mat>(3 * N,3 * N);
}
