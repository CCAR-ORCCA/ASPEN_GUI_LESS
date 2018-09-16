#include "PointNormal.hpp"


PointNormal::PointNormal(arma::vec point) {
	this -> point = point;
}

PointNormal::PointNormal(arma::vec point, arma::vec normal) {
	this -> point = point;
	this -> normal = normal;
}

PointNormal::PointNormal(arma::vec point, int inclusion_counter) {
	this -> point = point;
	this -> inclusion_counter = inclusion_counter;
}


arma::vec PointNormal::get_point() const {
	return this -> point;
}

arma::vec PointNormal::get_normal() const {
	return this -> normal;
}


void PointNormal::set_normal(arma::vec normal) {
	this -> normal = normal;
}


void PointNormal::set_point(arma::vec point) {
	this -> point = point;
}

double PointNormal::distance(std::shared_ptr<PointNormal> other_point) const {
	return arma::norm(this -> point - other_point -> get_point());
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

PointDescriptor PointNormal::get_descriptor() const{
	return this -> descriptor;
}


PointDescriptor * PointNormal::get_descriptor_ptr(){
	return &this -> descriptor;
}

std::vector<double> PointNormal::get_descriptor_histogram() const{
	return this -> descriptor. get_histogram();
}
unsigned int PointNormal::get_histogram_size() const{
	return this -> descriptor. get_histogram_size();
}

double PointNormal::get_histogram_value(int index) const{
	return this -> descriptor. get_histogram_value(index);
}

double PointNormal::descriptor_distance(std::shared_ptr<PointNormal> other_point) const{
	return this -> descriptor. distance_to(other_point -> get_descriptor());
}

void PointNormal::set_SPFH(SPFH spfh){
	this -> spfh = spfh;
}


SPFH * PointNormal::get_SPFH(){
	return &this -> spfh;
}






