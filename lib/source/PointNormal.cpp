#include "PointNormal.hpp"


PointNormal::PointNormal(arma::vec point,int index) {
	this -> global_index = index;
	this -> point = point;
}

PointNormal::PointNormal(arma::vec point, arma::vec normal,int index) {
	this -> point = point;
	this -> normal = normal;
	this -> global_index = index;
}



arma::vec PointNormal::get_point_coordinates() const {
	return this -> point;
}

arma::vec PointNormal::get_normal_coordinates() const {
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

PointDescriptor PointNormal::get_descriptor() const{
	return this -> descriptor;
}


const PointDescriptor * PointNormal::get_descriptor_ptr() const{
	return &this -> descriptor;
}

arma::vec PointNormal::get_descriptor_histogram() const{
	return this -> descriptor. get_histogram();
}
unsigned int PointNormal::get_histogram_size() const{
	return this -> descriptor. get_histogram_size();
}

double PointNormal::get_histogram_value(int index) const{
	return this -> descriptor. get_histogram_value(index);
}

arma::vec PointNormal::get_spfh_histogram() const{
	return this -> spfh.get_histogram();
}

double PointNormal::features_similarity_distance(std::shared_ptr<PointNormal> other_point) const{
	return this -> descriptor. distance_to_descriptor(other_point -> get_descriptor_ptr());
}


double PointNormal::features_similarity_distance(const arma::vec & histogram) const{
	PointDescriptor dummy_descriptor(histogram);

	return this -> descriptor. distance_to_descriptor(&dummy_descriptor);
}


void PointNormal::set_SPFH(SPFH spfh){
	this -> spfh = spfh;
}


SPFH * PointNormal::get_SPFH(){
	return &this -> spfh;
}


bool PointNormal::get_is_valid_feature() const{
	return this -> is_valid_feature;
}

void PointNormal::set_is_valid_feature(bool valid_feature){
	this -> is_valid_feature = valid_feature;
}


int  PointNormal::get_match() const{
	return this -> match;
}

void PointNormal::set_match(int match){
	this -> match = match;
}


void PointNormal::set_neighborhood(const std::vector<std::shared_ptr<PointNormal> > & neighborhood){
	
	this -> neighborhood.clear();

	for (auto it = neighborhood.begin(); it != neighborhood.end(); ++it){
		double distance_to_point = this -> distance(*it);
		this -> neighborhood[distance_to_point] = (*it) -> get_global_index();
	}

}


void PointNormal::set_neighborhood(const std::vector<PointNormal * > & neighborhood){
	
	this -> neighborhood.clear();

	for (auto it = neighborhood.begin(); it != neighborhood.end(); ++it){
		double distance_to_point = this -> distance(*it);
		this -> neighborhood[distance_to_point] = (*it) -> get_global_index();
	}

}


std::vector<int> PointNormal::get_neighborhood(double radius){
	std::vector<int> neighborhood;
	auto iter_neighborhood_end = this -> neighborhood.lower_bound(radius);

	for (auto it = this -> neighborhood.begin(); it != iter_neighborhood_end; ++it){
		neighborhood.push_back(it -> second);
	}
	return neighborhood;
}

int PointNormal::get_global_index() const{
	return this -> global_index;
}
void PointNormal::set_global_index (int global_index){
	this -> global_index = global_index;
}



const arma::vec & PointNormal::get_point_coordinates_ref() const{
	return this -> point;
}
