#include "FPFH.hpp"

#include <armadillo>
#include "PointNormal.hpp"

FPFH::FPFH(const std::shared_ptr<PointNormal> & query_point) : PointDescriptor(){

	auto neighbors_exclusive = query_point -> get_SPFH() -> get_exclusive_neighbors();

	this -> histogram = query_point -> get_SPFH() -> get_histogram();

	if (arma::max(this -> histogram) > 0){
		this -> histogram = this -> histogram / arma::max(this -> histogram);
	}

	for (unsigned int k = 0 ; k < neighbors_exclusive -> size(); ++k){
		double distance = arma::norm(query_point -> get_point() - neighbors_exclusive -> at(k) -> get_point());
		this -> histogram += neighbors_exclusive -> at(k) -> get_SPFH() -> get_histogram() / (distance * neighbors_exclusive -> size());
	}


	if (arma::max(this -> histogram) > 0){
		this -> histogram = this -> histogram / arma::max(this -> histogram);
	}

	this -> type = 1;
	

}

FPFH::FPFH() : PointDescriptor(){}


