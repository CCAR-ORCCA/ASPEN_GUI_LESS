#include "FPFH.hpp"

#include <armadillo>
#include "PointNormal.hpp"

FPFH::FPFH(const std::shared_ptr<PointNormal> & query_point) : PointDescriptor(){

	this -> histogram = std::vector<double>(query_point -> get_SPFH() -> get_histogram_size(),0);
	auto neighbors_exclusive = query_point -> get_SPFH() -> get_exclusive_neighbors();


	for (int i = 0; i < this -> histogram.size(); ++i){
		this -> histogram[i] += query_point -> get_SPFH() -> get_histogram_value(i);
	}

	double max_value = -1;
	for (unsigned int k = 0 ; k < neighbors_exclusive -> size(); ++k){
		double distance = arma::norm(query_point -> get_point() - neighbors_exclusive -> at(k) -> get_point());
		
		for (int i = 0; i < this -> histogram.size(); ++i){
			this -> histogram[i] += neighbors_exclusive -> at(k) -> get_SPFH() -> get_histogram_value(i) / (distance * neighbors_exclusive -> size());
			max_value = std::max(max_value,this -> histogram[i]);
		}
	}

	for (int i = 0; i < this -> histogram.size(); ++i){
		this -> histogram[i] = this -> histogram[i] / max_value;
	}

	
}

FPFH::FPFH() : PointDescriptor(){}


