#include "FPFH.hpp"

#include <armadillo>
#include "PointNormal.hpp"

FPFH::FPFH(const std::shared_ptr<PointNormal> & query_point,const std::vector<std::shared_ptr<PointNormal> > & points) : PointDescriptor(){

	this -> histogram = query_point -> get_SPFH() -> get_histogram();

	double distance_to_closest_neighbor = query_point -> get_SPFH() -> get_distance_to_closest_neighbor();

	for (unsigned int k = 0 ; k < points. size(); ++k){
		double distance = arma::norm(query_point -> get_point() - points . at(k) -> get_point());
		if (distance>0){
			this -> histogram += distance_to_closest_neighbor * points. at(k) -> get_SPFH() -> get_histogram() / (distance * points.size());
		}
	}

	if (arma::max(this -> histogram) > 0){
		this -> histogram = this -> histogram / arma::max(this -> histogram);
	}


	this -> type = 1;
	

}

FPFH::FPFH() : PointDescriptor(){}


