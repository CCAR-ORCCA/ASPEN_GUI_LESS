#ifndef HEADER_POINTNORMAL
#define HEADER_POINTNORMAL

#include <armadillo>
#include <memory>
#include "PointDescriptor.hpp"
#include "SPFH.hpp"

class PointNormal {

public:	

	PointNormal();
	PointNormal(arma::vec point,int index = 0);
	PointNormal(arma::vec point, arma::vec normal,int index = 0);


	double distance(const std::shared_ptr<PointNormal> & other_point) const;
	double distance(PointNormal * other_point) const ;

	const arma::vec & get_point_coordinates() const;
	const arma::vec & get_normal_coordinates() const;

	void set_normal_coordinates(arma::vec normal) ;
	void set_point_coordinates(arma::vec point) ;
	void set_descriptor(const PointDescriptor & descriptor) ;
	
	PointDescriptor get_descriptor() const;
	const PointDescriptor * get_descriptor_ptr() const;

	arma::vec get_descriptor_histogram() const;
	arma::vec get_spfh_histogram() const;

	unsigned int get_histogram_size() const;
	double get_histogram_value(int index) const;

	void decrement_inclusion_counter();

	int get_inclusion_counter() const;

	double features_similarity_distance(std::shared_ptr<PointNormal> other_point) const;
	double features_similarity_distance(const arma::vec & histogram) const;


	void set_SPFH(SPFH spfh);

	SPFH * get_SPFH();

	bool get_is_valid_feature() const;
	void set_is_valid_feature(bool valid_feature);

	bool get_is_matched() const;
	void set_is_matched(bool value);


	int get_match() const;
	void set_match(int match);

	void set_neighborhood(const std::vector<std::shared_ptr<PointNormal> > & neighborhood);
	void set_neighborhood(const std::vector<PointNormal * > & neighborhood);

	int get_global_index() const;
	void set_global_index (int global_index);

	std::vector<int> get_neighborhood(double radius) const;



protected:

	arma::vec point;
	arma::vec normal;

	int inclusion_counter = 0;
	int match = -1;
	PointDescriptor descriptor;
	SPFH spfh;
	bool is_valid_feature = true;
	int global_index;

	std::map<double , int > neighborhood;

};




#endif