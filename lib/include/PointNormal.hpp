#ifndef HEADER_POINTNORMAL
#define HEADER_POINTNORMAL

#include <armadillo>
#include <memory>
#include "PointDescriptor.hpp"
#include "SPFH.hpp"

class PointNormal {

public:

	PointNormal(arma::vec point);
	PointNormal(arma::vec point, int inclusion_counter) ;
	PointNormal(arma::vec point, arma::vec normal);


	double distance(std::shared_ptr<PointNormal> other_point) const;

	arma::vec get_point() const;

	arma::vec get_normal() const;

	void set_normal(arma::vec normal) ;
	void set_point(arma::vec point) ;
	void set_descriptor(const PointDescriptor & descriptor) ;
	
	PointDescriptor get_descriptor() const;
	PointDescriptor * get_descriptor_ptr();

	arma::vec get_descriptor_histogram() const;
	arma::vec get_spfh_histogram() const;

	unsigned int get_histogram_size() const;
	double get_histogram_value(int index) const;

	void decrement_inclusion_counter();

	int get_inclusion_counter() const;

	double descriptor_distance(std::shared_ptr<PointNormal> other_point) const;

	void set_SPFH(SPFH spfh);

	SPFH * get_SPFH();

	bool get_is_unique_feature() const;
	void set_is_unique_feature(bool is_unique);


protected:

	arma::vec point;
	arma::vec normal;

	int inclusion_counter = 0;
	PointDescriptor descriptor;
	SPFH spfh;
	bool is_unique = true;

};




#endif