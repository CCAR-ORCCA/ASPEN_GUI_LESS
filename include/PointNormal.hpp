#ifndef HEADER_POINTNORMAL
#define HEADER_POINTNORMAL

#include <armadillo>
#include <memory>


class PointNormal {

public:

	PointNormal(arma::vec point);
	PointNormal(arma::vec point, int inclusion_counter) ;
	PointNormal(arma::vec point, arma::vec normal);


	double distance(std::shared_ptr<PointNormal> other_point) const;

	arma::vec * get_point();

	arma::vec * get_normal();

	void set_normal(arma::vec normal) ;

	void decrement_inclusion_counter();

	int get_inclusion_counter() const;

protected:

	arma::vec point;
	arma::vec normal;

	int inclusion_counter = 0;

};




#endif