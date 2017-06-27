#ifndef HEADER_POINTNORMAL
#define HEADER_POINTNORMAL

#include <armadillo>
#include <memory>


class PointNormal {

public:

	PointNormal(arma::vec point);
	double distance(std::shared_ptr<PointNormal> other_point) const;
	
	arma::vec * get_point();

protected:

	arma::vec point;
	arma::vec normal;

};




#endif