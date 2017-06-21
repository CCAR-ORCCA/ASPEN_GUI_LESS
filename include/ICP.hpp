#ifndef HEADER_ICP
#define HEADER_ICP
#include <armadillo>

class ICP {
public:
	ICP(arma::vec * source_points, arma::vec * destination_points);

	void compute_normals();


protected:

	arma::vec * source_points;
	arma::vec * destination_points;

	arma::vec * source_normals;
	arma::vec * destination_normals;






};






#endif