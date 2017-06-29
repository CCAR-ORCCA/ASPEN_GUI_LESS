#ifndef HEADER_ICP
#define HEADER_ICP
#include <armadillo>
#include <memory>
#include "PC.hpp"


class ICP {
public:
	ICP(std::shared_ptr<PC> pc_destination, std::shared_ptr<PC> pc_source);



protected:
	std::shared_ptr<PC> pc_destination;
	std::shared_ptr<PC> pc_source;

	void register_pc_mrp_multiplicative_partials(
	    const unsigned int iterations_max,
	    const double rel_tol,
	    const double stol,
	    const bool pedantic);

	arma::rowvec dGdSigma_multiplicative(const arma::vec & mrp, const arma::vec & P, const arma::vec & n);

	arma::umat compute_pairs_closest_compatible_minimum_point_to_plane_dist(
	    const arma::mat & dcm,
	    const arma::mat & x);

	double compute_rms_residuals(
	    const arma::mat & dcm,
	    const arma::vec & x,
	    const arma::umat & point_pairs);



};






#endif