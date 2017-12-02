#ifndef HEADER_ICP
#define HEADER_ICP
#include <armadillo>
#include <memory>
#include "PC.hpp"
#include "CustomException.hpp"
#include "OMP_flags.hpp"

#pragma omp declare reduction (+ : arma::mat::fixed<6,6> : omp_out += omp_in)\
  initializer( omp_priv = omp_orig )
#pragma omp declare reduction (+ : arma::vec::fixed<6> : omp_out += omp_in)\
  initializer( omp_priv = omp_orig )


class ICP {
public:
	ICP(std::shared_ptr<PC> pc_destination, std::shared_ptr<PC> pc_source,
	    arma::mat dcm_0 = arma::eye<arma::mat>(3, 3),
	    arma::vec X_0 = arma::zeros<arma::vec>(3),
	    bool pedantic = false);

	arma::vec get_X() const;
	arma::mat get_M() const;
	arma::mat get_R() const;
	double get_J_res() const;

	std::vector<std::pair<std::shared_ptr<PointNormal>, std::shared_ptr<PointNormal> > > * get_point_pairs();



protected:
	std::shared_ptr<PC> pc_destination;
	std::shared_ptr<PC> pc_source;

	void register_pc_mrp_multiplicative_partials(
	    const unsigned int iterations_max,
	    const double rel_tol,
	    const double stol,
	    const bool pedantic,
	    arma::mat dcm_0 ,
	    arma::vec X_0);

	arma::rowvec dGdSigma_multiplicative(const arma::vec & mrp, const arma::vec & P, const arma::vec & n);

	double compute_rms_residuals(
	    const arma::mat & dcm,
	    const arma::vec & x) ;

	void compute_pairs_closest_compatible_minimum_point_to_plane_dist(
	    const arma::mat & dcm,
	    const arma::mat & x,
	    int h);

	void compute_pairs_closest_minimum_distance(
	    const arma::mat & dcm,
	    const arma::mat & x,
	    int h);

	// arma::mat compute_inertia(std::shared_ptr<PC> pc) ;


	arma::vec X;
	arma::mat DCM;
	arma::mat R;
	double J_res;

	std::vector<std::pair<std::shared_ptr<PointNormal>, std::shared_ptr<PointNormal> > > point_pairs;


};






#endif