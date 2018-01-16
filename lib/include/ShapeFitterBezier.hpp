#ifndef HEADER_ShapeFitterBezier
#define HEADER_ShapeFitterBezier

#include "ShapeModelBezier.hpp"
#include "ShapeFitter.hpp"

#include "PC.hpp"
#include <assert.h>
#include <map>
#include "CustomException.hpp"



#pragma omp declare reduction (+ : arma::mat : omp_out += omp_in)\
initializer( omp_priv = omp_orig )
#pragma omp declare reduction (+ : arma::vec : omp_out += omp_in)\
initializer( omp_priv = omp_orig )
#pragma omp declare reduction (+ : arma::sp_mat : omp_out += omp_in)\
initializer( omp_priv = omp_orig )

class ShapeFitterBezier : public ShapeFitter {
	
public:

	ShapeFitterBezier(ShapeModelBezier * shape_model,PC * pc);

	bool fit_shape_KF(
		unsigned int index,
		unsigned int N_iter, 
		double J,
		const arma::mat & DS, 
		const arma::vec & X_DS,
		double los_noise_sd_base,
		const arma::vec & u_dir); 


	// virtual bool fit_shape_KF(double J,const arma::mat & DS, const arma::vec & X_DS);


protected:

	ShapeModelBezier * shape_model;
	std::vector<Footpoint> find_footpoints() const;

	arma::mat update_shape(std::vector<Footpoint> & footpoints,
		bool & has_converged);
	bool update_element(Element * element, 
		std::vector<Footpoint> & footpoints,
		bool store_info_mat,double W,
		const arma::vec & u_dir
		);
	
	static void find_footpoint_in_patch(Bezier * patch,Footpoint & footpoint);

	virtual void get_barycentric_coordinates(const arma::vec & Pbar,double & u, double & v, Facet * facet) const;

	virtual double compute_residuals(std::vector<std::pair<arma::vec,Footpoint > > & measurement_pairs) const;

	virtual void find_footpoint(Footpoint & footpoint,Element * & element_guess) const ;

	virtual void save(std::string path, arma::mat & Pbar_mat) const ;


	std::vector<Footpoint> recompute_footpoints(const std::vector<Footpoint> & footpoints) const;
};


#endif