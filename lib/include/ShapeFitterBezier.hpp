#ifndef HEADER_ShapeFitterBezier
#define HEADER_ShapeFitterBezier

#include "ShapeModelBezier.hpp"
#include "ShapeFitter.hpp"

#include "PC.hpp"
#include <assert.h>
#include <map>
#include "CustomException.hpp"

class ShapeFitterBezier : public ShapeFitter {
	
public:

	ShapeFitterBezier(ShapeModelBezier * shape_model,PC * pc);

	virtual bool fit_shape_batch(
		unsigned int N_iter,
		double J,
		const arma::mat & DS, 
		const arma::vec & X_DS );

	// virtual bool fit_shape_KF(double J,const arma::mat & DS, const arma::vec & X_DS);



protected:

	ShapeModelBezier * shape_model;
	std::vector<Footpoint> find_footpoints() const;

	arma::sp_mat update_shape(std::vector<Footpoint> & footpoints);

	void update_patches_info_matrices(arma::sp_mat & info_mat,std::vector<Footpoint> & footpoints);


	static void find_footpoint_in_patch(Bezier * patch,Footpoint & footpoint);

	virtual void get_barycentric_coordinates(const arma::vec & Pbar,double & u, double & v, Facet * facet) const;

	virtual double compute_residuals(std::vector<std::pair<arma::vec,Footpoint > > & measurement_pairs) const;

	virtual void find_footpoint(Footpoint & footpoint) const ;

	virtual void save(std::string path, arma::mat & Pbar_mat) const ;

};


#endif