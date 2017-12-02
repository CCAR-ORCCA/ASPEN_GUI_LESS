#ifndef HEADER_ShapeFitterTri
#define HEADER_ShapeFitterTri

#include "ShapeModelTri.hpp"
#include "ShapeFitter.hpp"

#include "PC.hpp"
#include <assert.h>
#include <map>


class ShapeFitterTri : public ShapeFitter {
	
public:

	ShapeFitterTri(ShapeModelTri * shape_model,PC * pc);

	virtual bool fit_shape_batch(unsigned int N_iter,double J,
		const arma::mat & DS, 
		const arma::vec & X_DS );

	// virtual bool fit_shape_KF(double J,const arma::mat & DS, const arma::vec & X_DS);

protected:

	ShapeModelTri * shape_model;

	virtual void get_barycentric_coordinates(const arma::vec & Pbar,double & u, double & v, Facet * facet) const;

	virtual double compute_residuals(std::vector<std::pair<arma::vec,Footpoint > > & measurement_pairs) const;

	virtual void find_footpoint(Footpoint & footpoint) const;

	virtual void save(std::string path, arma::mat & Pbar_mat) const ;

};


#endif