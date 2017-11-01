#ifndef HEADER_SHAPEFITTER
#define HEADER_SHAPEFITTER


#include "ShapeModelTri.hpp"
#include "PC.hpp"
#include <assert.h>
#include <map>

class ShapeFitter{
	
public:

	ShapeFitter(ShapeModelTri * shape_model,PC * pc);
	void fit_shape();

protected:

	ShapeModelTri * shape_model;
	PC * pc;

	void apply_constraint_to_facet(const arma::vec & C0, 
		const arma::vec & C1,
		const arma::vec & C2,
		const arma::vec & n, 
		double u, 
		double v,
		arma::mat & info_mat,
		arma::vec & normal_mat,
		const arma::sp_mat & Mi);

	void freeze_facet(Facet * facet, 
		std::map<std::shared_ptr<ControlPoint> , unsigned int> & seen_vertices,
		arma::mat & info_mat,
		arma::vec & normal_mat);


};


#endif