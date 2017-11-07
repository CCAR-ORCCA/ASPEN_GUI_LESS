#ifndef HEADER_SHAPEFITTER
#define HEADER_SHAPEFITTER


#include "ShapeModelTri.hpp"
#include "PC.hpp"
#include <assert.h>
#include <map>



struct Footpoint{
	arma::vec Pbar;
	arma::vec Ptilde;
	double u;
	double v;
	arma::vec * n;
	Facet * facet;

};

class ShapeFitter{
	
public:

	ShapeFitter(ShapeModelTri * shape_model,PC * pc);

	bool fit_shape_batch(double J,
		const arma::mat & DS, 
		const arma::vec & X_DS );

	bool fit_shape_KF(double J,const arma::mat & DS, const arma::vec & X_DS);


protected:

	ShapeModelTri * shape_model;
	PC * pc;

	void apply_constraint_to_facet(const arma::vec & C0, 
		const arma::vec & C1,
		const arma::vec & C2,
		const arma::vec & n, 
		double u, 
		double v,
		arma::sp_mat & info_mat,
		arma::vec & normal_mat,
		const arma::sp_mat & Mi,
		double W);

	void freeze_facet(Facet * facet, 
		std::map<std::shared_ptr<ControlPoint> , unsigned int> & seen_vertices,
		arma::sp_mat & info_mat,
		arma::vec & normal_mat,
		double W);

	void get_barycentric_coordinates(const arma::vec & Pbar,double & u, double & v, Facet * facet);

	double compute_residuals(std::vector<std::pair<arma::vec,Footpoint > > & measurement_pairs) const;

	void find_footpoint(const arma::vec & Ptilde, 
	Footpoint & footpoint);

	void save(std::string path, arma::mat & Pbar_mat) const ;


};


#endif