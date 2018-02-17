#ifndef HEADER_ShapeFitter
#define HEADER_ShapeFitter

#include "ShapeModelTri.hpp"
#include "PC.hpp"
#include "Footpoint.hpp"
#include <assert.h>
#include <map>




class ShapeFitter{
	
public:

	ShapeFitter(PC * pc) {
		this -> pc = pc;
	}


protected:

	PC * pc;

	virtual void get_barycentric_coordinates(const arma::vec & Pbar,double & u, double & v, Facet * facet) const = 0;

	virtual double compute_residuals(std::vector<std::pair<arma::vec,Footpoint > > & measurement_pairs) const = 0;

	virtual void find_footpoint(Footpoint & footpoint,Element * & element_guess) const  = 0;

	virtual void save(std::string path, arma::mat & Pbar_mat) const  = 0;

};


#endif