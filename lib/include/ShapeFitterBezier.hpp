#ifndef HEADER_ShapeFitterBezier
#define HEADER_ShapeFitterBezier

#include "ShapeModelBezier.hpp"

#include "PC.hpp"
#include <assert.h>
#include <map>
#include "CustomException.hpp"

#include <Eigen/Sparse>
#include <Eigen/Jacobi>
#include <Eigen/Dense>

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
typedef Eigen::Triplet<double> T;
typedef Eigen::VectorXd EigVec;




#pragma omp declare reduction (+ : arma::mat : omp_out += omp_in)\
initializer( omp_priv = omp_orig )
#pragma omp declare reduction (+ : arma::vec : omp_out += omp_in)\
initializer( omp_priv = omp_orig )
#pragma omp declare reduction (+ : arma::sp_mat : omp_out += omp_in)\
initializer( omp_priv = omp_orig )

class ShapeFitterBezier {
	
public:

	ShapeFitterBezier(ShapeModelBezier * shape_model,PC * pc);

	bool fit_shape_batch(unsigned int N_iter, double ridge_coef);


protected:

	ShapeModelBezier * shape_model;

	std::vector<Footpoint> find_footpoints_omp() const;

void add_to_problem(std::vector<T>& coeffs,
	EigVec & N,
	const double y,
	// const arma::sp_mat & H_i,
	const std::vector<arma::rowvec> & elements_to_add,

	const std::vector<int> & global_indices);


	void penalize_tangential_motion(std::vector<T>& coeffs,unsigned int N_measurements);

	bool update_shape(std::vector<Footpoint> & footpoints,double ridge_coef);

	
	static void find_footpoint_in_patch_omp(Bezier * patch,Footpoint & footpoint);


	void find_footpoint_omp(Footpoint & footpoint) const ;
	
	PC * pc;



};


#endif