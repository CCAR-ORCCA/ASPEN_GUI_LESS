#ifndef HEADER_ShapeFitterBezier
#define HEADER_ShapeFitterBezier


#include <assert.h>
#include <map>
#include <CustomException.hpp>
#include <Eigen/Sparse>
#include <Eigen/Jacobi>
#include <Eigen/Dense>
#include <armadillo>

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
typedef Eigen::Triplet<double> T;
typedef Eigen::VectorXd EigVec;

template <class PointType> class ShapeModelBezier;
template <class PointType> class ShapeModelTri;
template <class PointType> class PointCloud;

class PointNormal;
class ControlPoint;

struct Footpoint;
class Bezier;



#pragma omp declare reduction (+ : arma::mat : omp_out += omp_in)\
initializer( omp_priv = omp_orig )
#pragma omp declare reduction (+ : arma::vec : omp_out += omp_in)\
initializer( omp_priv = omp_orig )
#pragma omp declare reduction (+ : arma::sp_mat : omp_out += omp_in)\
initializer( omp_priv = omp_orig )

class ShapeFitterBezier {
	
public:

	ShapeFitterBezier(ShapeModelTri<ControlPoint> * psr_shape,
		ShapeModelBezier<ControlPoint> * shape_model,
		PointCloud<PointNormal> * pc);

	bool fit_shape_batch(unsigned int N_iter, double ridge_coef);


protected:

	ShapeModelBezier<ControlPoint> * shape_model;
	ShapeModelTri<ControlPoint> * psr_shape;
	std::vector<Footpoint> find_footpoints_omp() const;



	void add_to_problem(std::vector<T>& coeffs,
		EigVec & N,
		const double y,
	// const arma::sp_mat & H_i,
		const std::vector<arma::rowvec> & elements_to_add,

		const std::vector<int> & global_indices);


	void penalize_tangential_motion(std::vector<T>& coeffs,unsigned int N_measurements);

	bool update_shape(std::vector<Footpoint> & footpoints,double ridge_coef);

	
	static bool refine_footpoint_coordinates(const Bezier & patch,Footpoint & footpoint);


	void find_footpoint_omp(Footpoint & footpoint) const ;
	
	PointCloud<PointNormal> * pc;





};


#endif