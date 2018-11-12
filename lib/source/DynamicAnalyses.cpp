#include "DynamicAnalyses.hpp"
#include <RigidBodyKinematics.hpp>
#define ATTITUDE_JACOBIAN_DEBUG 0

DynamicAnalyses::DynamicAnalyses(ShapeModelTri<ControlPoint> * shape_model) {
	this -> shape_model = shape_model;
}




arma::mat DynamicAnalyses::attitude_jacobian(arma::vec & attitude ,const arma::mat & inertia) const {


	arma::mat::fixed<6,6> A = arma::zeros<arma::mat>(6,6);
	arma::vec::fixed<3> sigma = attitude.subvec(0,2);
	arma::vec::fixed<3> omega = attitude.subvec(3,5);

	// dsigma_dot_dsigma
	A.submat(0,0,2,2) = (0.5 * (- omega * sigma.t() - RBK::tilde(omega) 
		+ arma::eye<arma::mat>(3,3)* arma::dot(sigma,omega) + sigma * omega.t()));


	#if ATTITUDE_JACOBIAN_DEBUG
	std::cout << "done computing dsigma_dot_dsigma\n";
	#endif

	// dsigma_dot_domega
	A.submat(0,3,2,5) = 1./4 * RBK::Bmat(sigma);



	#if ATTITUDE_JACOBIAN_DEBUG
	std::cout << "done computing dsigma_dot_domega\n";
	
	std::cout << RBK::tilde(omega) << std::endl;
	#endif

	// domega_dot_dsigma is zero 

	// domega_dot_domega

	A.submat(3,3,5,5) = arma::solve(inertia,- RBK::tilde(omega) * inertia + RBK::tilde(inertia * omega));


	#if ATTITUDE_JACOBIAN_DEBUG
	std::cout << "done computing domega_dot_domega\n";
	#endif
	return A;

}






