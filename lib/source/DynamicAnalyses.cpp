#include "DynamicAnalyses.hpp"
#include <RigidBodyKinematics.hpp>
#define ATTITUDE_JACOBIAN_DEBUG 1

DynamicAnalyses::DynamicAnalyses(ShapeModelTri<ControlPoint> * shape_model) {
	this -> shape_model = shape_model;
}

arma::vec DynamicAnalyses::point_mass_acceleration(arma::vec & point , double mass) const {


	if (point.n_rows != 3){
		throw (std::runtime_error("vector argument should have 3 components, not" + std::to_string(point.n_rows)));
	}

	arma::vec acc = - mass * arma::datum::G / arma::dot(point, point) * arma::normalise(point);
	

	return acc;


}

arma::mat DynamicAnalyses::point_mass_jacobian(arma::vec & point , double mass) const {


	if (point.n_rows != 3){
		throw (std::runtime_error("vector argument should have 3 components, not" + std::to_string(point.n_rows)));
	}

	arma::mat A = arma::zeros<arma::mat>(6,6);

	A.submat(0,3,2,5) = arma::eye<arma::mat>(3,3);
	A.submat(3,0,5,2) = - mass * arma::datum::G * (
		arma::eye<arma::mat>(3,3) / std::pow(arma::norm(point),3) 
		- 3 * point * point.t() / std::pow(arma::norm(point),5));

	return A;

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
	std::cout << omega.t() << std::endl;
	std::cout << inetia << std::endl;

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







void DynamicAnalyses::GetBnmNormalizedExterior(int n_degree,
	arma::mat & b_bar_real,
	arma::mat & b_bar_imag,
	arma::vec pos,
	double ref_radius){


	double r_sat = arma::norm(pos);
	double x_sat = pos(0);
	double y_sat = pos(1);
	double z_sat = pos(2);

	double delta_1_n;


    /*////////////////////////////////
    // -- Vertical Recurrences -- //
    ////////////////////////////////*/

	for (int mm = 0; mm <= n_degree+2; mm++) {


      // Pretty sure I don't need this

		double m = (double) mm;

		for (int nn = mm; nn <= n_degree+2; nn++) {

      // Pretty sure I don't need this

			double n = (double) nn;

            /*// Recursive Formulae*/

			if (mm == nn) {

				if (mm == 0) {

					b_bar_real(0,0) = ref_radius/r_sat;
					b_bar_imag(0,0) = 0.0;

				} else {

					if (nn == 1) {

						delta_1_n = 1.0;

					} else {

						delta_1_n = 0.0;

					} 

					b_bar_real(nn,nn) = sqrt( (1.0 + delta_1_n)*(2.0*n + 1.0)/(2.0*n) ) * (ref_radius/r_sat) * ( x_sat/r_sat*b_bar_real(nn-1,nn-1) - y_sat/r_sat*b_bar_imag(nn-1,nn-1) );
					b_bar_imag(nn,nn) = sqrt( (1.0 + delta_1_n)*(2.0*n + 1.0)/(2.0*n) ) * (ref_radius/r_sat) * ( y_sat/r_sat*b_bar_real(nn-1,nn-1) + x_sat/r_sat*b_bar_imag(nn-1,nn-1) );

				} 

			} 

			else {

				if ( nn >= 2 ) {

					b_bar_real(nn,mm) = sqrt( (4.0*n*n - 1.0)/(n*n - m*m) )*(ref_radius/r_sat)*(z_sat/r_sat)*b_bar_real(nn-1,mm) - sqrt( (2.0*n + 1.0)*( (n - 1.0)*(n - 1.0) - m*m )/( (2.0*n - 3.0)*(n*n - m*m) ) )*(ref_radius/r_sat)*(ref_radius/r_sat)*b_bar_real(nn-2,mm);
					b_bar_imag(nn,mm) = sqrt( (4.0*n*n - 1.0)/(n*n - m*m) )*(ref_radius/r_sat)*(z_sat/r_sat)*b_bar_imag(nn-1,mm) - sqrt( (2.0*n + 1.0)*( (n - 1.0)*(n - 1.0) - m*m )/( (2.0*n - 3.0)*(n*n - m*m) ) )*(ref_radius/r_sat)*(ref_radius/r_sat)*b_bar_imag(nn-2,mm);

				} else {

					b_bar_real(nn,mm) = sqrt( (4.0*n*n - 1.0)/(n*n - m*m) )*(ref_radius/r_sat)*(z_sat/r_sat)*b_bar_real(nn-1,mm);
					b_bar_imag(nn,mm) = sqrt( (4.0*n*n - 1.0)/(n*n - m*m) )*(ref_radius/r_sat)*(z_sat/r_sat)*b_bar_imag(nn-1,mm);

				} 

			} 

		} 

	} 

} 


arma::vec DynamicAnalyses::spherical_harmo_acc(const unsigned int n_degree,
	const double ref_radius,
	const double  mu,
	arma::vec pos, 
	arma::mat * Cbar,
	arma::mat * Sbar) {

	int n_max = 50;

	if (Cbar -> n_rows < n_degree){
		throw std::runtime_error("Cbar has fewer rows than the queried spherical expansion degree");
	}

	if (Sbar -> n_rows < n_degree){
		throw std::runtime_error("Sbar has fewer rows than the queried spherical expansion degree");
	}


	arma::mat b_bar_real = arma::zeros<arma::mat>(n_max + 3,n_max + 3);
	arma::mat b_bar_imag = arma::zeros<arma::mat>(n_max + 3,n_max + 3);

	GetBnmNormalizedExterior(n_degree,
		b_bar_real,
		b_bar_imag,
		pos,
		ref_radius);


	double K0 = 0.5 * mu / ref_radius / ref_radius;
	double x_ddot = 0;
	double y_ddot = 0;
	double z_ddot = 0;

	for (unsigned int nn = 0; nn<=n_degree; nn++){

		double n = (double) nn;

		for (unsigned int mm = 0; mm<=nn; mm++){

			double m = (double) mm;
			double delta_1_m;

			if (mm == 1){
				delta_1_m = 1.0;
			}
			else{
				delta_1_m = 0.0;
			} 

			double K1 = sqrt( (n+2.0) * (n+1.0) * (2.0*n+1.0) / 2.0 / (2.0*n+3.0) );
			double K2 = sqrt( (n+m+2.0) * (n+m+1.0) * (2.0*n+1.0) / (2.0*n+3.0) );
			double K3 = sqrt( 2.0 * (n-m+2.0) * (n-m+1.0) * (2.0*n+1.0) / (2.0 - delta_1_m) / (2.0*n+3.0) );
			if (mm == 0){

				x_ddot -= 2.0*K0 * ( Cbar -> at(nn,mm)*K1*b_bar_real(nn+1,mm+1) );
				y_ddot -= 2.0*K0 * ( Cbar -> at(nn,mm)*K1*b_bar_imag(nn+1,mm+1) );
				z_ddot -= 2.0*K0 * ( Cbar -> at(nn,mm)*sqrt((n-m+1.0)*(n+m+1.0)*(2.0*n+1.0)/(2.0*n+3.0))*b_bar_real(nn+1,mm) );


			}
			else{

				x_ddot += K0 * ( -Cbar -> at(nn,mm)*K2*b_bar_real(nn+1,mm+1) -Sbar -> at(nn,mm)*K2*b_bar_imag(nn+1,mm+1) +Cbar -> at(nn,mm)*K3*b_bar_real(nn+1,mm-1) +Sbar -> at(nn,mm)*K3*b_bar_imag(nn+1,mm-1));
				y_ddot += K0 * ( -Cbar -> at(nn,mm)*K2*b_bar_imag(nn+1,mm+1) +Sbar -> at(nn,mm)*K2*b_bar_real(nn+1,mm+1) -Cbar -> at(nn,mm)*K3*b_bar_imag(nn+1,mm-1) +Sbar -> at(nn,mm)*K3*b_bar_real(nn+1,mm-1));
				z_ddot -= 2.0*K0 * ( Cbar -> at(nn,mm)*sqrt((n-m+1.0)*(n+m+1.0)*(2.0*n+1.0)/(2.0*n+3.0))*b_bar_real(nn+1,mm) +Sbar -> at(nn,mm)*sqrt((n-m+1.0)*(n+m+1.0)*(2*n+1.0)/(2.0*n+3.0))*b_bar_imag(nn+1,mm) );

			} 

		} 

	} 

	arma::vec acceleration = {x_ddot,y_ddot,z_ddot};

	return acceleration;

} 