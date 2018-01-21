#include "EllipsoidFitter.hpp"
#include "DebugFlags.hpp"


arma::vec EllipsoidFitter::run(const arma::vec & X_bar,const arma::mat Pbar,
	unsigned int N_iter) const{


	unsigned int p = this -> pc -> get_size();
	arma::vec y = arma::zeros<arma::vec>(p);
	arma::vec X = X_bar;
	arma::vec dX_bar = arma::zeros<arma::vec>(6);

	double norm_res = -1;

	if (N_iter == 0){
		return X_bar;
	}

	for (unsigned int i = 0; i < N_iter; ++i){
		
		arma::mat info_mat;

		if (Pbar.max() == 0){
			info_mat = arma::zeros<arma::mat>(6,6);
		}
		else{
			info_mat = arma::inv(Pbar);
		}

		arma::vec normal_mat = info_mat * dX_bar;

		arma::rowvec H = arma::zeros<arma::rowvec>(6);

		for (unsigned int j = 0; j < p; ++j){

		// The residuals are computed
			arma::vec point = this -> pc -> get_point_coordinates(j);
			y(j) = 1. - this -> G(point,X);
			H = this -> jacobian(point,X);

			// The normal and information matrices are accumulated
			info_mat += H.t() * H;
			normal_mat += H.t() * y(j) ;
		}


		




		// The deviation in the state is solved
		arma::vec dX = arma::solve(info_mat,normal_mat);

		dX_bar = dX_bar - dX;




		X += dX;

		#if ELLISPOID_DEBUG
			std::cout << X << std::endl;
		#endif


		if (norm_res < 0){
			norm_res = arma::norm(y);
			#if ELLISPOID_DEBUG
				std::cout << "- Residuals: " << norm_res  << std::endl;
			#endif
		}
		else{
			double variation = std::abs(arma::norm(y) - norm_res) / norm_res;
			if ( variation < 1e-10 || norm_res < 1e-10){
				
				#if ELLISPOID_DEBUG
					std::cout << "- Converged in " << i + 1 << " iterations" << std::endl;
					std::cout << "- Residuals at convergence: " << arma::norm(y)  << std::endl;
				#endif

				break;
			}
			else{
				norm_res = arma::norm(y);

				#if ELLISPOID_DEBUG
					std::cout << "- Residuals: " << norm_res  << std::endl;
					std::cout << "- Variation: " << variation << std::endl;
				#endif
			}
		}
	}
	return X;

}

double EllipsoidFitter::G(const arma::vec & point, const arma::vec & X) const{
	arma::mat A = {
		{1./X(0),0,0},
		{0,1./X(1),0},
		{0,0,1./X(2)}
	};
	return arma::dot(X.subvec(3,5) - point,A * A * (X.subvec(3,5) - point));

}

arma::rowvec EllipsoidFitter::jacobian(const arma::vec & point, const arma::vec & X) const{
	
	arma::rowvec H = arma::zeros<arma::rowvec>(6);
	
	arma::mat A = {
		{1./X(0),0,0},
		{0,1./X(1),0},
		{0,0,1./X(2)}
	};

	arma::mat X_proj = {
		{1,0,0},
		{0,0,0},
		{0,0,0}
	};
	arma::mat Y_proj = {
		{0,0,0},
		{0,1,0},
		{0,0,0}
	};
	arma::mat Z_proj = {
		{0,0,0},
		{0,0,0},
		{0,0,1}
	};




	H(0) = - 2 * std::pow(X(3) - point(0),2) / std::pow(X(0),3);
	H(1) = - 2 * std::pow(X(4) - point(1),2) / std::pow(X(1),3);
	H(2) = - 2 * std::pow(X(5) - point(2),2) / std::pow(X(2),3);

	H.subvec(3,5) = 2 * (X.subvec(3,5) - point).t() * A * A;
	return H;
}

