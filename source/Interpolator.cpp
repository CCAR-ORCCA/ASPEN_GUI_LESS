#include "Interpolator.hpp"


Interpolator::Interpolator(arma::vec * T, arma::mat * X) {
	this -> T = T;
	this -> X = X;
}

arma::vec Interpolator::interpolate(double t,bool is_attitude) {


	unsigned int index_min = arma::abs((*this -> T) - t).index_min();
	unsigned int index_before, index_after;

	if (this -> T -> at(index_min) <= t && index_min < this -> T -> n_rows - 1) {
		index_before = index_min;
		index_after = index_min + 1;
	}

	else {
		index_after = index_min;
		index_before = index_min - 1;
	}

	arma::vec X_before = this -> X -> col(index_before);
	arma::vec X_after = this -> X -> col(index_after);

	if (arma::norm(X_after.rows(0, 2) - X_before.rows(0, 2)) > 1.5 && is_attitude) {
		index_before += 1;
		index_after += 1;

		X_before = this -> X -> col(index_before);
		X_after = this -> X -> col(index_after);

	}

	double t_before = this -> T -> at(index_before);
	double t_after = this -> T -> at(index_after);


	arma::vec interpolant = ((X_after - X_before)  * t
	                         + (t_after * X_before - t_before * X_after) ) / (t_after - t_before);


	return interpolant;
}
