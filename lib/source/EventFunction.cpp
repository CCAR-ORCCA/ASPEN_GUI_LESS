#include "EventFunction.hpp"


arma::vec EventFunction::event_function_mrp_omega(double t, arma::vec X, Args * args) {
	if (arma::norm(X.rows(0, 2)) > 1) {
		
		X.rows(0,2) = - X.rows(0, 2) / arma::dot(X . rows(0, 2), X . rows(0, 2));

		return X;
	}
	else {
		return X;
	}
}


arma::vec EventFunction::event_function_mrp(double t, arma::vec X, Args * args) {
	if (arma::norm(X.rows(0, 2)) > 1) {
		arma::vec mrp = - X.rows(0, 2) / arma::dot(X . rows(0, 2), X . rows(0, 2));
		return mrp;
	}
	else {
		return X;
	}
}


