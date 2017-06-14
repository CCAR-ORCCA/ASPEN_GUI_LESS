#include "Wrappers.hpp"

arma::vec pgm_dxdt_wrapper(double t, arma::vec X, Args * args) {

	DynamicAnalyses dyn_analyses(args -> get_shape_model());

	arma::vec mrp_TN = args -> get_interpolator() -> interpolate(t).rows(0, 2);

	args -> get_frame_graph() -> set_transform_mrp("N", "T", mrp_TN);
	arma::vec pos_inertial = X . rows(0, 2);
	arma::vec pos_body = args -> get_frame_graph() -> convert(pos_inertial, "N", "T");
	arma::vec acc_body = dyn_analyses.pgm_acceleration(pos_body.colptr(0) , args -> get_density());
	arma::vec acc_inertial = args -> get_frame_graph() -> convert(acc_body, "T", "N", true);

	arma::vec dxdt = {X(3), X(4), X(5), acc_inertial(0), acc_inertial(1), acc_inertial(2)};
	return dxdt;

}

arma::vec attitude_dxdt_wrapper(double t, arma::vec  X, Args * args) {

	arma::vec dxdt = dXattitudedt(t, X , args -> get_shape_model() -> get_body_inertia());
	return dxdt;

}

arma::vec event_function_mrp(double t, arma::vec X, Args * args) {
	if (arma::norm(X.rows(0, 2)) > 1) {
		arma::vec mrp = - X.rows(0, 2) / arma::dot(X . rows(0, 2), X . rows(0, 2));
		arma::vec state = {mrp(0), mrp(1), mrp(2), X(3), X(4), X(5)};
		return state;
	}
	else {
		return X;
	}
}

arma::vec event_function_collision(double t, arma::vec X, Args * args) {

	arma::vec mrp_TN = args -> get_interpolator() -> interpolate(t).rows(0, 2);

	args -> get_frame_graph() -> set_transform_mrp("N", "T", mrp_TN);
	arma::vec pos_inertial = X . rows(0, 2);
	arma::vec pos_body = args -> get_frame_graph() -> convert(pos_inertial, "N", "T");

	if (args -> get_shape_model() -> contains(pos_body.colptr(0))) {
		std::cout << " The spacecraft collided with the surface at time t = " << t << " s" << std::endl;
		args -> set_stopping_bool(true);
	}
	return X;
}


double energy_attitude(double t, arma::vec X , Args * args) {

	arma::vec omega = X . rows(3, 5);

	return 0.5 * arma::dot(omega, args -> get_shape_model() -> get_body_inertia() * omega);


}
double energy_orbit(double t, arma::vec X , Args * args) {

	DynamicAnalyses dyn_analyses(args -> get_shape_model());

	arma::vec pos_inertial = X . rows(0, 2);
	arma::vec pos_body = args -> get_frame_graph() -> convert(pos_inertial, "N", "T");

	double potential = dyn_analyses.pgm_potential(pos_body.colptr(0) , args -> get_density());
	return 0.5 * arma::dot(X . rows(3, 5), X . rows(3, 5)) - potential;

}
