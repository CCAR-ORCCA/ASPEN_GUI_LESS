#include "Dynamics.hpp"

arma::vec Dynamics::point_mass_dxdt(double t, arma::vec X, Args * args) {

	

	arma::vec pos_inertial = X . rows(0, 2);
	arma::vec acc_inertial = args -> get_dyn_analyses() -> point_mass_acceleration(pos_inertial , args -> get_mass());

	arma::vec dxdt = { X(3), X(4), X(5), acc_inertial(0), acc_inertial(1), acc_inertial(2)};

	return dxdt;

}


arma::vec Dynamics::point_mass_dxdt_odeint(double t, const arma::vec & x, const Args & args) {
	arma::vec pos_inertial = x . rows(0, 2);
	arma::vec acc_inertial = args.get_dyn_analyses() -> point_mass_acceleration(pos_inertial , args . get_mass());

	arma::vec dxdt = { x(3), x(4), x(5), acc_inertial(0), acc_inertial(1), acc_inertial(2)};

	return dxdt;

}

arma::mat Dynamics::point_mass_jac_odeint(double t, const arma::vec & x, const Args & args) {

	arma::vec pos_inertial = x . rows(0, 2);
	return args.get_dyn_analyses() -> point_mass_jacobian(pos_inertial , args . get_mass());


}


arma::mat Dynamics::gamma_OD(double dt){
	arma::mat gamma = arma::zeros<arma::mat>(6,3);
	gamma.submat(0,0,2,2) = 0.5 * dt * arma::eye<arma::mat>(3,3);
	gamma.submat(3,0,5,2) = arma::eye<arma::mat>(3,3);

	gamma = dt * gamma;
	return gamma;
}


arma::vec Dynamics::point_mass_dxdt_body_frame(double t, arma::vec X, Args * args) {

	arma::vec attitude_state = args -> get_interpolator() -> interpolate(t, true);

	arma::vec mrp_TN = attitude_state.rows(0, 2);
	arma::vec omega_TN = attitude_state.rows(3, 5);

	arma::vec pos_body = X . rows(0, 2);
	arma::vec vel_body = X . rows(3, 5);

	arma::vec acc_body_grav = args -> get_dyn_analyses() -> point_mass_acceleration(pos_body , args -> get_mass());
	arma::vec acc_body_frame = acc_body_grav - (2 * arma::cross(omega_TN, vel_body) + omega_TN * omega_TN.t() * pos_body - pos_body * omega_TN.t() * omega_TN);

	arma::vec dxdt = { X(3), X(4), X(5), acc_body_frame(0), acc_body_frame(1), acc_body_frame(2)};
	return dxdt;

}


// arma::vec Dynamics::attitude_dxdt(double t, arma::vec  X, Args * args) {

// 	arma::vec dxdt = RBK::dXattitudedt(t, X , args -> get_active_inertia());

// 	return dxdt;

// }

arma::vec Dynamics::attitude_dxdt(double t, const arma::vec & X, const Args & args) {

	arma::vec dxdt = RBK::dXattitudedt(t, X , args . get_active_inertia());

	return dxdt;

}



double Dynamics::energy_attitude(double t, arma::vec X , Args * args) {

	arma::vec omega = X . rows(3, 5);

	return 0.5 * arma::dot(omega, args -> get_active_inertia() * omega);

}

arma::vec Dynamics::joint_sb_spacecraft_body_frame_dyn(double t, arma::vec  X, Args * args){

	arma::vec dxdt(X.n_rows);

	// arma::vec sigma = X.rows(0,3);
	// arma::vec omega = X.rows(3,5);
	// arma::vec pos = X.rows(6,8);
	// arma::vec vel = X.rows(9,11);
	

	// dxdt.rows(0,5) = attitude_dxdt(t,X.rows(0,5),args);

	// arma::vec omega_dot = dxdt.rows(3,5);

	// arma::vec acc_sph = args -> get_dyn_analyses() -> spherical_harmo_acc(
	// 	args -> get_degree(),
	// 	args -> get_ref_radius(),
	// 	args -> get_mu(),
	// 	pos, 
	// 	args -> get_Cnm(),
	// 	args -> get_Snm());

	// dxdt.rows(6,8) = X.rows(9,11);
	// dxdt.rows(9,11) = (acc_sph - arma::cross(omega_dot,pos) - 2 * arma::cross(omega,vel)
	// 	- arma::cross(omega,arma::cross(omega,pos)));

	return dxdt;

}

