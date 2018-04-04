#include "Dynamics.hpp"

arma::vec Dynamics::point_mass_dxdt(double t, arma::vec X, Args * args) {

	arma::vec pos_inertial = X . subvec(0, 2);
	arma::vec acc_inertial = args -> get_dyn_analyses() -> point_mass_acceleration(pos_inertial , args -> get_mass());

	arma::vec dxdt = { X(3), X(4), X(5), acc_inertial(0), acc_inertial(1), acc_inertial(2)};

	return dxdt;

}


arma::vec Dynamics::point_mass_dxdt_odeint(double t, const arma::vec & x, const Args & args) {
	arma::vec pos_inertial = x . subvec(0, 2);
	arma::vec acc_inertial = args.get_dyn_analyses() -> point_mass_acceleration(pos_inertial , args . get_mass());

	arma::vec dxdt = { x(3), x(4), x(5), acc_inertial(0), acc_inertial(1), acc_inertial(2)};

	return dxdt;

}

arma::mat Dynamics::point_mass_jac_odeint(double t, const arma::vec & x, const Args & args) {

	arma::vec pos_inertial = x . subvec(0, 2);
	return args.get_dyn_analyses() -> point_mass_jacobian(pos_inertial , args . get_mass());


}


arma::mat Dynamics::gamma_OD(double dt){
	arma::mat gamma = arma::zeros<arma::mat>(6,3);
	gamma.submat(0,0,2,2) = 0.5 * dt * arma::eye<arma::mat>(3,3);
	gamma.submat(3,0,5,2) = arma::eye<arma::mat>(3,3);

	gamma = dt * gamma;
	return gamma;
}


arma::mat Dynamics::gamma_OD_augmented(double dt){
	arma::mat gamma = arma::zeros<arma::mat>(12,3);
	gamma.submat(0,0,2,2) = 0.5 * dt * arma::eye<arma::mat>(3,3);
	gamma.submat(3,0,5,2) = arma::eye<arma::mat>(3,3);

	gamma = dt * gamma;
	return gamma;
}


arma::vec Dynamics::point_mass_dxdt_body_frame(double t, arma::vec X, Args * args) {

	arma::vec attitude_state = args -> get_interpolator() -> interpolate(t, true);

	arma::vec mrp_TN = attitude_state.rows(0, 2);
	arma::vec omega_TN = attitude_state.rows(3, 5);

	arma::vec pos_body = X . subvec(0, 2);
	arma::vec vel_body = X . subvec(3, 5);

	arma::vec acc_body_grav = args -> get_dyn_analyses() -> point_mass_acceleration(pos_body , args -> get_mass());
	arma::vec acc_body_frame = acc_body_grav - (2 * arma::cross(omega_TN, vel_body) + omega_TN * omega_TN.t() * pos_body - pos_body * omega_TN.t() * omega_TN);

	arma::vec dxdt = { X(3), X(4), X(5), acc_body_frame(0), acc_body_frame(1), acc_body_frame(2)};
	return dxdt;

}


arma::vec Dynamics::point_mass_attitude_dxdt_body_frame(double t,const arma::vec & X, const Args & args) {

	arma::vec pos_body = X . subvec(0, 2);
	arma::vec vel_body = X . subvec(3, 5);

	arma::vec X_spacecraft = X . subvec(0, 5);

	arma::vec mrp_TN = X . subvec(6, 8);
	arma::vec omega_TN = X . subvec(9, 11);

	arma::vec X_small_body = X . subvec(6, 11);

	arma::vec acc_body_grav = args. get_dyn_analyses() -> point_mass_acceleration(pos_body , args. get_mass());
	arma::vec acc_body_frame = acc_body_grav - (2 * arma::cross(omega_TN, vel_body) + omega_TN * omega_TN.t() * pos_body - pos_body * omega_TN.t() * omega_TN);

	arma::vec dxdt = arma::zeros<arma::vec>(12);
	arma::vec dxdt_spacecraft = { X(3), X(4), X(5), acc_body_frame(0), acc_body_frame(1), acc_body_frame(2)};
	
	arma::vec dxdt_small_body = Dynamics::attitude_dxdt(t, X_small_body, args);
	
	dxdt.subvec(0,5) = dxdt_spacecraft;
	dxdt.subvec(6,11) = dxdt_small_body;

	return dxdt;

}


arma::vec Dynamics::estimated_point_mass_attitude_dxdt_body_frame(double t,const arma::vec & X, const Args & args) {

	arma::vec pos_body = X . subvec(0, 2);
	arma::vec vel_body = X . subvec(3, 5);

	arma::vec X_spacecraft = X . subvec(0, 5);

	arma::vec mrp_TN = X . subvec(6, 8);
	arma::vec omega_TN = X . subvec(9, 11);

	arma::vec X_small_body = X . subvec(6, 11);

	arma::vec acc_body_grav = args. get_dyn_analyses() -> point_mass_acceleration(pos_body , args. get_estimated_mass());
	arma::vec acc_body_frame = acc_body_grav - (2 * arma::cross(omega_TN, vel_body) + omega_TN * omega_TN.t() * pos_body - pos_body * omega_TN.t() * omega_TN);

	arma::vec dxdt = arma::zeros<arma::vec>(12);
	arma::vec dxdt_spacecraft = { X(3), X(4), X(5), acc_body_frame(0), acc_body_frame(1), acc_body_frame(2)};
	
	arma::vec dxdt_small_body = Dynamics::attitude_dxdt(t, X_small_body, args);
	
	dxdt.subvec(0,5) = dxdt_spacecraft;
	dxdt.subvec(6,11) = dxdt_small_body;

	return dxdt;

}


arma::vec Dynamics::harmonics_attitude_dxdt_body_frame(double t,const arma::vec & X, const Args & args) {

	arma::vec pos_body = X . subvec(0, 2);
	arma::vec vel_body = X . subvec(3, 5);

	arma::vec X_spacecraft = X . subvec(0, 5);

	arma::vec mrp_TN = X . subvec(6, 8);
	arma::vec omega_TN = X . subvec(9, 11);

	arma::vec X_small_body = X . subvec(6, 11);

	arma::vec acc_body_grav = args. get_dyn_analyses() -> spherical_harmo_acc(
		args.get_harmonics_degree(),
		args.get_ref_radius(),
		args.get_mu(),
		pos_body, 
		args.get_Cnm(),
		args.get_Snm());
	

	arma::vec acc_body_frame = acc_body_grav - (2 * arma::cross(omega_TN, vel_body) + omega_TN * omega_TN.t() * pos_body - pos_body * omega_TN.t() * omega_TN);

	arma::vec dxdt = arma::zeros<arma::vec>(12);
	arma::vec dxdt_spacecraft = { X(3), X(4), X(5), acc_body_frame(0), acc_body_frame(1), acc_body_frame(2)};
	
	arma::vec dxdt_small_body = Dynamics::attitude_dxdt(t, X_small_body, args);
	
	dxdt.subvec(0,5) = dxdt_spacecraft;
	dxdt.subvec(6,11) = dxdt_small_body;

	return dxdt;

}

arma::mat Dynamics::point_mass_jac_attitude_dxdt_body_frame(double t, const arma::vec & X, const Args & args){

	arma::mat A = arma::zeros<arma::mat>(12,12);

	arma::vec pos_body = X . subvec(0, 2);
	arma::vec vel_body = X . subvec(3, 5);

	arma::vec mrp_TN = X . subvec(6, 8);
	arma::vec omega_TN = X . subvec(9, 11);

	A.submat(0,0,5,5) += args.get_dyn_analyses() -> point_mass_jacobian(pos_body , args . get_mass());
	A.submat(3,0,5,2) += - omega_TN * omega_TN.t() + arma::eye<arma::mat>(3,3) * arma::dot(omega_TN,omega_TN);
	A.submat(3,3,5,5) = - 2 * RBK::tilde(omega_TN);

	return A;

}


arma::mat Dynamics::estimated_point_mass_jac_attitude_dxdt_body_frame(double t, const arma::vec & X, const Args & args){

	arma::mat A = arma::zeros<arma::mat>(12,12);

	arma::vec pos_body = X . subvec(0, 2);
	arma::vec vel_body = X . subvec(3, 5);

	arma::vec mrp_TN = X . subvec(6, 8);
	arma::vec omega_TN = X . subvec(9, 11);


	A.submat(0,0,5,5) += args.get_dyn_analyses() -> point_mass_jacobian(pos_body , args . get_estimated_mass());

	A.submat(3,0,5,2) += - omega_TN * omega_TN.t() + arma::eye<arma::mat>(3,3) * arma::dot(omega_TN,omega_TN);

	A.submat(3,3,5,5) = - 2 * RBK::tilde(omega_TN);


	return A;


}



arma::vec Dynamics::point_mass_attitude_dxdt_inertial(double t,const arma::vec & X, const Args & args) {

	arma::vec pos = X . subvec(0, 2);

	arma::vec X_small_body = X . subvec(6, 11);

	arma::vec acc_body_grav = args. get_dyn_analyses() -> point_mass_acceleration(pos , args. get_mass());

	arma::vec dxdt = arma::zeros<arma::vec>(12);
	arma::vec dxdt_spacecraft = { X(3), X(4), X(5), acc_body_grav(0), acc_body_grav(1), acc_body_grav(2)};
	arma::vec dxdt_small_body = Dynamics::attitude_dxdt(t, X_small_body, args);
	
	dxdt.subvec(0,5) = dxdt_spacecraft;
	dxdt.subvec(6,11) = dxdt_small_body;

	return dxdt;

}


arma::vec Dynamics::estimated_point_mass_attitude_dxdt_inertial(double t,const arma::vec & X, const Args & args) {

	arma::vec pos = X . subvec(0, 2);

	arma::vec X_small_body = X . subvec(6, 11);

	arma::vec acc_body_grav = args. get_dyn_analyses() -> point_mass_acceleration(pos , args. get_estimated_mass());

	arma::vec dxdt = arma::zeros<arma::vec>(12);
	arma::vec dxdt_spacecraft = { X(3), X(4), X(5), acc_body_grav(0), acc_body_grav(1), acc_body_grav(2)};
	
	arma::vec dxdt_small_body = Dynamics::attitude_dxdt(t, X_small_body, args);
	
	dxdt.subvec(0,5) = dxdt_spacecraft;
	dxdt.subvec(6,11) = dxdt_small_body;

	return dxdt;

}


arma::vec Dynamics::harmonics_attitude_dxdt_inertial(double t,const arma::vec & X, const Args & args) {

	arma::vec pos = X . subvec(0, 2);

	arma::vec X_spacecraft = X . subvec(0, 5);

	arma::vec X_small_body = X . subvec(6, 11);

	arma::vec pos_body_frame = args.get_frame_graph() -> convert(pos,"N","B");

	arma::vec acc_grav_inertial = args.get_frame_graph() -> convert(
		args. get_dyn_analyses() -> spherical_harmo_acc(
			args.get_harmonics_degree(),
			args.get_ref_radius(),
			args.get_mu(),
			pos_body_frame, 
			args.get_Cnm(),
			args.get_Snm()),
		"B","N");
	

	arma::vec dxdt = arma::zeros<arma::vec>(12);
	arma::vec dxdt_spacecraft = { X(3), X(4), X(5), acc_grav_inertial(0), acc_grav_inertial(1), acc_grav_inertial(2)};
	
	arma::vec dxdt_small_body = Dynamics::attitude_dxdt(t, X_small_body, args);
	
	dxdt.subvec(0,5) = dxdt_spacecraft;
	dxdt.subvec(6,11) = dxdt_small_body;

	return dxdt;

}

arma::mat Dynamics::point_mass_jac_attitude_dxdt_inertial(double t, const arma::vec & X, const Args & args){

	arma::mat A = arma::zeros<arma::mat>(12,12);

	arma::vec pos = X . subvec(0, 2);

	A.submat(0,0,5,5) += args.get_dyn_analyses() -> point_mass_jacobian(pos , args . get_mass());

	return A;

}


arma::mat Dynamics::estimated_point_mass_jac_attitude_dxdt_inertial(double t, const arma::vec & X, const Args & args){

	arma::mat A = arma::zeros<arma::mat>(12,12);

	arma::vec pos = X . subvec(0, 2);

	A.submat(0,0,5,5) += args.get_dyn_analyses() -> point_mass_jacobian(pos , args . get_estimated_mass());

	return A;


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

	arma::vec omega = X . subvec(3, 5);

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

