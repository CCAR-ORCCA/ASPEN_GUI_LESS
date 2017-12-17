#include "Wrapper.hpp"


arma::vec Wrapper::point_mass_dxdt_wrapper(double t, arma::vec X, Args * args) {

	

	arma::vec pos_inertial = X . rows(0, 2);
	arma::vec acc_inertial = args -> get_dyn_analyses() -> point_mass_acceleration(pos_inertial , args -> get_mass());

	arma::vec dxdt = { X(3), X(4), X(5), acc_inertial(0), acc_inertial(1), acc_inertial(2)};

	return dxdt;

}


arma::vec Wrapper::point_mass_dxdt_wrapper_odeint(double t, const arma::vec & x, const Args & args) {
	arma::vec pos_inertial = x . rows(0, 2);
	arma::vec acc_inertial = args.get_dyn_analyses() -> point_mass_acceleration(pos_inertial , args . get_mass());

	arma::vec dxdt = { x(3), x(4), x(5), acc_inertial(0), acc_inertial(1), acc_inertial(2)};

	return dxdt;

}

arma::mat Wrapper::point_mass_jac_wrapper_odeint(double t, const arma::vec & x, const Args & args) {

	arma::vec pos_inertial = x . rows(0, 2);
	return args.get_dyn_analyses() -> point_mass_jacobian(pos_inertial , args . get_mass());


}


arma::mat Wrapper::gamma_OD(double dt){
	arma::mat gamma = arma::zeros<arma::mat>(6,3);
	gamma.submat(0,0,2,2) = 0.5 * dt * arma::eye<arma::mat>(3,3);
	gamma.submat(3,0,5,2) = arma::eye<arma::mat>(3,3);

	gamma = dt * gamma;
	return gamma;
}

arma::vec Wrapper::sigma_dot_wrapper(double t, arma::vec X, Args * args) {


	arma::vec omega = args -> get_constant_omega();
	arma::vec attitude_set = {X(0), X(1), X(2), omega(0), omega(1), omega(2)};

	arma::vec sigma_dot =  RBK::dmrpdt(t, attitude_set );
	return sigma_dot;
}





arma::vec Wrapper::point_mass_dxdt_wrapper_body_frame(double t, arma::vec X, Args * args) {

	
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


arma::vec Wrapper::attitude_dxdt_wrapper(double t, arma::vec  X, Args * args) {

	arma::vec dxdt = RBK::dXattitudedt(t, X , args -> get_shape_model() -> get_inertia());

	return dxdt;

}


arma::vec Wrapper::validation_dxdt_wrapper(double t, arma::vec  X, Args * args) {

	arma::vec dxdt = {cos(t),sin(t)};

	return dxdt;

}

arma::vec Wrapper::event_function_mrp_omega(double t, arma::vec X, Args * args) {
	if (arma::norm(X.rows(0, 2)) > 1) {
		
		X.rows(0,2) = - X.rows(0, 2) / arma::dot(X . rows(0, 2), X . rows(0, 2));

		return X;
	}
	else {
		return X;
	}
}


arma::vec Wrapper::event_function_mrp(double t, arma::vec X, Args * args) {
	if (arma::norm(X.rows(0, 2)) > 1) {
		arma::vec mrp = - X.rows(0, 2) / arma::dot(X . rows(0, 2), X . rows(0, 2));
		return mrp;
	}
	else {
		return X;
	}
}

arma::vec Wrapper::event_function_collision(double t, arma::vec X, Args * args) {

	arma::vec mrp_TN = args -> get_interpolator() -> interpolate(t, true).rows(0, 2);

	args -> get_frame_graph() -> set_transform_mrp("N", "T", mrp_TN);
	arma::vec pos_inertial = X . rows(0, 2);
	arma::vec pos_body = args -> get_frame_graph() -> convert(pos_inertial, "N", "T");

	if (args -> get_shape_model() -> contains(pos_body.colptr(0))) {
		std::cout << " The spacecraft collided with the surface at time t = " << t << " s" << std::endl;
		args -> set_stopping_bool(true);
	}
	return X;
}


arma::vec Wrapper::event_function_collision_body_frame(double t, arma::vec X, Args * args) {


	arma::vec pos_body = X . rows(0, 2);

	if (args -> get_shape_model() -> contains(pos_body.colptr(0))) {
		std::cout << " The spacecraft collided with the surface at time t = " << t << " s" << std::endl;
		args -> set_stopping_bool(true);
	}
	return X;
}


double Wrapper::energy_attitude(double t, arma::vec X , Args * args) {

	arma::vec omega = X . rows(3, 5);

	return 0.5 * arma::dot(omega, args -> get_shape_model() -> get_inertia() * omega);

}

arma::vec Wrapper::joint_sb_spacecraft_body_frame_dyn(double t, arma::vec  X, Args * args){

	arma::vec dxdt(X.n_rows);

	arma::vec sigma = X.rows(0,3);
	arma::vec omega = X.rows(3,5);
	arma::vec pos = X.rows(6,8);
	arma::vec vel = X.rows(9,11);
	

	dxdt.rows(0,5) = attitude_dxdt_wrapper(t,X.rows(0,5),args);

	arma::vec omega_dot = dxdt.rows(3,5);

	arma::vec acc_sph = args -> get_dyn_analyses() -> spherical_harmo_acc(
		args -> get_degree(),
		args -> get_ref_radius(),
		args -> get_mu(),
		pos, 
		args -> get_Cnm(),
		args -> get_Snm());

	dxdt.rows(6,8) = X.rows(9,11);
	dxdt.rows(9,11) = (acc_sph - arma::cross(omega_dot,pos) - 2 * arma::cross(omega,vel)
		- arma::cross(omega,arma::cross(omega,pos)));



	

	return dxdt;

}

arma::vec Wrapper::obs_long_lat(double t,const arma::vec & x, const Args & args){

	arma::vec X = {1,0,0};
	arma::vec Y = {0,1,0};
	arma::vec Z = {0,0,1};
	arma::mat XY = {{1,0,0},
	{0,1,0},
	{0,0,0}};

	arma::vec long_lat(2);

	arma::vec coords_station = args.get_coords_station();
	arma::vec omega = args.get_constant_omega();
	arma::mat DCM = RBK::M1(coords_station(1)) * RBK::M3(coords_station(0) + t * omega(2)) ;
	arma::vec pos_station = {args.get_ref_radius(),0,0};
	arma::vec rho = x.rows(0,2) - DCM.t() * pos_station;

	long_lat(0) = std::atan2(arma::dot(Y, rho),
		arma::dot(X, rho));

	long_lat(1) = std::atan2(arma::dot(Z, rho),
		arma::dot(rho, XY * rho));

	return long_lat;
}

arma::mat Wrapper::obs_jac_long_lat(double t,const arma::vec & x, const Args & args){

	arma::vec X = {1,0,0};
	arma::vec Y = {0,1,0};
	arma::vec Z = {0,0,1};
	arma::mat XY = {{1,0,0},
	{0,1,0},
	{0,0,0}};


	arma::vec coords_station = args.get_coords_station();
	arma::vec omega = args.get_constant_omega();
	arma::mat DCM = RBK::M1(coords_station(1)) * RBK::M3(coords_station(0) + t * omega(2)) ;
	arma::vec pos_station = {args.get_ref_radius(),0,0};
	arma::vec rho = x.rows(0,2) - DCM.t() * pos_station;

	arma::mat jacobian = arma::zeros<arma::mat>(2,x.n_rows);

	jacobian.row(0).cols(0,2) = (rho.t() * (X * Y.t() - Y * X.t())
		/arma::dot(rho, XY * rho));

	jacobian.row(1).cols(0,2) = std::sqrt(arma::dot(rho, XY * rho)) / arma::dot(rho,rho) * Z.t() * (arma::eye<arma::mat>(3,3)
		- rho * rho.t() * XY / arma::dot(rho, XY * rho));

	return jacobian;

}



