#include "RK4.hpp"

RK4::RK4(arma::vec & X0,
         double t0,
         double tf,
         double dt,
         arma::vec * T,
         arma::mat * X
        ) {


	unsigned int n_times = (unsigned int)((tf - t0) / dt) + 2;
	*T = arma::vec(n_times);
	*X = arma::mat(X0.n_rows, n_times);

	// The vector of times is populated
	(*T)(0) = t0;
	for (unsigned int i = 1; i < n_times - 1; ++i ) {
		(*T)(i) = t0 + dt * i;
	}
	(*T)(n_times - 1) = tf;

	// The first (initial) state is inserted
	(*X).col(0) = X0;


}



RK4_attitude::RK4_attitude(arma::vec & X0,
                           double t0,
                           double tf,
                           double dt,
                           arma::vec * T,
                           arma::mat * X,
                           arma::mat I,
                           arma::vec (*dXdt)(double, arma::vec,
                                   arma::mat &),
                           bool check_energy_conservation
                          ) : RK4( X0, t0, tf, dt, T, X) {

	arma::vec energy(T -> n_rows);
	if (check_energy_conservation == true) {

		arma::vec omega = (X -> col(0)).rows(3, 5);

		energy(0) = 0.5 * arma::dot(omega, I * omega);
	}

	for (unsigned int i = 1; i < T -> size(); ++i) {
		double dt = (*T)(i) - (*T)(i - 1);

		arma::vec k1 = (*dXdt)((*T)(i - 1), X -> col(i - 1), I);
		arma::vec k2 = (*dXdt)((*T)(i - 1) + dt / 2, X -> col(i - 1) + k1 * dt / 2, I);
		arma::vec k3 = (*dXdt)((*T)(i - 1) + dt / 2, X -> col(i - 1) + k2 * dt / 2, I);
		arma::vec k4 = (*dXdt)((*T)(i - 1) + dt , X -> col(i - 1) + k3 * dt , I);
		X -> col(i) = X -> col(i - 1) + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);

		// Switching to shadow set if need be
		double sigma = arma::norm(X -> col(i).rows(0, 2));
		if (sigma > 1) {
			X -> col(i).rows(0, 2) = - (X -> col(i).rows(0, 2)) / (sigma * sigma);
		}

		if (check_energy_conservation == true) {

			arma::vec omega = (X -> col(i)).rows(3, 5);

			energy(i) = 0.5 * arma::dot(omega, I * omega);
		}

	}

	if (check_energy_conservation == true) {
		energy.save("attitude_energy.txt", arma::raw_ascii);
	}




}


RK4_orbit::RK4_orbit(arma::vec & X0,
                     double t0,
                     double tf,
                     double dt,
                     arma::vec * T,
                     arma::mat * X,
                     double density,
                     ShapeModel * shape_model,
                     FrameGraph * frame_graph,
                     bool check_energy_conservation
                    ) : RK4( X0, t0, tf, dt, T, X) {


	// An instance of DynamicAnalyses is created to handle the computation of the acceleration
	DynamicAnalyses dyn_analyses(shape_model);

	arma::vec mrp_TN = {0, 0, 0};
	arma::vec energy(T -> n_rows);
	if (check_energy_conservation == true) {

		arma::vec pos_inertial = (X -> col(0)).rows(0, 2);
		arma::vec pos_body = frame_graph -> convert(pos_inertial, "N", "T");

		double potential = dyn_analyses.pgm_potential(pos_body.colptr(0) , density);

		energy(0) = 0.5 * arma::dot((X -> col(0)).rows(3, 5), (X -> col(0)).rows(3, 5)) - potential;
	}


	for (unsigned int i = 1; i < T -> size(); ++i) {
		double dt = (*T)(i) - (*T)(i - 1);

		// k1
		double tk1 = (*T)(i - 1);

		// INTERPOLATE mrp_TN to tk1 HERE

		frame_graph -> set_transform_mrp("N", "T", mrp_TN);
		arma::vec pos_tk1_inertial = (X -> col(i - 1)).rows(0, 2);
		arma::vec pos_tk1_body = frame_graph -> convert(pos_tk1_inertial, "N", "T");
		arma::vec acc_tk1_body = dyn_analyses.pgm_acceleration(pos_tk1_body.colptr(0) , density);
		arma::vec acc_tk1_inertial = frame_graph -> convert(acc_tk1_body, "T", "N", true);
		arma::vec k1(6);
		k1.rows(0, 2) = X -> col(i - 1).rows(3, 5);
		k1.rows(3, 5) = acc_tk1_inertial;


		// k2
		double tk2 = (*T)(i - 1) + dt / 2;

		// INTERPOLATE mrp_TN to tk2 HERE

		frame_graph -> set_transform_mrp("N", "T", mrp_TN);
		arma::vec pos_tk2_inertial = X -> col(i - 1).rows(0, 2) + k1 .rows(0, 2) * dt / 2;

		arma::vec pos_tk2_body = frame_graph -> convert(
		                             pos_tk2_inertial, "N", "T");
		arma::vec acc_tk2_body = dyn_analyses.pgm_acceleration(pos_tk2_body.colptr(0) , density);
		arma::vec acc_tk2_inertial = frame_graph -> convert(acc_tk2_body, "T", "N", true);
		arma::vec k2(6);
		k2.rows(0, 2) = X -> col(i - 1).rows(3, 5) + k1.rows(3, 5) * dt / 2;
		k2.rows(3, 5) = acc_tk2_inertial;

		// k3
		double tk3 = tk2;

		// INTERPOLATE mrp_TN to tk3 HERE

		frame_graph -> set_transform_mrp("N", "T", mrp_TN);
		arma::vec pos_tk3_inertial = X -> col(i - 1).rows(0, 2) + k2 .rows(0, 2) * dt / 2;
		arma::vec pos_tk3_body = frame_graph -> convert(
		                             pos_tk3_inertial, "N", "T");
		arma::vec acc_tk3_body = dyn_analyses.pgm_acceleration(pos_tk3_body.colptr(0) , density);
		arma::vec acc_tk3_inertial = frame_graph -> convert(acc_tk3_body, "T", "N", true);
		arma::vec k3(6);
		k3.rows(0, 2) = X -> col(i - 1).rows(3, 5) + k2.rows(3, 5) * dt / 2;
		k3.rows(3, 5) = acc_tk3_inertial;


		// k4
		double tk4 = (*T)(i - 1) + dt ;

		// INTERPOLATE mrp_TN to tk4 HERE

		frame_graph -> set_transform_mrp("N", "T", mrp_TN);
		arma::vec pos_tk4_inertial = X -> col(i - 1).rows(0, 2) + k3.rows(0, 2) * dt ;
		arma::vec pos_tk4_body = frame_graph -> convert(
		                             pos_tk4_inertial, "N", "T");
		arma::vec acc_tk4_body = dyn_analyses.pgm_acceleration(pos_tk4_body.colptr(0) , density);
		arma::vec acc_tk4_inertial = frame_graph -> convert(acc_tk4_body, "T", "N", true);

		arma::vec k4(6);
		k4.rows(0, 2) = X -> col(i - 1).rows(3, 5) + k3.rows(3, 5) * dt ;
		k4.rows(3, 5) = acc_tk4_inertial;

		// Sum the contributions to obtain the new state
		X -> col(i) = X -> col(i - 1) + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);


		// Check if the spacecraft has collected with the surface
		if (shape_model -> contains(X -> col(i).colptr(0))) {
			std::cout << "The spacecraft collided with the surface at t = " << (*T)(i - 1) << " s" << std::endl;
			break;
		}

		// Check conservation of energy
		if (check_energy_conservation == true) {
			arma::vec pos_inertial = (X -> col(i)).rows(0, 2);
			arma::vec pos_body = frame_graph -> convert(pos_inertial, "N", "T");

			double potential = dyn_analyses.pgm_potential(pos_body.colptr(0) , density);
			energy(i) = 0.5 * arma::dot((X -> col(i)).rows(3, 5), (X -> col(i)).rows(3, 5)) - potential;

		}


	}

	if (check_energy_conservation == true) {
		energy.save("orbit_energy.txt", arma::raw_ascii);
	}



}