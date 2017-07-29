#include "RK.hpp"



RK::RK(arma::vec X0,
       double t0,
       double tf,
       double dt,
       Args * args,
       bool check_energy_conservation,
       std::string title
      ) {

	this -> X0 = X0;
	this -> t0 = t0;
	this -> tf = tf;
	this -> dt = dt;
	this -> args = args;
	this -> check_energy_conservation = check_energy_conservation;
	this -> title = title;

}

arma::vec * RK::get_T() {
	return (&this -> T);
};

arma::mat * RK::get_X() {
	return (&this -> X);
};

RK4::RK4(
    arma::vec X0,
    double t0,
    double tf,
    double dt,
    Args * args,
    bool check_energy_conservation,
    std::string title) : RK(X0, t0, tf, dt, args, check_energy_conservation, title) {

	unsigned int n_times = (unsigned int)((tf - t0) / dt) + 2;
	this -> T = arma::vec(n_times);
	this -> X = arma::mat(X0.n_rows, n_times);

	// The vector of times is populated
	this -> T(0) = t0;
	for (unsigned int i = 1; i < n_times - 1; ++i ) {
		this -> T(i) = t0 + dt * i;
	}
	this -> T(n_times - 1) = tf;

	// The first (initial) state is inserted
	this -> X.col(0) = X0;

	this -> energy = arma::vec(n_times);
}


RK45::RK45(
    arma::vec X0,
    double t0,
    double tf,
    double dt,
    Args * args,
    bool check_energy_conservation,
    std::string title,
    double tol) : RK(X0, t0, tf, dt, args, check_energy_conservation, title) {
	this -> tol = tol;

}


void RK4::run(arma::vec (*dXdt)(double, arma::vec , Args * args),
              double (*energy_fun)(double t, arma::vec , Args * args),
              arma::vec (*event_function)(double t, arma::vec, Args *),
              std::string savepath) {


	if (this -> check_energy_conservation == true) {

		this -> energy(0) = (*energy_fun)(this -> T(0), this -> X . col(0), args);
	}

	for (unsigned int i = 1; i < this -> T . n_rows; ++i) {
		double dt = this -> T(i) - this -> T(i - 1);

		arma::vec k1 = (*dXdt)(this -> T(i - 1), this -> X . col(i - 1), this -> args);
		arma::vec k2 = (*dXdt)(this -> T(i - 1) + dt / 2, this -> X . col(i - 1) + k1 * dt / 2, this -> args);
		arma::vec k3 = (*dXdt)(this -> T(i - 1) + dt / 2, this -> X . col(i - 1) + k2 * dt / 2, this -> args);
		arma::vec k4 = (*dXdt)(this -> T(i - 1) + dt , this -> X . col(i - 1) + k3 * dt , this -> args);
		this -> X . col(i) = this -> X . col(i - 1) + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);

		// Applying event function to state if need be
		if (event_function != nullptr) {
			this -> X . col(i) = (*event_function)(this -> T(i), this -> X . col(i), this -> args);
		}

		if (check_energy_conservation == true) {
			this -> energy(i) = (*energy_fun)(this -> T(i), this -> X . col(i), this -> args);
		}

		if (this -> args -> get_stopping_bool() == true) {
			break;
		}

	}

	if (check_energy_conservation == true) {
		energy.save(savepath + "energy_RK4_" + this -> title + ".txt", arma::raw_ascii);
	}
	if (savepath != "") {
		this -> T.save(savepath + "T_RK4_" + this -> title + ".txt", arma::raw_ascii);
		this -> X.save(savepath + "X_RK4_" + this -> title + ".txt", arma::raw_ascii);
	}

}

void RK45::run(arma::vec (*dXdt)(double, arma::vec , Args * args),
               double (*energy_fun)(double, arma::vec, Args * args),
               arma::vec (*event_function)(double t, arma::vec, Args *),
               bool verbose,
               std::string savepath) {


	std::vector<double> T_v;
	std::vector<arma::vec> X_v;
	std::vector<double> energy_v;


	T_v.push_back(this -> t0);
	X_v.push_back(this -> X0);


	if (this -> check_energy_conservation == true) {

		energy_v.push_back((*energy_fun)(this -> t0, X_v[0], args));
	}

	while (T_v[T_v.size() - 1 ] < this -> tf && this -> args -> get_stopping_bool() == false) {

		double tk = T_v[T_v.size() - 1 ];
		arma::vec yk = X_v[X_v.size() - 1];

		// times
		double tk1 = tk;
		double tk2 = tk + 1. / 4. * this -> dt;
		double tk3 = tk + 3. / 8. * this -> dt;
		double tk4 = tk + 12. / 13. * this -> dt;
		double tk5 = tk + this -> dt;
		double tk6 = tk + 1. / 2. * this -> dt;

		// ks
		arma::vec k1 = (*dXdt)(tk1, yk, this -> args);
		arma::vec k2 = (*dXdt)(tk2 , yk + k1 * this -> dt / 4, this -> args);
		arma::vec k3 = (*dXdt)(tk3, yk + k1 * this -> dt * 3. / 32. + k2 * this -> dt * 9. / 32., this -> args);
		arma::vec k4 = (*dXdt)(tk4, yk + k1 * this -> dt * 1932. / 2197. - k2 * this -> dt * 7200. / 2197. + k3 * this -> dt * 7296. / 2197., this -> args);
		arma::vec k5 = (*dXdt)(tk5 , yk + k1 * this -> dt * 439. / 216. - k2 * this -> dt * 8. + k3 * this -> dt * 3680. / 513. - k4 * this -> dt * 845. / 4104., this -> args);
		arma::vec k6 = (*dXdt)(tk6 , yk - k1 * this -> dt * 8. / 27. + 2 * k2 * this -> dt - k3 * this -> dt * 3544. / 2565. + k4 * this -> dt * 1859. / 4104. - k5 * this -> dt * 11. / 40., this -> args);


		// Solutions
		arma::vec y_order_4 = yk + this -> dt * (25. / 216 * k1 + 1408. / 2565 * k3 + 2197. / 4101 * k4 - 1. / 5. * k5);
		arma::vec y_order_5 = yk + this -> dt * (16. / 135 * k1 + 6656. / 12825 * k3 + 28561. / 56430 * k4 - 9. / 50. * k5 + 2. / 55. * k6);

		X_v .push_back (y_order_5);
		T_v.push_back(tk5);

		// Timestep update
		this -> dt = std::pow(this -> tol / (2 * arma::norm(y_order_5 - y_order_4)), 0.25) * this -> dt;

		// Minimum timestep
		this -> dt = std::min(this -> dt, (this -> tf - this -> t0) / 100);

		// Applying event function to state if need be

		if (event_function != nullptr) {
			X_v[X_v.size() - 1] = (*event_function)( tk5, X_v[X_v.size() - 1], this -> args);
		}

		if (check_energy_conservation == true) {
			energy_v.push_back((*energy_fun)(tk5, y_order_5, this -> args));
		}

		if (verbose == true) {
			double percentage = std::abs((tk - this -> t0) / (this -> tf - this -> t0)) * 100 ;
			std::cout << "Completion: " << percentage << " %" << std::endl;
		}

	}



	this -> T = arma::vec(T_v.size());
	this -> X = arma::mat(this -> X0.n_rows, T_v.size());
	arma::vec energy = arma::vec(T_v.size());



	for (unsigned int i = 0; i < this -> T . n_rows; ++i) {
		this -> T(i) = T_v[i];
		this -> X.col(i) = X_v[i];


		if (check_energy_conservation == true) {
			if (this -> args -> get_stopping_bool() == false) {
				energy(i) = energy_v[i];
			}
			else {
				if (i < this -> T.n_rows - 1) {
					energy(i) = energy_v[i];
				}
				else {
					energy(i) = energy(i - 1);
				}
			}
		}

	}

	// Extrapolation "backwards" if need be
	if (this -> args -> get_stopping_bool() == false) {

		Interpolator interpolator(&this -> T, &this -> X);

		this -> X.col(this -> T.n_rows - 1) = interpolator . interpolate(this -> tf, this -> args -> get_is_attitude_bool());

		this -> T(this -> T.n_rows - 1) = this -> tf;
	}


	if (check_energy_conservation == true) {
		energy.save(savepath + "energy_RK45_" + this -> title + ".txt", arma::raw_ascii);
	}
	if (savepath != "") {

		this -> T.save(savepath + "T_RK45_" + this -> title + ".txt", arma::raw_ascii);
		this -> X.save(savepath + "X_RK45_" + this -> title + ".txt", arma::raw_ascii);
	}




}