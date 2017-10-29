#include "RK.hpp"
#include <chrono>

RK::RK(arma::vec X0,
       double t0,
       double tf,
       double dt,
       Args * args,
       std::string title
      ) {

	this -> X0 = X0;
	this -> t0 = t0;
	this -> tf = tf;
	this -> dt = dt;
	this -> args = args;
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
    std::string title) : RK(X0, t0, tf, dt, args, title) {

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

}


RK45::RK45(
    arma::vec X0,
    double t0,
    double tf,
    double dt,
    Args * args,
    std::string title,
    double tol) : RK(X0, t0, tf, dt, args, title) {
	this -> tol = tol;

}


void RK4::run(arma::vec (*dXdt)(double, arma::vec , Args * args),
              arma::vec (*event_function)(double t, arma::vec, Args *),
              std::string savepath) {


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

		if (this -> args -> get_stopping_bool() == true) {
			break;
		}

	}

	if (savepath != "") {
		this -> T.save(savepath + "T_RK4_" + this -> title + ".txt", arma::raw_ascii);
		this -> X.save(savepath + "X_RK4_" + this -> title + ".txt", arma::raw_ascii);
	}

}



void RK45::propagate(
	    arma::vec (*dXdt)(double, arma::vec, Args *),
	    const double t,
	    const arma::vec & y,
	    arma::vec & y_order_4,
	    arma::vec & y_order_5,
	    arma::mat & K){

		// times
		double tk0 = t;
		double tk1 = t + 1. / 4. * this -> dt;
		double tk2 = t + 3. / 8. * this -> dt;
		double tk3 = t + 12. / 13. * this -> dt;
		double tk4 = t + this -> dt;
		double tk5 = t + 1. / 2. * this -> dt;

		// K
		K.col(0) = (*dXdt)(tk0, y, this -> args);
		K.col(1) = (*dXdt)(tk1 , y + K.col(0) * this -> dt / 4, this -> args);
		K.col(2) = (*dXdt)(tk2, y + K.col(0) * this -> dt * 3. / 32. + K.col(1) * this -> dt * 9. / 32., this -> args);
		K.col(3) = (*dXdt)(tk3, y + K.col(0) * this -> dt * 1932. / 2197. - K.col(1) * this -> dt * 7200. / 2197. + K.col(2) * this -> dt * 7296. / 2197., this -> args);
		K.col(4) = (*dXdt)(tk4 , y + K.col(0) * this -> dt * 439. / 216. - K.col(1) * this -> dt * 8. + K.col(2) * this -> dt * 3680. / 513. - K.col(3) * this -> dt * 845. / 4104., this -> args);
		K.col(5) = (*dXdt)(tk5 , y - K.col(0) * this -> dt * 8. / 27. + 2 * K.col(1) * this -> dt - K.col(2) * this -> dt * 3544. / 2565. + K.col(3) * this -> dt * 1859. / 4104. - K.col(4) * this -> dt * 11. / 40., this -> args);

		// Solutions
		y_order_4 = y + this -> dt * (25. / 216 * K.col(0) + 1408. / 2565 * K.col(2) + 2197. / 4104 * K.col(3) - 1. / 5. * K.col(4));
		y_order_5 = y + this -> dt * (16. / 135 * K.col(0) + 6656. / 12825 * K.col(2) + 28561. / 56430 * K.col(3) - 9. / 50. * K.col(4) + 2. / 55. * K.col(5));

}

void RK45::run(arma::vec (*dXdt)(double, arma::vec , Args * args),
               arma::vec (*event_function)(double t, arma::vec, Args *),
               bool verbose,
               std::string savepath) {


	std::vector<double> T_v;
	std::vector<arma::vec> X_v;

	T_v.push_back(this -> t0);
	X_v.push_back(this -> X0);

	arma::vec y_order_4 = arma::zeros<arma::vec>(this -> X0.n_rows);
	arma::vec y_order_5 = arma::zeros<arma::vec>(this -> X0.n_rows);

	arma::mat K = arma::zeros<arma::mat>(this -> X0.n_rows,6);
	arma::vec yk = arma::zeros<arma::vec>(this -> X0.n_rows);

	// The time step guess is refined
	bool time_step_guess_refined = false;


	while(!time_step_guess_refined){
		
		this -> propagate(dXdt,T_v.back(),X_v.back(),y_order_4,y_order_5,K);
		
		double norm_diff = arma::norm(y_order_5 - y_order_4);
				
		// Timestep update
		double factor = std::pow(this -> tol * this -> dt / (2 * norm_diff),0.25);
		if (norm_diff < 1e-10){
			factor = 2;
		}
		if (factor < 0.1){
			factor = 0.1;
		}
		else if (factor > 2){
			factor = 2;
		}
		
		if (abs(1 - factor) < 1e-1 ){
			time_step_guess_refined = true;
		}

		else{
			this -> dt = this -> dt * factor;
		}

	}
	if (verbose){
		std::cout << " Initial dt: " << this -> dt << std::endl;
	}

	while (T_v.back() < this -> tf && this -> args -> get_stopping_bool() == false) {

		this -> propagate(dXdt,T_v.back(),X_v.back(),y_order_4,y_order_5,K);
		double norm_diff = arma::norm(y_order_5 - y_order_4);

		X_v.push_back(y_order_5);
		T_v.push_back(T_v.back() + this -> dt);

		// Timestep update
		double factor;
		if (norm_diff < 1e-10){
			factor = 2;
		}
		else{
			factor = std::pow(this -> tol * this -> dt / (2 * norm_diff),0.25);
		}

		if (factor < 0.1){
			factor = 0.1;
		}
		else if (factor > 2){
			factor = 2;
		}

		this -> dt = factor * this -> dt;
		
		// Applying event function to state if need be
		if (event_function != nullptr) {
			X_v.back() = (*event_function)( T_v.back(), X_v.back(), this -> args);
		}

	}

	this -> T = arma::vec(T_v.size());
	this -> X = arma::mat(this -> X0.n_rows, T_v.size());

	for (unsigned int i = 0; i < this -> T . n_rows; ++i) {
		this -> T(i) = T_v[i];
		this -> X.col(i) = X_v[i];
	}


	// The last value is obtained from an RK4 instantiated with a fraction of the last current timestep
	if (this -> args -> get_stopping_bool() == false) {

		RK4 rk_4(this -> X.col(this -> T.n_rows - 2),
	                 this -> T(this -> T.n_rows - 2),
	                 this -> tf,
	                 this -> dt / 5,
	                 this -> args,
	                 "");

		rk_4.run(dXdt,
	                nullptr,
	                "");

		this -> X.col(this -> T.n_rows - 1) = rk_4.get_X() -> tail_cols( 1 );
		this -> T(this -> T.n_rows - 1) = this -> tf;
	}

	if (savepath != "") {
		this -> T.save(savepath + "T_RK45_" + this -> title + ".txt", arma::raw_ascii);
		this -> X.save(savepath + "X_RK45_" + this -> title + ".txt", arma::raw_ascii);
	}

}