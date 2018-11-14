#include "StatePropagator.hpp"




void StatePropagator::propagateOrbit(std::vector<double> & T, std::vector<arma::vec> & X_augmented, 
	const double t0, const double tf, const double dt , arma::vec  initial_state,
	arma::vec (*dynamics_fun)(double, const arma::vec & , const Args & args),const Args & args,
	std::string savefolder,std::string label){

	int N_times = (int)(std::abs(tf - t0) / dt);

	if (N_times <= 0){
		throw(std::runtime_error("Cannot have a negative number of steps"));
	}

	arma::vec times = arma::linspace<arma::vec>(t0, tf,N_times); 

	for (unsigned int i = 0; i < times.n_rows; ++i){
		T.push_back(times(i));
	}

	StatePropagator::propagate(T, X_augmented,initial_state,dynamics_fun,args);

	if (savefolder.length() > 0){

		arma::mat X_arma(initial_state.n_rows,X_augmented.size());

		for (unsigned int i = 0; i < X_augmented.size(); ++i){
			X_arma.col(i) = X_augmented[i];
		}

		X_arma.save(savefolder +"state_" + label + ".txt",arma::raw_ascii);
		times.save(savefolder +"time_" + label + ".txt",arma::raw_ascii);

	}

}







void StatePropagator::propagateOrbit(std::vector<double> & T, std::vector<arma::vec> & X_augmented, 
	const double t0, const double dt, int N_times, arma::vec initial_state,
	arma::vec (*dynamics_fun)(double, const arma::vec & , const Args & args),const Args & args,
	std::string savefolder ,std::string label ){
	
	if (N_times <= 0){
		throw(std::runtime_error("Cannot have a negative number of steps"));
	}

	arma::vec times = arma::regspace<arma::vec>(t0, dt, t0 + dt * (N_times - 1)); 

	for (unsigned int i = 0; i < times.n_rows; ++i){
		T.push_back(times(i));
	}

	StatePropagator::propagate(T, X_augmented,initial_state,dynamics_fun,args);

	if (savefolder.length() > 0){

		arma::mat X_arma(initial_state.n_rows,X_augmented.size());

		for (unsigned int i = 0; i < X_augmented.size(); ++i){
			X_arma.col(i) = X_augmented[i];
		}

		X_arma.save(savefolder +"state_" + label + ".txt",arma::raw_ascii);
		times.save(savefolder +"time_" + label + ".txt",arma::raw_ascii);

	}

}

void StatePropagator::propagateOrbit( const double t0, const double tf, const double dt, arma::vec initial_state,
	arma::vec (*dynamics_fun)(double, const arma::vec & , const Args & args),const Args & args,
	std::string savefolder ,std::string label ){



	std::vector<double> T;
	std::vector<arma::vec> X_augmented;

	int N_times = (int)(std::abs(tf - t0) / dt);

	if (N_times <= 0){
		throw(std::runtime_error("Cannot have a negative number of steps"));
	}

	arma::vec times = arma::linspace<arma::vec>(t0, tf,N_times); 

	for (unsigned int i = 0; i < times.n_rows; ++i){
		T.push_back(times(i));
	}

	StatePropagator::propagate(T, X_augmented,initial_state,dynamics_fun,args);

	if (savefolder.length() > 0){

		arma::mat X_arma(initial_state.n_rows,X_augmented.size());

		for (unsigned int i = 0; i < X_augmented.size(); ++i){
			X_arma.col(i) = X_augmented[i];
		}

		X_arma.save(savefolder +"state_" + label + ".txt",arma::raw_ascii);
		times.save(savefolder +"time_" + label + ".txt",arma::raw_ascii);

	}








}


void StatePropagator::propagateOrbit( const double t0, const double dt, int N_times, arma::vec initial_state,
	arma::vec (*dynamics_fun)(double, const arma::vec & , const Args & args),const Args & args,
	std::string savefolder ,std::string label ){

	std::vector<double> T;
	std::vector<arma::vec> X_augmented;

	if (N_times <= 0){
		throw(std::runtime_error("Cannot have a negative number of steps"));
	}

	arma::vec times = arma::regspace<arma::vec>(t0, dt, t0 + dt * (N_times - 1)); 
	

	for (unsigned int i = 0; i < times.n_rows; ++i){
		T.push_back(times(i));
	}

	StatePropagator::propagate(T, X_augmented,initial_state,dynamics_fun,args);

	if (savefolder.length() > 0){

		arma::mat X_arma(initial_state.n_rows,X_augmented.size());

		for (unsigned int i = 0; i < X_augmented.size(); ++i){
			X_arma.col(i) = X_augmented[i];
		}

		X_arma.save(savefolder +"state_" + label + ".txt",arma::raw_ascii);
		times.save(savefolder +"time_" + label + ".txt",arma::raw_ascii);

	}








}


void StatePropagator::propagate(std::vector<double> & T, std::vector<arma::vec> & X_augmented,arma::vec initial_state,
	arma::vec (*dynamics_fun)(double, const arma::vec & , const Args & args),const Args & args){

	System dynamics(args,initial_state.n_rows,dynamics_fun);

	typedef boost::numeric::odeint::runge_kutta_cash_karp54< arma::vec > error_stepper_type;
	auto stepper = boost::numeric::odeint::make_controlled<error_stepper_type>( 1.0e-10 , 1.0e-16 );

	std::cout << "In StatePropagator::propagate\n";
	boost::numeric::odeint::integrate_times(stepper, 
		dynamics, 
		initial_state,
		T.begin(), 
		T.end(),
		1e-3,
		Observer::push_back_augmented_state(X_augmented));



}
