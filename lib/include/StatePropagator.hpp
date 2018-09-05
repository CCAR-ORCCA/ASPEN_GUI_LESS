#ifndef HEADER_STATEPROPAGATOR
#define HEADER_STATEPROPAGATOR
#include <boost/numeric/odeint.hpp>
#include <Dynamics.hpp>
#include <System.hpp>
#include <Args.hpp>
#include <Observer.hpp>


class StatePropagator{

public:



	static void propagateOrbit(std::vector<double> & T, std::vector<arma::vec> & X_augmented, 
		const double t0, const double tf, const double dt, arma::vec initial_state,
		arma::vec (*dynamics_fun)(double, const arma::vec & , const Args & args),const Args & args,
		std::string savefolder = "",std::string label = "");



	static void propagateOrbit(std::vector<double> & T, std::vector<arma::vec> & X_augmented, 
		const double t0, const double dt, int N_times, arma::vec initial_state,
		arma::vec (*dynamics_fun)(double, const arma::vec & , const Args & args),const Args & args,
		std::string savefolder = "",std::string label = "");


	static void propagateOrbit( const double t0, const double tf, const double dt, arma::vec initial_state,
		arma::vec (*dynamics_fun)(double, const arma::vec & , const Args & args),const Args & args,
		std::string savefolder = "",std::string label = "");



	static void propagateOrbit( const double t0, const double dt, int N_times, arma::vec initial_state,
		arma::vec (*dynamics_fun)(double, const arma::vec & , const Args & args),const Args & args,
		std::string savefolder = "",std::string label = "");




private:


	static void propagate(std::vector<double> & T, std::vector<arma::vec> & X_augmented,arma::vec initial_state,
		arma::vec (*dynamics_fun)(double, const arma::vec & , const Args & args),const Args & args);


};










#endif