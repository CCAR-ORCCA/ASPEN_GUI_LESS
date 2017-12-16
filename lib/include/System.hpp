#ifndef HEADER_SYSTEM
#define HEADER_SYSTEM

#include <armadillo>
#include "FixVectorSize.hpp"

template <typename state_type,typename jacobian_type = arma::mat> class System {
public:

	System(const Args & args,
		unsigned int N_est,
		state_type (*estimate_dynamics_fun)(double, const state_type & , const Args & args) ,
		jacobian_type (*jacobian_estimate_dynamics_fun)(double, const state_type & , const Args & args),
		unsigned int N_true = 0,
		state_type (*true_dynamics_fun)(double, const state_type & , const Args & args) = nullptr) 
	: N_est(N_est), N_true(N_true){
		
		this -> estimate_dynamics_fun = estimate_dynamics_fun;
		this -> true_dynamics_fun = true_dynamics_fun;
		this -> jacobian_estimate_dynamics_fun = jacobian_estimate_dynamics_fun;
		this -> args = args;

	}

	System(const Args & args,
		unsigned int N_true,
		state_type (*true_dynamics_fun)(double, const state_type & , const Args & args)) 
	: N_est(0), N_true(N_true){
		this -> true_dynamics_fun = true_dynamics_fun;
		this -> args = args;
	}

	void operator() (const state_type & x , state_type & dxdt , const double t ){


		if (this -> true_dynamics_fun != nullptr){

			dxdt.rows(this -> N_est + this -> N_est * this -> N_est,
				this -> N_est + this -> N_est * this -> N_est + this -> N_true - 1) = this -> true_dynamics_fun(t,
				
				x.rows(this -> N_est + this -> N_est * this -> N_est,
					this -> N_est + this -> N_est * this -> N_est + this -> N_true - 1),args);

			}	

			if (this -> estimate_dynamics_fun != nullptr){

				arma::mat Phi = arma::reshape(x.rows(this -> N_est,
					this -> N_est + this -> N_est * this -> N_est - 1), this -> N_est, this -> N_est );

				arma::mat A = this -> jacobian_estimate_dynamics_fun(t,x.rows(0,this -> N_est - 1),this -> args);

				dxdt.rows(0,this -> N_est - 1) = this -> estimate_dynamics_fun(t,x.rows(0,this -> N_est - 1),this -> args);
				dxdt.rows(this -> N_est,
					this -> N_est + this -> N_est * this -> N_est - 1) = arma::vectorise(A * Phi);

			}



		}

	protected:
		const unsigned int N_est;
		const unsigned int N_true;
		state_type (*estimate_dynamics_fun)(double, const state_type &  , const Args & args) = nullptr;
		state_type (*true_dynamics_fun)(double, const state_type &  , const Args & args) = nullptr;
		jacobian_type (*jacobian_estimate_dynamics_fun)(double, const state_type &  , const Args & args) = nullptr;
		Args args;
	};



	#endif