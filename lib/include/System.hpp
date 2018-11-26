#ifndef HEADER_SYSTEM
#define HEADER_SYSTEM

#include <armadillo>
#include "FixVectorSize.hpp"
#include "Args.hpp"

#define OPERATOR_DEBUG 0
#define OPERATOR_DEBUG_ESTIMATED_STATE 0



class System {
public:

	System(const Args & args,
		unsigned int N_est,
		arma::vec (*estimate_dynamics_fun)(double, const arma::vec & , const Args & args) ,
		arma::mat (*jacobian_estimate_dynamics_fun)(double, const arma::vec & , const Args & args),
		unsigned int N_true = 0,
		arma::vec (*true_dynamics_fun)(double, const arma::vec & , const Args & args) = nullptr) 
	: N_est(N_est), N_true(N_true){
		
		this -> estimate_dynamics_fun = estimate_dynamics_fun;
		this -> true_dynamics_fun = true_dynamics_fun;
		this -> jacobian_estimate_dynamics_fun = jacobian_estimate_dynamics_fun;
		this -> args = args;

	}

	System(const Args & args,
		unsigned int N_true,
		arma::vec (*true_dynamics_fun)(double, const arma::vec & , const Args & args)) 
	: N_est(0), N_true(N_true){
		this -> true_dynamics_fun = true_dynamics_fun;
		this -> args = args;
	}

	void operator() (const arma::vec & X , arma::vec & dxdt , const double t ){
		


		#if OPERATOR_DEBUG
		std::cout << "in System::operator\n";
		#endif

		if (this -> N_est == 0){

			arma::vec derivative =  this -> true_dynamics_fun(t,X,args);
			dxdt = derivative;

		}

		else{

			if (this -> true_dynamics_fun != nullptr){


				arma::vec derivative = this -> true_dynamics_fun(t,

					X.subvec(this -> N_est + this -> N_est * this -> N_est,
						this -> N_est + this -> N_est * this -> N_est + this -> N_true - 1),args);

				dxdt.subvec(this -> N_est + this -> N_est * this -> N_est,
					this -> N_est + this -> N_est * this -> N_est + this -> N_true - 1) = derivative;



			}	

			if (this -> estimate_dynamics_fun != nullptr){

				#if OPERATOR_DEBUG_ESTIMATED_STATE
				std::cout << "in operator() with this -> estimate_dynamics_fun != nullptr\n";
				std::cout << "Number of estimated states: " << this -> N_est << std::endl;
				#endif

				arma::vec X_spc_estimated = arma::vec(this -> N_est);

				for (unsigned int i = 0; i < X_spc_estimated.n_rows; ++i){
					X_spc_estimated(i) = X(i);
				}
				
				arma::mat Phi = arma::reshape(X.subvec(this -> N_est,
					this -> N_est + this -> N_est * this -> N_est - 1), 
				this -> N_est, this -> N_est );
				

				#if OPERATOR_DEBUG_ESTIMATED_STATE
				std::cout << "built stm" << std::endl;
				#endif

				arma::mat A = this -> jacobian_estimate_dynamics_fun(t,
					X_spc_estimated,this -> args);

				#if OPERATOR_DEBUG_ESTIMATED_STATE
				std::cout << "built A matrix" << std::endl;
				#endif

				arma::vec derivative = this -> estimate_dynamics_fun(t,
					X_spc_estimated,this -> args);

				#if OPERATOR_DEBUG_ESTIMATED_STATE
				std::cout << "computed state derivative" << std::endl;
				#endif

				dxdt.subvec(0,this -> N_est - 1) = derivative;
				dxdt.subvec(this -> N_est,
					this -> N_est + this -> N_est * this -> N_est - 1) = arma::vectorise(A * Phi);
				#if OPERATOR_DEBUG_ESTIMATED_STATE
				std::cout << "leaving operator() with this -> estimate_dynamics_fun != nullptr\n";
				#endif
			}


		}
	}




protected:
	const unsigned int N_est;
	const unsigned int N_true;
	arma::vec (*estimate_dynamics_fun)(double, const arma::vec &  , const Args & args) = nullptr;
	arma::vec (*true_dynamics_fun)(double, const arma::vec &  , const Args & args) = nullptr;
	arma::mat (*jacobian_estimate_dynamics_fun)(double, const arma::vec &  , const Args & args) = nullptr;
	Args args;
};



	#endif