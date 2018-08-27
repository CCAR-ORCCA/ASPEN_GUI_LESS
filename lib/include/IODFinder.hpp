#ifndef HEADER_IOD_FINDER
#define HEADER_IOD_FINDER

#include <armadillo>
#include <OrbitConversions.hpp>

struct RigidTransform{

	arma::mat::fixed<3,3> M_k;
	arma::vec::fixed<3> X_k;
	double t_k;

};



class IODFinder{


public:

	IODFinder(std::vector<RigidTransform> * rigid_transforms, 
		int N_iter, 
		int particles);

	static double cost_function(arma::vec particle, std::vector<RigidTransform> * args,int verbose_level = 0);

	void run( arma::vec lower_bounds = {}, arma::vec upper_bounds = {},int verbose_level = 0,const arma::vec & guess  = {});
	OC::KepState get_result() const;


	static arma::mat::fixed<6,6> compute_P_I_prime_k(
		const arma::mat::fixed<12,12> P_V_tilde_k,
		const arma::mat::fixed<3,3> & M_k_tilde_bar,
		const arma::vec::fixed<3> & X_k_tilde_bar,
		const arma::mat::fixed<3,3> & M_km1_tilde_bar,
		const arma::vec::fixed<3> & X_km1_tilde_bar,
		const arma::mat::fixed<3,3> & LN_k,
		const arma::mat::fixed<3,3> & LN_km1
		);



	




	


protected:

	static arma::mat::fixed<6,12> compute_dIprime_k_dVtilde_k(
		const arma::mat::fixed<3,3> & M_k_tilde_bar,
		const arma::vec::fixed<3> & X_k_tilde_bar,
		const arma::mat::fixed<3,3> & M_km1_tilde_bar,
		const arma::vec::fixed<3> & X_km1_tilde_bar,
		const arma::mat::fixed<3,3> & LN_k,
		const arma::mat::fixed<3,3> & LN_km1);

	


	static arma::mat::fixed<6,12> compute_dsigmatilde_kdZ_k(
		const arma::mat::fixed<3,3> & M_k_tilde_bar,
		const arma::mat::fixed<3,3> & M_km1_tilde_bar,
		const arma::mat::fixed<3,3> & LN_k,
		const arma::mat::fixed<3,3> & LN_km1);


	static arma::mat::fixed<3,6> compute_J_k(const arma::mat::fixed<3,3> & M_kp1_tilde_bar,
		const arma::vec::fixed<3> CL_kp1_bar);


	static arma::mat compute_H_k(const arma::mat & Phi_k, const arma::mat::fixed<3,3> & M_kp1_tilde_bar);


	static arma::vec::fixed<3> compute_y_k(
		const arma::vec::fixed<3> & CL_k_bar,
		const arma::vec::fixed<3> & CL_kp1_bar,
		const arma::mat::fixed<3,3> & M_kp1_prime_bar,
		const arma::vec::fixed<3> & X_kp1_prime_bar);


	static arma::mat::fixed<3,3> compute_R_k(const arma::mat::fixed<3,3> & M_kp1_tilde_bar,
		const arma::vec::fixed<3> & CL_kp1_bar,
		const arma::mat::fixed<6,6> & P_I_k_prime);















	int particles;
	int N_iter;
	std::vector<RigidTransform> * rigid_transforms;
	OC::KepState keplerian_state_at_epoch;


};




#endif