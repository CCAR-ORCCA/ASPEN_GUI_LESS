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
		std::vector<arma::mat> * rigid_transforms_covariances,
		int N_iter, 
		int particles);

	static double cost_function(arma::vec particle, std::vector<RigidTransform> * args,int verbose_level = 0);
	static double cost_function_cartesian(arma::vec particle, std::vector<RigidTransform> * args,int verbose_level = 0);

	void run( arma::vec lower_bounds = {}, arma::vec upper_bounds = {},
		std::string type = "keplerian", 
		int verbose_level = 0,const arma::vec & guess  = {});
	OC::KepState get_result() const;


	static arma::mat::fixed<6,6> compute_P_I_prime_k(
		const arma::mat P_V_tilde_k,
		const arma::mat::fixed<3,3> & M_k_tilde_bar,
		const arma::vec::fixed<3> & X_k_tilde_bar,
		const arma::mat::fixed<3,3> & M_km1_tilde_bar,
		const arma::vec::fixed<3> & X_km1_tilde_bar,
		const arma::mat::fixed<3,3> & LN_k,
		const arma::mat::fixed<3,3> & LN_km1
		);





	void run_batch(arma::vec & state,arma::mat & cov);

	static void debug_rigid_transforms();


	static arma::mat compute_dIprime_k_dVtilde_k(
		const arma::mat::fixed<3,3> & M_k_tilde_bar,
		const arma::vec::fixed<3> & X_k_tilde_bar,
		const arma::mat::fixed<3,3> & M_km1_tilde_bar,
		const arma::vec::fixed<3> & X_km1_tilde_bar,
		const arma::mat::fixed<3,3> & LN_k,
		const arma::mat::fixed<3,3> & LN_km1);


	void debug_stms(const std::vector<RigidTransform> * rigid_transforms);


	static void seq_transform_from_epoch_transform(int k, RigidTransform & seq_transform_k,
		const RigidTransform & epoch_transform_k,
		const RigidTransform & epoch_transform_km1, 
		const std::vector<arma::vec> & mrps_LN);


protected:



	static void build_normal_equations(
		arma::mat & info_mat,
		arma::vec & normal_mat,
		arma::vec & residual_vector,
		const std::vector<RigidTransform> * rigid_transforms,
		const std::vector<arma::mat> * rigid_transforms_covariances,
		arma::vec & apriori_state,
		std::string dynamics_name);



	


	static arma::mat::fixed<3,6> compute_dsigmatilde_kdZ_k(
		const arma::mat::fixed<3,3> & M_k_tilde_bar,
		const arma::mat::fixed<3,3> & M_km1_tilde_bar,
		const arma::mat::fixed<3,3> & LN_k,
		const arma::mat::fixed<3,3> & LN_km1);


	static arma::mat::fixed<3,6> compute_J_k(const arma::mat::fixed<3,3> & M_kp1_tilde_bar,
		const arma::vec::fixed<3> CL_kp1_bar);


	static arma::mat::fixed<3,7> compute_H_k(const arma::mat & Phi_k, 
		const arma::mat & Phi_kp1, 
		const arma::mat::fixed<3,3> & M_kp1_tilde_bar);

	static arma::vec::fixed<3> compute_y_k(
		const arma::vec::fixed<3> & CL_k_bar,
		const arma::vec::fixed<3> & CL_kp1_bar,
		const arma::mat::fixed<3,3> & M_kp1_prime_bar,
		const arma::vec::fixed<3> & X_kp1_prime_bar);


	static arma::mat::fixed<3,3> compute_R_k(const arma::mat::fixed<3,3> & M_kp1_tilde_bar,
		const arma::vec::fixed<3> & CL_kp1_bar,
		const arma::mat::fixed<6,6> & P_I_k_prime);




	static void compute_stms(const arma::vec::fixed<7> & X_hat,
		const std::vector<RigidTransform> * rigid_transforms,
		std::vector<arma::mat> & stms);



	int particles;
	int N_iter;
	std::vector<RigidTransform> * rigid_transforms;
	std::vector<arma::mat> * rigid_transforms_covariances;



	OC::KepState keplerian_state_at_epoch;


};




#endif