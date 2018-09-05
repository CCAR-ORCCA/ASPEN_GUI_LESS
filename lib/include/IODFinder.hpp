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

	IODFinder(std::vector<RigidTransform> * sequential_rigid_transforms, 
		std::vector<RigidTransform> *  absolute_rigid_transforms, 
		std::vector<arma::vec> mrps_LN,
		double stdev_Xtilde,
		double stdev_sigmatilde,
		int N_iter, 
		int particles);

	static double cost_function(arma::vec particle, std::vector<RigidTransform> * args,int verbose_level = 0);
	static double cost_function_cartesian(arma::vec particle, std::vector<RigidTransform> * args,int verbose_level = 0);

	void run( arma::vec lower_bounds = {}, arma::vec upper_bounds = {},
		std::string type = "keplerian", 
		int verbose_level = 0,const arma::vec & guess  = {});
	
	OC::KepState get_result() const;



	void run_batch(arma::vec & state,
		arma::mat & cov);

	static void debug_rigid_transforms();


	arma::mat::fixed<6,12> compute_dIprime_k_dVtilde_k(int k) const;

	arma::sp_mat compute_partial_V_partial_T() const;
	arma::sp_mat compute_partial_I_partial_V() const;

	arma::sp_mat compute_partial_y_partial_I(const std::vector<arma::vec::fixed<3>> & positions) const;
	arma::sp_mat compute_partial_y_partial_T(const std::vector<arma::vec::fixed<3>> & positions) const;


	void debug_R() const;

	static void debug_stms(const std::vector<RigidTransform> * rigid_transforms);


	static void seq_transform_from_epoch_transform(int k, RigidTransform & seq_transform_k,
		const RigidTransform & epoch_transform_k,
		const RigidTransform & epoch_transform_km1, 
		const std::vector<arma::vec> & mrps_LN);

	void compute_W(const std::vector<arma::vec::fixed<3>> & positions);


protected:


	void build_normal_equations(
		arma::mat & info_mat,
		arma::vec & normal_mat,
		arma::vec & residual_vector,
		arma::vec & apriori_state,
		const std::vector<arma::vec::fixed<3>> & positions,
		const std::vector<arma::mat> & stms) const;

	static arma::mat::fixed<3,6> compute_dsigmatilde_kdZ_k(
		const arma::mat::fixed<3,3> & M_k_p,
		const arma::mat::fixed<3,3> & M_k_tilde_bar,
		const arma::mat::fixed<3,3> & M_km1_tilde_bar,
		const arma::mat::fixed<3,3> & LN_k,
		const arma::mat::fixed<3,3> & LN_km1);


	arma::mat::fixed<3,6> compute_J_k(int k,const std::vector<arma::vec::fixed<3>> & positions) const;


	static arma::mat::fixed<3,7> compute_H_k(const arma::mat & Phi_k, 
		const arma::mat & Phi_kp1, 
		const arma::mat::fixed<3,3> & M_kp1_tilde_bar);

	static arma::vec::fixed<3> compute_y_k(
		const arma::vec::fixed<3> & CL_k_bar,
		const arma::vec::fixed<3> & CL_kp1_bar,
		const arma::mat::fixed<3,3> & M_kp1_prime_bar,
		const arma::vec::fixed<3> & X_kp1_prime_bar);


	static void compare_rigid_transforms(std::vector<RigidTransform> * s1,std::vector<RigidTransform> * s2);


	void compute_state_stms(const arma::vec::fixed<7> & X_hat,
		std::vector<arma::vec::fixed<3> > & positions,
		std::vector<arma::mat> & stms) const;

	void compute_P_T();


	int particles;
	int N_iter;
	std::vector<arma::vec> mrps_LN;

	std::vector<arma::mat> rigid_transforms_covariances;
	OC::KepState keplerian_state_at_epoch;

	arma::mat W;
	double stdev_sigmatilde;
	double stdev_Xtilde;

	arma::mat P_T;

	std::vector<RigidTransform> * absolute_rigid_transforms;
	std::vector<RigidTransform> * sequential_rigid_transforms;


};




#endif