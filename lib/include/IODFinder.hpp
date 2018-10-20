#ifndef HEADER_IOD_FINDER
#define HEADER_IOD_FINDER

#include <armadillo>
#include <OrbitConversions.hpp>

struct RigidTransform{

	arma::mat::fixed<3,3> M;
	arma::vec::fixed<3> X;
	int index_start;
	int index_end;
	double t_start;
	double t_end;

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

	IODFinder(std::vector<RigidTransform> * sequential_rigid_transforms,
		std::vector<RigidTransform> * absolute_rigid_transforms,  
		std::vector<arma::vec> mrps_LN,
		int N_iter, 
		int particles);

	static double cost_function(arma::vec particle, std::vector<RigidTransform> * args,int verbose_level = 0);
	static double cost_function_cartesian(arma::vec particle, std::vector<RigidTransform> * args,int verbose_level = 0);

	void run( arma::vec lower_bounds = {}, arma::vec upper_bounds = {},
		std::string type = "keplerian", 
		int verbose_level = 0,const arma::vec & guess  = {});
	
	arma::vec get_result() const;


	void run_batch(arma::vec & state,arma::mat & cov);
	void run_batch(arma::vec & state,arma::mat & cov,
		const std::map<int, arma::mat::fixed<6,6> > & R_pcs);

	static void debug_rigid_transforms();


	arma::mat::fixed<6,12> compute_dIprime_k_dVtilde_k(int k) const;

	arma::sp_mat compute_partial_V_partial_T() const;
	arma::sp_mat compute_partial_I_partial_V() const;

	arma::sp_mat compute_partial_y_partial_I(const std::vector<arma::vec::fixed<3>> & positions) const;
	arma::sp_mat compute_partial_y_partial_T(const std::vector<arma::vec::fixed<3>> & positions) const;


	void debug_R() const;
	static void debug_rp_partial();


	static void debug_stms(const std::vector<RigidTransform> * rigid_transforms);



	void compute_W(const std::vector<arma::vec::fixed<3>> & positions);

	static arma::rowvec::fixed<7> partial_rp_partial_state(const arma::vec::fixed<7> & state );
	
protected:


	void build_normal_equations(
		arma::mat & info_mat,
		arma::vec & normal_mat,
		arma::vec & residual_vector,
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



	static arma::rowvec::fixed<2> partial_rp_partial_ae(const double & a, const double & e);
	static arma::mat::fixed<2,4> partial_ae_partial_aevec(const double & a, const arma::vec::fixed<3> & e);
	static arma::rowvec::fixed<3> partial_a_partial_rvec(const double & a, const arma::vec::fixed<3> & r);
	static arma::rowvec::fixed<3> partial_a_partial_rdotvec(const double & a, const arma::vec::fixed<3> & r_dot, const double & mu);


	static arma::mat::fixed<3,3> partial_evec_partial_r(const arma::vec::fixed<7> & state);
	static arma::mat::fixed<3,3> partial_evec_partial_rdot(const arma::vec::fixed<7> & state);
	static arma::vec::fixed<3> partial_evec_partial_mu(const arma::vec::fixed<7> & state);

	static arma::mat::fixed<4,7> partial_aevec_partial_state(const double & a,const arma::vec::fixed<7> & state);

	static double partial_a_partial_mu(const double & a, const arma::vec::fixed<3> & r_dot, const double & mu);

	static arma::rowvec::fixed<7> partial_a_partial_state(const double & a,const arma::vec::fixed<7> & state );

	static void compare_rigid_transforms(std::vector<RigidTransform> * s1,std::vector<RigidTransform> * s2);


	static arma::mat::fixed<3,7> partial_evec_partial_state(const arma::vec::fixed<7> & state);


	void compute_state_stms(const arma::vec::fixed<7> & X_hat,
		std::vector<arma::vec::fixed<3> > & positions,
		std::vector<arma::mat> & stms) const;

	void compute_P_T();


	int particles;
	int N_iter;
	std::vector<arma::vec> mrps_LN;

	std::vector<arma::mat> rigid_transforms_covariances;
	arma::vec state_at_epoch;

	arma::mat W;
	double stdev_sigmatilde;
	double stdev_Xtilde;

	arma::mat P_T;

	std::vector<RigidTransform> * absolute_rigid_transforms;
	std::vector<RigidTransform> * sequential_rigid_transforms;


};




#endif