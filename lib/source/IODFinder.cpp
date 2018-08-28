#include "IODFinder.hpp"
#include "Psopt.hpp"
#include "IODBounds.hpp"
#include <RigidBodyKinematics.hpp>
#include "System.hpp"
#include "Args.hpp"
#include "Dynamics.hpp"
#include "Observer.hpp"

IODFinder::IODFinder(std::vector<RigidTransform> * rigid_transforms, 
	int N_iter, 
	int particles){

	this -> N_iter = N_iter;
	this -> particles = particles;
	this -> rigid_transforms = rigid_transforms;

	for (unsigned int i = 0; i < this -> rigid_transforms-> size(); ++i){
		this -> rigid_transforms_covariances.push_back(arma::eye<arma::mat>(6,6));
	}

}



void IODFinder::run(arma::vec lower_bounds,arma::vec upper_bounds,int verbose_level,const arma::vec & guess){

	if (lower_bounds.n_rows == 0){
		lower_bounds = {A_MIN,E_MIN,I_MIN,RAAN_MIN,OMEGA_MIN,M0_MIN,MU_MIN};
		upper_bounds = {A_MAX,E_MAX,I_MAX,RAAN_MAX,OMEGA_MAX,M0_MAX,MU_MAX};
	}
	
	Psopt<std::vector<RigidTransform> *> psopt(IODFinder::cost_function, 
		lower_bounds,
		upper_bounds, 
		this -> particles,
		this -> N_iter,
		this -> rigid_transforms,
		guess);


	std::map<int,std::string> boundary_conditions = {
		std::make_pair<int,std::string>(2,"w"),
		std::make_pair<int,std::string>(3,"w"),
		std::make_pair<int,std::string>(4,"w"),
		std::make_pair<int,std::string>(5,"w"),
	};

	psopt.run(false,verbose_level,boundary_conditions);

	arma::vec elements = psopt.get_result();

	this -> keplerian_state_at_epoch = OC::KepState(elements.subvec(0,5),elements(6));


}

OC::KepState IODFinder::get_result() const{
	return this -> keplerian_state_at_epoch;
}

double IODFinder::cost_function(arma::vec particle, std::vector<RigidTransform> * args,int verbose_level){

	// Particle State ordering:
	// [a,e,i,Omega,omega,M0_0,mu]

	OC::KepState kep_state(particle.subvec(0,5),particle(6));

	int N =  args -> size();
	arma::mat positions(3,N + 1);

	// Since the dt at which images are captured is constant, the epoch can be inferred
	// by subtracting dt to the first rigid transform's t_k

	double dt = args -> at(1).t_k -  args -> at(0).t_k;
	assert(dt == args -> at(2).t_k -  args -> at(1).t_k);
	double epoch_time = args -> at(0).t_k - dt;

	if (verbose_level > 1){
		std::cout << "\n - Epoch time: " << epoch_time;
		std::cout << "\n - dt: " << dt << std::endl;
	}

	positions.col(0) = kep_state.convert_to_cart(epoch_time).get_position_vector();

	for (int k = 1; k < N + 1; ++k){
		double t_k = args -> at(k - 1).t_k;
		double time_from_epoch = t_k - epoch_time;
		positions.col(k) = kep_state.convert_to_cart(time_from_epoch).get_position_vector();

		if (verbose_level > 1){
			std::cout << " - Transform index : " << k << std::endl;
			std::cout << " - Time from 0 : " << t_k << std::endl;
			std::cout << " - Time from epoch : " << time_from_epoch << std::endl << std::endl;
		}
	}

	arma::vec epsilon = arma::zeros<arma::vec>(3 * N);

	for (int k = 0; k < N; ++k ){

		arma::mat M_k = args -> at(k).M_k;
		arma::mat X_k = args -> at(k).X_k;

		epsilon.subvec( 3 * k, 3 * k + 2) = positions.col(k) - M_k * positions.col(k + 1) + X_k;
		
		if (verbose_level > 1){
			std::cout << "\t Epsilon " << k << " = " << arma::norm(epsilon.subvec( 3 * k, 3 * k + 2)) << std::endl;

		}





	}


	if (verbose_level > 2){
		std::cout << positions.col(0).t() << std::endl;
		std::cout << positions.col(1).t() << std::endl;
		std::cout << args -> at(0).M_k << std::endl;
		std::cout << args -> at(0).X_k << std::endl;
	}


	return arma::norm(epsilon);

}



arma::mat::fixed<6,6> IODFinder::compute_P_I_prime_k(
	const arma::mat::fixed<12,12> P_V_tilde_k,
	const arma::mat::fixed<3,3> & M_k_tilde_bar,
	const arma::vec::fixed<3> & X_k_tilde_bar,
	const arma::mat::fixed<3,3> & M_km1_tilde_bar,
	const arma::vec::fixed<3> & X_km1_tilde_bar,
	const arma::mat::fixed<3,3> & LN_k,
	const arma::mat::fixed<3,3> & LN_km1){

	arma::mat::fixed<6,12> partial_I_prime_k_partial_Vtilde_k = IODFinder::compute_dIprime_k_dVtilde_k(
		M_k_tilde_bar,
		X_k_tilde_bar,
		M_km1_tilde_bar,
		X_km1_tilde_bar,
		LN_k,
		LN_km1);

	return partial_I_prime_k_partial_Vtilde_k * P_V_tilde_k * partial_I_prime_k_partial_Vtilde_k.t();

}





arma::mat::fixed<6,12> IODFinder::compute_dIprime_k_dVtilde_k(
	const arma::mat::fixed<3,3> & M_k_tilde_bar,
	const arma::vec::fixed<3> & X_k_tilde_bar,
	const arma::mat::fixed<3,3> & M_km1_tilde_bar,
	const arma::vec::fixed<3> & X_km1_tilde_bar,
	const arma::mat::fixed<3,3> & LN_k,
	const arma::mat::fixed<3,3> & LN_km1){

	arma::vec::fixed<3> a_k_bar = M_km1_tilde_bar.t() * (X_k_tilde_bar - X_km1_tilde_bar);
	arma::mat::fixed<3,3> Uk_bar = -4 * RBK::tilde(a_k_bar);

	arma::mat::fixed<6,12> partial_I_prime_k_partial_Vtilde_k = arma::zeros<arma::mat>(6,12);


	partial_I_prime_k_partial_Vtilde_k.submat(0,0,2,2) = LN_km1.t() * M_km1_tilde_bar.t();
	partial_I_prime_k_partial_Vtilde_k.submat(0,3,2,5) = - LN_km1.t() * M_km1_tilde_bar.t();
	partial_I_prime_k_partial_Vtilde_k.submat(0,9,2,11) = LN_km1.t() * Uk_bar;




	partial_I_prime_k_partial_Vtilde_k.submat(3,6,5,11) = IODFinder::compute_dsigmatilde_kdZ_k(
		M_k_tilde_bar,
		M_km1_tilde_bar,
		LN_k,
		LN_km1);



	return partial_I_prime_k_partial_Vtilde_k;

}






arma::mat::fixed<6,12> IODFinder::compute_dsigmatilde_kdZ_k(
	const arma::mat::fixed<3,3> & M_k_tilde_bar,
	const arma::mat::fixed<3,3> & M_km1_tilde_bar,
	const arma::mat::fixed<3,3> & LN_k,
	const arma::mat::fixed<3,3> & LN_km1){

	arma::mat::fixed<3,3> Abar_k = M_km1_tilde_bar.t() * M_k_tilde_bar;
	arma::mat::fixed<3,6> partial_sigmatildek_partial_Zk;

	arma::vec e0 = {1,0,0};
	arma::vec e1 = {0,1,0};
	arma::vec e2 = {0,0,1};

	partial_sigmatildek_partial_Zk.submat(0,0,0,2) = -arma::dot(e2,M_k_tilde_bar.t() * LN_km1.t() * Abar_k * RBK::tilde(LN_k * e1));
	partial_sigmatildek_partial_Zk.submat(1,0,1,2) = -arma::dot(e0,M_k_tilde_bar.t() * LN_km1.t() * Abar_k * RBK::tilde(LN_k * e2));
	partial_sigmatildek_partial_Zk.submat(2,0,2,2) = -arma::dot(e1,M_k_tilde_bar.t() * LN_km1.t() * Abar_k * RBK::tilde(LN_k * e0));

	partial_sigmatildek_partial_Zk.submat(0,3,0,5) = arma::dot(e2,M_k_tilde_bar.t() * LN_km1.t() * RBK::tilde(Abar_k * LN_k * e1));
	partial_sigmatildek_partial_Zk.submat(1,3,1,5) = arma::dot(e0,M_k_tilde_bar.t() * LN_km1.t() * RBK::tilde(Abar_k * LN_k * e2));
	partial_sigmatildek_partial_Zk.submat(2,3,2,5) = arma::dot(e1,M_k_tilde_bar.t() * LN_km1.t() * RBK::tilde(Abar_k * LN_k * e0));

	return partial_sigmatildek_partial_Zk;
}





arma::mat::fixed<3,6> IODFinder::compute_J_k(const arma::mat::fixed<3,3> & M_kp1_tilde_bar,
	const arma::vec::fixed<3> CL_kp1_bar){

	arma::mat::fixed<3,6> J;

	J.submat(0,0,2,2) = arma::eye<arma::mat>(3,3);
	J.submat(0,3,2,5) = 4 * M_kp1_tilde_bar * RBK::tilde(CL_kp1_bar);

	return J;

}

arma::mat IODFinder::compute_H_k(const arma::mat & Phi_k, 
	const arma::mat & Phi_kp1, 
	const arma::mat::fixed<3,3> & M_kp1_tilde_bar){
	return - Phi_k.rows(0,2) + M_kp1_tilde_bar * Phi_kp1.rows(0,2);

}	

arma::vec::fixed<3> IODFinder::compute_y_k(
	const arma::vec::fixed<3> & CL_k_bar,
	const arma::vec::fixed<3> & CL_kp1_bar,
	const arma::mat::fixed<3,3> & M_kp1_prime_bar,
	const arma::vec::fixed<3> & X_kp1_prime_bar){


	return CL_k_bar - M_kp1_prime_bar * CL_kp1_bar + X_kp1_prime_bar;

}

arma::mat::fixed<3,3> IODFinder::compute_R_k(
	const arma::mat::fixed<3,3> & M_kp1_tilde_bar,
	const arma::vec::fixed<3> & CL_kp1_bar,
	const arma::mat::fixed<6,6> & P_I_k_prime){


	arma::mat::fixed<3,6> J_k = IODFinder::compute_J_k(M_kp1_tilde_bar,CL_kp1_bar);

	return J_k * P_I_k_prime * J_k.t();

}



void IODFinder::build_normal_equations(
	arma::mat & info_mat,
	arma::vec & normal_mat,
	double & residuals,
	const std::vector<RigidTransform> * rigid_transforms,
	const std::vector<arma::mat> & rigid_transforms_covariances,
	arma::vec & apriori_state,
	std::string dynamics_name){

	info_mat.fill(0);
	normal_mat.fill(0);
	residuals = 0;

	std::vector<arma::vec> positions;
	std::vector<arma::mat> stms;

	positions.push_back(apriori_state.rows(0,2));

	if (dynamics_name == "keplerian"){
		OC::CartState cart_state(apriori_state.rows(0,5),apriori_state(6));
		OC::KepState kep_state = cart_state.convert_to_kep(0);

		IODFinder::compute_stms(apriori_state,rigid_transforms,stms);
		
		for (int k = 0; k < rigid_transforms -> size(); ++ k){
			positions.push_back(kep_state.convert_to_cart(rigid_transforms -> at(k).t_k).get_position_vector());
		}

	}


	for (int k = 0; k < rigid_transforms -> size(); ++ k){

		auto M_kp1_prime_bar = rigid_transforms -> at(k).M_k;
		auto X_kp1_prime_bar = rigid_transforms -> at(k).X_k;

		auto H_k = IODFinder::compute_H_k(stms[k],stms[k+1],M_kp1_prime_bar);
		

		auto y_k = IODFinder::compute_y_k(
			positions[k],
			positions[k+1],
			M_kp1_prime_bar,
			X_kp1_prime_bar);

		auto P_I_k_prime = rigid_transforms_covariances.at(k);


		auto W_k = arma::inv(IODFinder::compute_R_k(
			M_kp1_prime_bar,
			positions[k+1],
			P_I_k_prime));

		info_mat += H_k.t() * W_k * H_k;
		normal_mat += H_k.t() * W_k * y_k;
		residuals += arma::dot(y_k,W_k * y_k);

	}


}


void IODFinder::run_batch(){

	OC::KepState apriori_kepstate = this -> keplerian_state_at_epoch ;
	arma::vec apriori_state(7);
	apriori_state.rows(0,5) = apriori_kepstate.convert_to_cart(0).get_state();
	apriori_state(6) = apriori_kepstate.get_mu();


	int N_iter = 10;

	arma::mat info_mat(7,7);
	arma::vec normal_mat(7);
	double residuals;

	for (int i = 0; i < N_iter; ++i){
		std::cout << "\t\t Iteration " << i + 1 << std::endl;

		IODFinder::build_normal_equations(

			info_mat,
			normal_mat,
			residuals,
			this -> rigid_transforms,
			this -> rigid_transforms_covariances,
			apriori_state,
			"keplerian");

		arma::vec deviation = arma::solve(info_mat,normal_mat);

		std::cout << "\t\t Info mat : " << std::endl;
		std::cout << info_mat << std::endl;

		std::cout << "\t\t Normal mat : " << std::endl;
		std::cout << normal_mat << std::endl;


		std::cout << "\t\t Deviation : " << std::endl;
		std::cout << deviation << std::endl;

		apriori_state += deviation;
		OC::CartState new_cartstate(apriori_state.rows(0,5),apriori_state(6));
		OC::KepState new_kepstate = new_cartstate.convert_to_kep(0);
		
		std::cout << "\t\t Residuals : " << residuals << std::endl;

		std::cout << "\t\t Keplerian state at epoch : " << std::endl;
		std::cout << new_kepstate.get_state() << std::endl;
		
	}

}


void IODFinder::compute_stms(const arma::vec::fixed<7> & X_hat,
	const std::vector<RigidTransform> * rigid_transforms,
	std::vector<arma::mat> & stms){

	stms.clear();

	int N_est = X_hat.n_rows;


	Args args;
	args.set_mu(X_hat(6));
	System dynamics(args,
		N_est,
		Dynamics::point_mass_mu_dxdt_odeint ,
		Dynamics::point_mass_mu_jac_odeint,
		0,
		nullptr );

	arma::vec x(N_est + N_est * N_est);
	x.rows(0,N_est - 1) = X_hat;
	x.rows(N_est,N_est + N_est * N_est - 1) = arma::vectorise(arma::eye<arma::mat>(N_est,N_est));


	std::vector<arma::vec> augmented_state_history;
	typedef boost::numeric::odeint::runge_kutta_cash_karp54< arma::vec > error_stepper_type;
	auto stepper = boost::numeric::odeint::make_controlled<error_stepper_type>( 1.0e-13 ,
		1.0e-16 );

	std::vector<double> times;

	times.push_back(0);

	for (int i = 0; i < rigid_transforms -> size(); ++i){
		times.push_back(rigid_transforms -> at(i).t_k);
	}

	auto tbegin = times.begin();
	auto tend = times.end();
	boost::numeric::odeint::integrate_times(stepper, dynamics, x, tbegin, tend,1e-10,
		Observer::push_back_augmented_state_no_mrp(augmented_state_history));

	for (int i = 0; i < rigid_transforms -> size()+1; ++i){

		arma::mat::fixed<7,7> stm = arma::reshape(
			augmented_state_history[i].rows(N_est,N_est + N_est * N_est - 1),
			N_est,N_est);

		stms.push_back(stm);
	}
	
}





