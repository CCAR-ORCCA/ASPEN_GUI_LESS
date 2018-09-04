#include "IODFinder.hpp"
#include "Psopt.hpp"
#include "IODBounds.hpp"
#include <RigidBodyKinematics.hpp>
#include "System.hpp"
#include "Args.hpp"
#include "Dynamics.hpp"
#include "Observer.hpp"

IODFinder::IODFinder(std::vector<RigidTransform> * sequential_rigid_transforms,
	std::vector<RigidTransform> * absolute_rigid_transforms,  
	std::vector<arma::vec> mrps_LN,
	double stdev_Xtilde,
	double stdev_sigmatilde,
	int N_iter, 
	int particles,
	bool remove_time_correlations_in_mes){

	this -> N_iter = N_iter;
	this -> particles = particles;
	this -> sequential_rigid_transforms = sequential_rigid_transforms;
	this -> absolute_rigid_transforms = absolute_rigid_transforms;
	this -> mrps_LN = mrps_LN;
	this -> stdev_Xtilde = stdev_Xtilde;
	this -> stdev_sigmatilde = stdev_sigmatilde;
	this -> remove_time_correlations_in_mes = remove_time_correlations_in_mes;


	// this -> debug_R();

	this -> compute_P_T();



}



void IODFinder::run(arma::vec lower_bounds,
	arma::vec upper_bounds,std::string type,
	int verbose_level,const arma::vec & guess){

	arma::vec results;
	if (type == "keplerian"){
		if (lower_bounds.n_rows == 0){
			lower_bounds = {A_MIN,E_MIN,I_MIN,RAAN_MIN,OMEGA_MIN,M0_MIN,MU_MIN};
			upper_bounds = {A_MAX,E_MAX,I_MAX,RAAN_MAX,OMEGA_MAX,M0_MAX,MU_MAX};
		}

		Psopt<std::vector<RigidTransform> *> psopt(IODFinder::cost_function, 
			lower_bounds,
			upper_bounds, 
			this -> particles,
			this -> N_iter,
			this -> sequential_rigid_transforms,
			guess);


		std::map<int,std::string> boundary_conditions = {
			std::make_pair<int,std::string>(2,"w"),
			std::make_pair<int,std::string>(3,"w"),
			std::make_pair<int,std::string>(4,"w"),
			std::make_pair<int,std::string>(5,"w"),
		};
		psopt.run(false,verbose_level,boundary_conditions);
		results = psopt.get_result();
		this -> keplerian_state_at_epoch = OC::KepState(results.subvec(0,5),results(6));

	}
	else if (type == "cartesian"){

		Psopt<std::vector<RigidTransform> *> psopt(IODFinder::cost_function_cartesian, 
			lower_bounds,
			upper_bounds, 
			this -> particles,
			this -> N_iter,
			this -> sequential_rigid_transforms,
			guess);
		psopt.run(false,verbose_level);
		results = psopt.get_result();

		OC::CartState cart_state(results.subvec(0,5),results(6));
		this -> keplerian_state_at_epoch = cart_state.convert_to_kep(0);

	}



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

double IODFinder::cost_function_cartesian(arma::vec particle, 
	std::vector<RigidTransform> * args,int verbose_level){

	// Particle State ordering:
	// [a,e,i,Omega,omega,M0_0,mu]
	OC::CartState cart_state(particle.subvec(0,5),particle(6));
	OC::KepState kep_state = cart_state.convert_to_kep(0);

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

arma::mat::fixed<6,12> IODFinder::compute_dIprime_k_dVtilde_k(int k) const{

	arma::vec Xk,Xkm1;
	arma::mat Mk,Mkm1,LN_km1,LN_k;
	arma::mat::fixed<6,12> partial_I_prime_k_partial_Vtilde_k = arma::zeros<arma::mat>(6,12);

	Xk = this -> absolute_rigid_transforms -> at(k).X_k;
	Mk =  this -> absolute_rigid_transforms -> at(k).M_k;
	LN_k = RBK::mrp_to_dcm(this -> mrps_LN.at(k));

	if (k > 0){
		Xkm1 = this -> absolute_rigid_transforms -> at(k- 1).X_k;
		Mkm1 =  this -> absolute_rigid_transforms -> at(k- 1).M_k;
		LN_km1 = RBK::mrp_to_dcm(this -> mrps_LN.at(k - 1));

	}
	else{
		Xkm1 = arma::zeros<arma::vec>(3);
		Mkm1 = arma::eye<arma::mat>(3,3);
		LN_km1 = arma::eye<arma::mat>(3,3);

	}


	arma::vec::fixed<3> a_k_bar = Mkm1.t() * (Xk - Xkm1);
	arma::mat::fixed<3,3> Uk_bar = - 4 * RBK::tilde(a_k_bar);

	partial_I_prime_k_partial_Vtilde_k.submat(0,0,2,2) = - LN_km1.t() * Mkm1.t();
	partial_I_prime_k_partial_Vtilde_k.submat(0,3,2,5) = LN_km1.t() * Uk_bar;

	partial_I_prime_k_partial_Vtilde_k.submat(0,6,2,8) = LN_km1.t() * Mkm1.t();

	arma::mat dsigmadZ = IODFinder::compute_dsigmatilde_kdZ_k(
		this -> sequential_rigid_transforms -> at (k - 1).M_k,
		Mk,
		Mkm1,
		LN_k,
		LN_km1);

	partial_I_prime_k_partial_Vtilde_k.submat(3,3,5,5) = dsigmadZ.cols(0,2);
	partial_I_prime_k_partial_Vtilde_k.submat(3,9,5,11) = dsigmadZ.cols(3,5);

	return partial_I_prime_k_partial_Vtilde_k;

}






arma::mat::fixed<3,6> IODFinder::compute_dsigmatilde_kdZ_k(
	const arma::mat::fixed<3,3> & M_k_p,
	const arma::mat::fixed<3,3> & M_k_tilde_bar,
	const arma::mat::fixed<3,3> & M_km1_tilde_bar,
	const arma::mat::fixed<3,3> & LN_k,
	const arma::mat::fixed<3,3> & LN_km1){

	arma::mat::fixed<3,3> Abar_k = M_km1_tilde_bar.t() * M_k_tilde_bar;
	arma::mat::fixed<3,6> partial_sigmatildek_partial_Zk = arma::zeros<arma::mat>(3,6);

	arma::vec e0 = {1,0,0};
	arma::vec e1 = {0,1,0};
	arma::vec e2 = {0,0,1};


	partial_sigmatildek_partial_Zk.submat(0,0,0,2) = e2.t() * M_k_p.t() * LN_km1.t() * RBK::tilde(Abar_k * LN_k * e1);
	partial_sigmatildek_partial_Zk.submat(1,0,1,2) = e0.t() * M_k_p.t() * LN_km1.t() * RBK::tilde(Abar_k * LN_k * e2);
	partial_sigmatildek_partial_Zk.submat(2,0,2,2) = e1.t() * M_k_p.t() * LN_km1.t() * RBK::tilde(Abar_k * LN_k * e0);

	partial_sigmatildek_partial_Zk.submat(0,3,0,5) = - e2.t() * M_k_p.t() * LN_km1.t() * Abar_k * RBK::tilde(LN_k * e1);
	partial_sigmatildek_partial_Zk.submat(1,3,1,5) = - e0.t() * M_k_p.t() * LN_km1.t() * Abar_k * RBK::tilde(LN_k * e2);
	partial_sigmatildek_partial_Zk.submat(2,3,2,5) = - e1.t() * M_k_p.t() * LN_km1.t() * Abar_k * RBK::tilde(LN_k * e0);

	return partial_sigmatildek_partial_Zk;
}



arma::mat::fixed<3,6> IODFinder::compute_J_k(int k,const std::vector<arma::vec::fixed<3>> & positions) const{

	arma::mat::fixed<3,6> J ;

	J.submat(0,0,2,2) = arma::eye<arma::mat>(3,3);
	J.submat(0,3,2,5) = - 4 * this -> sequential_rigid_transforms -> at(k).M_k * RBK::tilde(positions.at(k + 1));

	return J;

}




arma::mat::fixed<3,7> IODFinder::compute_H_k(const arma::mat & Phi_k, 
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


arma::mat IODFinder::compute_partial_y_partial_T(const std::vector<arma::vec::fixed<3>> & positions) const{

	return (this -> compute_partial_y_partial_I(positions)
		*  this -> compute_partial_I_partial_V() 
		* this -> compute_partial_V_partial_T());

}

arma::mat IODFinder::compute_partial_y_partial_I(const std::vector<arma::vec::fixed<3>> & positions) const{

	arma::mat dydI = arma::zeros<arma::mat>(3 * this -> sequential_rigid_transforms -> size(),
		6 * this -> sequential_rigid_transforms -> size());

	for (int k = 0; k < this -> sequential_rigid_transforms -> size(); ++k){
		dydI.submat(3 * k, 6 * k, 3 * k + 2, 6 * k + 5) = this -> compute_J_k(k,positions);
	}

	return dydI;
}

arma::mat IODFinder::compute_partial_I_partial_V() const{

	arma::mat dIdV = arma::zeros<arma::mat>(6 * this -> sequential_rigid_transforms -> size(),
		6 + 12 * this -> sequential_rigid_transforms -> size());


	for (int k = 0; k < this -> sequential_rigid_transforms -> size(); ++k){

		arma::mat::fixed<6,12> dIdVk = IODFinder::compute_dIprime_k_dVtilde_k(k + 1);

		dIdV.submat(6 * k, 
			6 + 12 * k, 
			6 * k + 5, 
			6 + 12 * k + 11 ) = dIdVk;

		
	}






	return dIdV;

}

arma::mat IODFinder::compute_partial_V_partial_T() const{


	arma::mat dVdT = arma::zeros<arma::mat>(6 + 12 * this -> sequential_rigid_transforms -> size() ,
		6 * (this -> sequential_rigid_transforms -> size() + 1) );

	// V0 only contains T0
	dVdT.submat(0,0,5,5) = arma::eye<arma::mat>(6,6);
	for (int k = 0; k < this -> sequential_rigid_transforms -> size() ; ++k){

		dVdT.submat(12 * k + 6, 
			6 * k,
			12 * k + 11 + 6, 
			6 * k + 11) = arma::eye<arma::mat>(12,12);;
	}

	return dVdT;

}











void IODFinder::build_normal_equations(
	arma::mat & info_mat,
	arma::vec & normal_mat,
	arma::vec & residual_vector,
	arma::vec & apriori_state,
	const std::vector<arma::vec::fixed<3>> & positions,
	const std::vector<arma::mat> & stms) const{

	residual_vector.fill(0);

	arma::mat H = arma::zeros<arma::mat>(3 * this -> sequential_rigid_transforms -> size(), 7);
	
	// H matrices
	// y vector
	// R matrix


	for (int k = 0; k < this -> sequential_rigid_transforms -> size(); ++ k){

		arma::mat Mkp1 = this -> sequential_rigid_transforms -> at(k).M_k;
		arma::vec Xkp1 = this -> sequential_rigid_transforms -> at(k).X_k;

		H.rows(3 * k, 3 * k + 2) = IODFinder::compute_H_k(stms[k],stms[k+1],Mkp1);
		residual_vector.rows(3 * k, 3 * k + 2) = IODFinder::compute_y_k(
			positions[k],
			positions[k+1],
			Mkp1,
			Xkp1);
	}

	info_mat = H.t() * this -> W * H;
	normal_mat = H.t() * this -> W * residual_vector;

	
}


void IODFinder::run_batch(arma::vec & state,
	arma::mat & cov){


	OC::KepState apriori_kepstate = this -> keplerian_state_at_epoch ;
	arma::vec apriori_state(7);
	apriori_state.rows(0,5) = apriori_kepstate.convert_to_cart(0).get_state();
	apriori_state(6) = apriori_kepstate.get_mu();

	int N_iter = 10;

	arma::mat info_mat(7,7);
	arma::vec normal_mat(7);
	arma::vec residual_vector = arma::vec(3 * this -> sequential_rigid_transforms -> size());


	std::vector<arma::vec::fixed<3> > positions;
	std::vector<arma::mat> stms;

	for (int i = 0; i < N_iter; ++i){

		this -> compute_state_stms(apriori_state,positions,stms);
		this -> compute_W(positions);
		this -> build_normal_equations(
			info_mat,
			normal_mat,
			residual_vector,
			apriori_state,
			positions,
			stms);

		arma::vec deviation = arma::solve(info_mat,normal_mat);

		apriori_state += deviation;

	}

	state = apriori_state;
	cov = arma::inv(info_mat);
	arma::vec index_v = arma::randi<arma::vec>(1);
	double index = index_v(0);
	residual_vector.save("../output/residual_vector_" +std::to_string(index) + ".txt",arma::raw_ascii);


}


void IODFinder::compute_state_stms(const arma::vec::fixed<7> & X_hat,
	std::vector<arma::vec::fixed<3> > & positions,
	std::vector<arma::mat> & stms) const{

	positions.clear();
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

	for (int i = 0; i < this -> sequential_rigid_transforms -> size(); ++i){
		times.push_back(this -> sequential_rigid_transforms -> at(i).t_k);
	}

	auto tbegin = times.begin();
	auto tend = times.end();
	boost::numeric::odeint::integrate_times(stepper, dynamics, x, tbegin, tend,1e-10,
		Observer::push_back_augmented_state_no_mrp(augmented_state_history));

	for (int i = 0; i < times.size(); ++i){

		arma::mat::fixed<7,7> stm = arma::reshape(
			augmented_state_history[i].rows(N_est,N_est + N_est * N_est - 1),
			N_est,N_est);
		positions.push_back(augmented_state_history[i].rows(0,2));
		stms.push_back(stm);
	}

}



void IODFinder::debug_stms(const std::vector<RigidTransform> * rigid_transforms){

	std::vector<double> times;

	times.push_back(0);

	for (int i = 0; i < rigid_transforms -> size(); ++i){
		times.push_back(rigid_transforms -> at(i).t_k);
	}

	auto tbegin = times.begin();
	auto tend = times.end();

	int N_est = 7;
	arma::vec state = {1000,0.1,1,1,1,0};
	OC::KepState kep_state(state,2);	

	arma::vec X_hat(7); 
	X_hat.rows(0,5) = kep_state.convert_to_cart(0).get_state();
	X_hat(6) = kep_state.get_mu();


	arma::vec dX = 1e-4 * X_hat;

	Args args;
	System dynamics(args,
		7,
		Dynamics::point_mass_mu_dxdt_odeint);

	arma::vec x_non_linear_0 = X_hat;
	std::vector<arma::vec> state_history_0;

	typedef boost::numeric::odeint::runge_kutta_cash_karp54< arma::vec > error_stepper_type;
	auto stepper = boost::numeric::odeint::make_controlled<error_stepper_type>( 1.0e-13 ,
		1.0e-16 );


	boost::numeric::odeint::integrate_times(stepper, dynamics, x_non_linear_0, tbegin, tend,1e-10,
		Observer::push_back_augmented_state_no_mrp(state_history_0));



	arma::vec x_non_linear_1 = X_hat + dX;
	std::vector<arma::vec> state_history_1;

	typedef boost::numeric::odeint::runge_kutta_cash_karp54< arma::vec > error_stepper_type;
	auto stepper_1 = boost::numeric::odeint::make_controlled<error_stepper_type>( 1.0e-13 ,
		1.0e-16 );


	boost::numeric::odeint::integrate_times(stepper_1, dynamics, x_non_linear_1, tbegin, tend,1e-10,
		Observer::push_back_augmented_state_no_mrp(state_history_1));





	// Linear dynamics

	std::vector<arma::mat> stms;



	System dynamics_lin(args,
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
	auto stepper_lin = boost::numeric::odeint::make_controlled<error_stepper_type>( 1.0e-13 ,
		1.0e-16 );

	boost::numeric::odeint::integrate_times(stepper_lin, dynamics_lin, x, tbegin, tend,1e-10,
		Observer::push_back_augmented_state_no_mrp(augmented_state_history));

	for (int i = 0; i < rigid_transforms -> size()+1; ++i){

		arma::mat::fixed<7,7> stm = arma::reshape(
			augmented_state_history[i].rows(N_est,N_est + N_est * N_est - 1),
			N_est,N_est);

		stms.push_back(stm);
	}


	arma::mat state_history_0_mat(7,times.size());
	arma::mat state_history_1_mat(7,times.size());
	arma::mat state_history_lin_mat(7,times.size());


	for (int i =0; i < times.size(); ++i){
		state_history_0_mat.col(i) = state_history_0[i];
		state_history_1_mat.col(i) = state_history_1[i];
		state_history_lin_mat.col(i) = stms[i] * dX + X_hat;
	}


	state_history_0_mat.save("../output/state_history_0.txt",arma::raw_ascii);
	state_history_1_mat.save("../output/state_history_1.txt",arma::raw_ascii);
	state_history_lin_mat.save("../output/state_history_lin.txt",arma::raw_ascii);

	// debug observations

	int k = 10;

	arma::mat Mkp1 = rigid_transforms -> at(k).M_k;
	arma::mat Xkp1 = rigid_transforms -> at(k).X_k;


	arma::vec Gk_nom = state_history_0[k].rows(0,2) - Mkp1 * state_history_0[k + 1].rows(0,2) + rigid_transforms -> at(k).X_k;

	arma::vec Gk_perp = state_history_1[k].rows(0,2) - Mkp1 * state_history_1[k + 1].rows(0,2) + rigid_transforms -> at(k).X_k;

	arma::vec dG_true = Gk_perp - Gk_nom;

	arma::mat H_k = IODFinder::compute_H_k(stms[k], 
		stms[k+1], 
		Mkp1);

	arma::vec dG_linear = H_k * dX;



	std::cout << dG_true.t() << std::endl;
	std::cout << dG_linear.t() << std::endl;

	throw;







}






void IODFinder::seq_transform_from_epoch_transform(int k, 
	RigidTransform & seq_transform_k,
	const RigidTransform & epoch_transform_k,
	const RigidTransform & epoch_transform_km1, 
	const std::vector<arma::vec> & mrps_LN){

	seq_transform_k.t_k = epoch_transform_k.t_k;

	seq_transform_k.M_k = RBK::mrp_to_dcm(mrps_LN[k - 1]).t() * epoch_transform_km1.M_k.t() * epoch_transform_k.M_k * RBK::mrp_to_dcm(mrps_LN[k]);
	seq_transform_k.X_k = RBK::mrp_to_dcm(mrps_LN[k - 1]).t() * epoch_transform_km1.M_k.t() * (epoch_transform_k.X_k - epoch_transform_km1.X_k);


}




void IODFinder::compute_W(const std::vector<arma::vec::fixed<3>> & positions){

	arma::mat dydT = IODFinder::compute_partial_y_partial_T(positions);


	arma::mat R = dydT * this -> P_T * dydT.t();

	// for (int k = 0 ; k < this -> sequential_rigid_transforms -> size(); ++k){

	// 	for (int j = 0 ; j <= k; ++j){

	// 		arma::mat Rkj = this -> compute_Rkj(k,j,positions);

	// 		R.submat(3 * k, 3 * j, 3 * k + 2, 3 * j + 2) = Rkj;
	// 		R.submat(3 * j, 3 * k, 3 * j + 2, 3 * k + 2) = Rkj.t();

	// 	}

	// }
	
	R.save("../output/R_mat.txt",arma::raw_ascii);

	this -> W = arma::inv(R);


}







void IODFinder::debug_R() const{

	arma::vec true_particle = { 6.4514e+02,1.9006e+02,3.3690e+02,-2.7270e-02, 4.0806e-03,5.4598e-02,2.25535};

	std::vector<arma::vec::fixed<3> > positions;
	std::vector<arma::mat> stms;
	this -> compute_state_stms(true_particle,positions,stms);



	int N_rigid_transforms = this -> sequential_rigid_transforms -> size();

	std::cout << "Number of sequential transforms: " << this -> sequential_rigid_transforms -> size() << std::endl;
	std::cout << "Number of absolute transforms: " << this -> absolute_rigid_transforms -> size() << std::endl;

	
	// Nominal sequential transforms
	arma::vec I_nom(6 * N_rigid_transforms);
	arma::vec V_nom(12 * N_rigid_transforms + 6);


	V_nom.subvec(0,2) = this -> absolute_rigid_transforms -> at(0).X_k;
	V_nom.subvec(3,5) = RBK::dcm_to_mrp(this -> absolute_rigid_transforms -> at(0).M_k);

	for (int k = 1 ; k < N_rigid_transforms + 1; ++ k){

		
		I_nom.subvec(6 * (k-1), 6 * (k-1) + 2) = this -> sequential_rigid_transforms -> at(k-1).X_k;
		I_nom.subvec(6 * (k-1) + 3, 6 * (k-1) + 5) = RBK::dcm_to_mrp(this -> sequential_rigid_transforms -> at(k-1).M_k) ;

		V_nom.subvec(6 + 12 * (k - 1) , 6 + 12 * (k - 1) + 2) = this -> absolute_rigid_transforms -> at(k-1).X_k;
		V_nom.subvec(6 + 12 * (k - 1) + 3, 6 + 12 * (k - 1) + 5) = RBK::dcm_to_mrp(this -> absolute_rigid_transforms -> at(k-1).M_k);

		V_nom.subvec(6 + 12 * (k - 1) + 6, 6 + 12 * (k - 1) + 8) = this -> absolute_rigid_transforms -> at(k).X_k;
		V_nom.subvec(6 + 12 * (k - 1) + 9, 6 + 12 * (k - 1) + 11) = RBK::dcm_to_mrp(this -> absolute_rigid_transforms -> at(k).M_k);


	}

	// Nominal y
	arma::vec y_nom = arma::zeros<arma::vec>(3 * this -> sequential_rigid_transforms -> size());

	for (int k = 0; k < this -> sequential_rigid_transforms -> size(); ++ k){

		arma::mat Mkp1 = this -> sequential_rigid_transforms ->  at(k).M_k;
		arma::vec Xkp1 = this -> sequential_rigid_transforms ->  at(k).X_k;

		y_nom.rows(3 * k, 3 * k + 2) = IODFinder::compute_y_k(
			positions[k],
			positions[k+1],
			Mkp1,
			Xkp1);
	}

	arma::vec dT = arma::zeros<arma::vec>(6 * this -> sequential_rigid_transforms -> size() + 6);
	arma::vec dI = arma::zeros<arma::vec>(6 * this -> sequential_rigid_transforms -> size());


	for (int i = 1; i < this -> sequential_rigid_transforms -> size(); ++i){
			dT.subvec(6 * i, 6 * i + 2) =  1 * arma::randn<arma::vec>(3);
			dT.subvec(6 * i + 3, 6 * i + 2 + 3) = 0.001 * arma::randn<arma::vec>(3);
	}

	for (int i = 0; i < this -> sequential_rigid_transforms -> size(); ++i){

		dI.subvec(6 * (i), 6 * (i) + 2) = 0.0 * arma::randn<arma::vec>(3);
		dI.subvec(6 * (i) + 3, 6 * (i) + 2 + 3) = 0.0 * arma::randn<arma::vec>(3);
	}


	// Perturbed sequential transforms
	std::vector<RigidTransform> rigid_transforms_seq_p;
	arma::vec I_p(6 * N_rigid_transforms);
	arma::vec V_p(12 * N_rigid_transforms + 6);
	arma::vec dV_non_lin = arma::zeros<arma::vec>(12 * N_rigid_transforms + 6);
	arma::vec dI_non_lin = arma::zeros<arma::vec>(6 * N_rigid_transforms);


	V_p.subvec(0,2) = this -> absolute_rigid_transforms -> at(0).X_k;
	V_p.subvec(3,5) = RBK::dcm_to_mrp(this -> absolute_rigid_transforms -> at(0).M_k);

	for (int k = 1 ; k < N_rigid_transforms + 1; ++ k){

		arma::vec X_tilde_km1_p = this -> absolute_rigid_transforms -> at(k-1).X_k + dT.subvec(6 * (k - 1) , 6 * (k - 1) + 2);
		arma::vec X_tilde_k_p = this -> absolute_rigid_transforms -> at(k).X_k + dT.subvec(6 * k , 6 * k + 2);

		arma::mat M_tilde_km1_p = this -> absolute_rigid_transforms -> at(k-1).M_k * RBK::mrp_to_dcm(dT.subvec(6 *(k - 1) + 3, 6 * (k - 1) + 5));
		arma::mat M_tilde_k_p = this -> absolute_rigid_transforms -> at(k).M_k * RBK::mrp_to_dcm(dT.subvec(6 * k + 3, 6 * k + 5));

		arma::mat M_p_k = RBK::mrp_to_dcm(this -> mrps_LN[k - 1]).t() * M_tilde_km1_p .t() * M_tilde_k_p * RBK::mrp_to_dcm(this -> mrps_LN[k]);
		arma::vec X_p_k = RBK::mrp_to_dcm(this -> mrps_LN[k - 1]).t() * M_tilde_km1_p .t() * (X_tilde_k_p - X_tilde_km1_p);


		X_p_k = X_p_k + dI.subvec(6 * (k - 1) , 6 * (k - 1) + 2 );
		M_p_k = M_p_k * RBK::mrp_to_dcm(dI.subvec(6 * (k - 1) + 3, 6 * (k - 1) + 2 + 3));


		RigidTransform rigid_transform;
		rigid_transform.M_k = M_p_k;
		rigid_transform.X_k = X_p_k;
		rigid_transform.t_k = this -> sequential_rigid_transforms -> at(k - 1).t_k;
		rigid_transforms_seq_p.push_back(rigid_transform);

		I_p.subvec(6 * (k-1), 6 * (k-1) + 2) = X_p_k;
		I_p.subvec(6 * (k-1) + 3, 6 * (k-1) + 5) = RBK::dcm_to_mrp(M_p_k) ;

		V_p.subvec(6 + 12 * (k - 1) , 6 + 12 * (k - 1) + 2) = X_tilde_km1_p;
		V_p.subvec(6 + 12 * (k - 1) + 3, 6 + 12 * (k - 1) + 5) = RBK::dcm_to_mrp(M_tilde_km1_p);

		V_p.subvec(6 + 12 * (k - 1) + 6, 6 + 12 * (k - 1) + 8) = X_tilde_k_p;
		V_p.subvec(6 + 12 * (k - 1) + 9, 6 + 12 * (k - 1) + 11) = RBK::dcm_to_mrp(M_tilde_k_p);

		dV_non_lin.subvec(6 + 12 * (k - 1) , 6 + 12 * (k - 1) + 2) = V_p.subvec(6 + 12 * (k - 1) , 6 + 12 * (k - 1) + 2) - V_nom.subvec(6 + 12 * (k - 1) , 6 + 12 * (k - 1) + 2) ;
		dV_non_lin.subvec(6 + 12 * (k - 1) + 3, 6 + 12 * (k - 1) + 5) = RBK::dcm_to_mrp(RBK::mrp_to_dcm(V_nom.subvec(6 + 12 * (k - 1) + 3, 6 + 12 * (k - 1) + 5)).t() * RBK::mrp_to_dcm(V_p.subvec(6 + 12 * (k - 1) + 3, 6 + 12 * (k - 1) + 5)));

		dV_non_lin.subvec(6 + 12 * (k - 1) + 6, 6 + 12 * (k - 1) + 8) = V_p.subvec(6 + 12 * (k - 1) + 6, 6 + 12 * (k - 1) + 8) - V_nom.subvec(6 + 12 * (k - 1) + 6, 6 + 12 * (k - 1) + 8);
		dV_non_lin.subvec(6 + 12 * (k - 1) + 9, 6 + 12 * (k - 1) + 11) = RBK::dcm_to_mrp(RBK::mrp_to_dcm(V_nom.subvec(6 + 12 * (k - 1) + 9, 6 + 12 * (k - 1) + 11)).t() * RBK::mrp_to_dcm(V_p.subvec(6 + 12 * (k - 1) + 9, 6 + 12 * (k - 1) + 11)));

		dI_non_lin.subvec(6 * (k-1), 6 * (k-1) + 2) = I_p.subvec(6 * (k-1), 6 * (k-1) + 2) - I_nom.subvec(6 * (k-1), 6 * (k-1) + 2);
		dI_non_lin.subvec(6 * (k-1) + 3 , 6 * (k-1) + 5) = RBK::dcm_to_mrp(RBK::mrp_to_dcm(I_nom.subvec(6 * (k-1) + 3, 6 * (k-1) + 5)).t() * RBK::mrp_to_dcm( I_p.subvec(6 * (k-1) + 3, 6 * (k-1) + 5)));


	}	


	// perturbed y
	arma::vec y_p = arma::zeros<arma::vec>(3 * rigid_transforms_seq_p. size());

	for (int k = 0; k < rigid_transforms_seq_p. size(); ++ k){

		arma::mat Mkp1 = rigid_transforms_seq_p.at(k).M_k;
		arma::vec Xkp1 = rigid_transforms_seq_p.at(k).X_k;

		y_p.rows(3 * k, 3 * k + 2) = IODFinder::compute_y_k(
			positions[k],
			positions[k+1],
			Mkp1,
			Xkp1);
	}


	// arma::vec dV = this -> compute_partial_V_partial_T() * dT;


	// std::cout << "Original dI : \n";
	// std::cout << dI << std::endl;

	// dI += this -> compute_partial_I_partial_V() * dV;

	// std::cout << "dI from dV : \n";
	// std::cout <<  this -> compute_partial_I_partial_V() * dV << std::endl;

	// arma::mat dydI = this -> compute_partial_y_partial_I(positions);

	// arma::vec dy_lin = dydI * dI;


	// arma::vec dy_non_lin = y_p - y_nom;

	// std::cout << "dy_lin: \n";
	// std::cout << dy_lin<< std::endl << std::endl;


	// std::cout << "dy_non_lin: \n";
	// std::cout << dy_non_lin<< std::endl << std::endl;

	// std::cout <<  "\t dy_non_lin - dy_lin (%) \n";
	// std::cout << arma::abs(dy_non_lin - dy_lin)/arma::max(arma::abs(dy_non_lin)) * 100 << std::endl;








	arma::mat dVdT =  this -> compute_partial_V_partial_T();




	arma::vec dV_lin = dVdT * dT;

	std::cout <<  "\tdV_non_lin" << std::endl << std::endl;
	std::cout << dV_non_lin << std::endl;

	std::cout <<  "\tdV_lin" << std::endl << std::endl;

	std::cout << dV_lin << std::endl;
	std::cout <<  "\t dV_non_lin - dV_lin = \n";
	std::cout << dV_non_lin - dV_lin << std::endl;

	std::cout <<  "\t Max dV_non_lin - dV_lin (%) = " << arma::max(arma::abs((dV_non_lin - dV_lin))/dV_non_lin) * 100 << std::endl;

	arma::mat dIdT = this -> compute_partial_I_partial_V() * dVdT;

	arma::vec dI_lin = dIdT * dT;

	std::cout <<  "\tdI_non_lin" << std::endl << std::endl;
	std::cout << dI_non_lin << std::endl;

	std::cout <<  "\tdI_lin" << std::endl << std::endl;

	std::cout << dI_lin << std::endl;

	std::cout <<  "\t dI_non_lin - dI_lin (%) = \n";
	std::cout << (dI_non_lin - dI_lin)/dI_lin * 100 << std::endl;

	std::cout <<  "\t Max dI_non_lin - dI_lin (%) = " << arma::max(arma::abs((dI_non_lin - dI_lin))/dI_lin) * 100 << std::endl;

	arma::vec dy_non_lin =  y_p - y_nom;
	arma::vec dy_lin = this -> compute_partial_y_partial_I(positions) * dI_lin;


	std::cout << "dy_non_lin\n ";
	std::cout << dy_non_lin << std::endl;


	std::cout << "dy_lin\n ";
	std::cout << dy_lin << std::endl;


	std::cout <<  "\t dy_non_lin - dy_lin  \n";
	std::cout << arma::abs(dy_non_lin - dy_lin) << std::endl;


	std::cout <<  "\t Max dy_non_lin - dy_lin (%) \n";
	std::cout << arma::abs(dy_non_lin - dy_lin)/dy_non_lin * 100 << std::endl;



	// arma::mat dydT = IODFinder::compute_partial_y_partial_T(positions);



	// arma::vec dy_non_lin =  y_p - y_nom;
	// arma::vec dy_lin = dydT * dT;

	// std::cout <<  "\tdy_non_lin" << std::endl << std::endl;
	// std::cout << dy_non_lin << std::endl;

	// std::cout <<  "\tdy_lin" << std::endl << std::endl;

	// std::cout << dy_lin << std::endl;
	// std::cout <<  "\t dy_non_lin - dy_lin \n";
	// std::cout << dy_non_lin - dy_lin << std::endl;

	// double cost_fun_from_y = arma::dot(y_nom,y_nom);
	// double cost_fun_from_yp = arma::dot(y_p,y_p);

	// double dcost_lin = 2 * arma::dot(y_nom, dy_lin);
	// std::cout << "\t cost_fun_from_y: " << cost_fun_from_y << std::endl;

	// std::cout << "\t cost_fun_from_yp: " << cost_fun_from_yp << std::endl;

	// std::cout << "\t dcost non lin: " << cost_fun_from_yp - cost_fun_from_y << std::endl;
	// std::cout << "\t dcost lin: " << dcost_lin + arma::dot(dy_lin,dy_lin) << std::endl;



	throw;
}


void IODFinder::compute_P_T(){

	this -> P_T = arma::zeros<arma::mat>(6 * this -> absolute_rigid_transforms -> size() ,6 * this -> absolute_rigid_transforms -> size());

	for (int i = 1; i < this -> absolute_rigid_transforms -> size(); ++i){
		this -> P_T.submat(6 * i, 6 * i, 6 * i + 2, 6 * i + 2) = std::pow(this -> stdev_Xtilde,2) * arma::eye<arma::mat>(3,3);
		this -> P_T.submat(6 * i + 3, 6 * i + 3, 6 * i + 5, 6 * i + 5) = std::pow(this -> stdev_sigmatilde,2) * arma::eye<arma::mat>(3,3);
	}

	this -> P_T. save("../output/P_T.txt",arma::raw_ascii);

}

void IODFinder::compare_rigid_transforms(std::vector<RigidTransform> * s1,std::vector<RigidTransform> * s2){


	if (s1 -> size () != s2 -> size()){
		std::cout << "The two sequences have different lengths: " << s1 -> size() << " and " << s2 -> size() << std::endl;
		return;
	}


	for (int i = 0; i < s1 -> size(); ++i){
		std::cout << i << std::endl;
		arma::mat dM = s1 -> at(i).M_k.t() * s2-> at(i).M_k;
		arma::vec dX = s1 -> at(i).X_k - s2-> at(i).X_k;
		double dt = s1 -> at(i).t_k - s2-> at(i).t_k;

		std::cout << arma::norm(RBK::dcm_to_prv(dM)) << std::endl;
		std::cout << arma::norm(dX) << std::endl ;
		std::cout << dt << std::endl << std::endl;

	}



}







