#include "IODFinder.hpp"
#include "Psopt.hpp"
#include "IODBounds.hpp"
#include <RigidBodyKinematics.hpp>
#include "System.hpp"
#include "Args.hpp"
#include "Dynamics.hpp"
#include "Observer.hpp"

IODFinder::IODFinder(std::vector<RigidTransform> * rigid_transforms, 
	std::vector<arma::vec> mrps_LN,
	double stdev_Xtilde,
	double stdev_sigmatilde,
	int N_iter, 
	int particles ){

	this -> N_iter = N_iter;
	this -> particles = particles;
	this -> rigid_transforms = rigid_transforms;
	this -> mrps_LN = mrps_LN;
	this -> stdev_Xtilde = stdev_Xtilde;
	this -> stdev_sigmatilde = stdev_sigmatilde;


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
			this -> rigid_transforms,
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
			this -> rigid_transforms,
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

double IODFinder::cost_function_cartesian(arma::vec particle, std::vector<RigidTransform> * args,int verbose_level){

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







arma::mat::fixed<6,6> IODFinder::compute_P_Ik_Ij(int k, int j) const{

	return (IODFinder::compute_dIprime_k_dVtilde_k(k) 
		* IODFinder::compute_P_VkVj(k,j) 
		* IODFinder::compute_dIprime_k_dVtilde_k(j).t());
	

}





arma::mat::fixed<6,12> IODFinder::compute_dIprime_k_dVtilde_k(int k) const{

	arma::vec Xk,Xkm1;
	arma::mat Mk,Mkm1,LN_km1,LN_k;


	Xk = this -> rigid_transforms -> at(k).X_k;
	Mk =  this -> rigid_transforms -> at(k).M_k;
	LN_k = RBK::mrp_to_dcm(this -> mrps_LN.at(k));

	if (k > 0){
		Xkm1 = this -> rigid_transforms -> at(k- 1).X_k;
		Mkm1 =  this -> rigid_transforms -> at(k- 1).M_k;
		LN_km1 = RBK::mrp_to_dcm(this -> mrps_LN.at(k - 1));

	}
	else{
		Xkm1 = arma::zeros<arma::vec>(3);
		Mkm1 = arma::eye<arma::mat>(3,3);
		LN_km1 = arma::eye<arma::mat>(3,3);

	}

	arma::vec::fixed<3> a_k_bar = Mkm1.t() * (Xk - Xkm1);
	arma::mat::fixed<3,3> Uk_bar = -4 * RBK::tilde(a_k_bar);

	arma::mat::fixed<6,12> partial_I_prime_k_partial_Vtilde_k = arma::zeros<arma::mat>(6,12);

	partial_I_prime_k_partial_Vtilde_k.submat(0,0,2,2) = LN_km1.t() * Mkm1.t();
	partial_I_prime_k_partial_Vtilde_k.submat(0,3,2,5) = - LN_km1.t() * Mkm1.t();
	partial_I_prime_k_partial_Vtilde_k.submat(0,9,2,11) = LN_km1.t() * Uk_bar;

	partial_I_prime_k_partial_Vtilde_k.submat(3,6,5,11) = IODFinder::compute_dsigmatilde_kdZ_k(
		Mk,
		Mkm1,
		LN_k,
		LN_km1);

	return partial_I_prime_k_partial_Vtilde_k;

}






arma::mat::fixed<3,6> IODFinder::compute_dsigmatilde_kdZ_k(
	const arma::mat::fixed<3,3> & M_k_tilde_bar,
	const arma::mat::fixed<3,3> & M_km1_tilde_bar,
	const arma::mat::fixed<3,3> & LN_k,
	const arma::mat::fixed<3,3> & LN_km1){

	arma::mat::fixed<3,3> Abar_k = M_km1_tilde_bar.t() * M_k_tilde_bar;
	arma::mat::fixed<3,6> partial_sigmatildek_partial_Zk;

	arma::vec e0 = {1,0,0};
	arma::vec e1 = {0,1,0};
	arma::vec e2 = {0,0,1};

	partial_sigmatildek_partial_Zk.submat(0,0,0,2) = -e2.t()*M_k_tilde_bar.t() * LN_km1.t() * Abar_k * RBK::tilde(LN_k * e1);
	partial_sigmatildek_partial_Zk.submat(1,0,1,2) = -e0.t()*M_k_tilde_bar.t() * LN_km1.t() * Abar_k * RBK::tilde(LN_k * e2);
	partial_sigmatildek_partial_Zk.submat(2,0,2,2) = -e1.t()*M_k_tilde_bar.t() * LN_km1.t() * Abar_k * RBK::tilde(LN_k * e0);

	partial_sigmatildek_partial_Zk.submat(0,3,0,5) = e2.t()*M_k_tilde_bar.t() * LN_km1.t() * RBK::tilde(Abar_k * LN_k * e1);
	partial_sigmatildek_partial_Zk.submat(1,3,1,5) = e0.t()*M_k_tilde_bar.t() * LN_km1.t() * RBK::tilde(Abar_k * LN_k * e2);
	partial_sigmatildek_partial_Zk.submat(2,3,2,5) = e1.t()*M_k_tilde_bar.t() * LN_km1.t() * RBK::tilde(Abar_k * LN_k * e0);

	return partial_sigmatildek_partial_Zk;
}



arma::mat::fixed<3,6> IODFinder::compute_J_k(int k,const std::vector<arma::vec> & positions) const{

	arma::mat::fixed<3,6> J ;

	J.submat(0,0,2,2) = - arma::eye<arma::mat>(3,3);
	J.submat(0,3,2,5) = 4 * rigid_transforms -> at(k).M_k * RBK::tilde(positions.at(k + 1));

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



arma::mat::fixed<12,12> IODFinder::compute_P_VkVj(int k, int j) const{

	arma::mat P_VkVj = arma::zeros<arma::mat>(12,12);

	if (std::abs(k - j) <= 1){


		std::vector<int> r_indices = {k,k-1,k,k-1};
		std::vector<int> c_indices = {j,j-1,j,j-1};

		for (int r = 0; r < r_indices.size(); ++r){
			int r_index = r_indices[r];

			if (r_index < 0){
				continue;
			}

			for (int c = 0; c < c_indices.size() ; ++c){
				int c_index = c_indices[c];
				
				if (c_index < 0){
					continue;
				}

				if (r_index == c_index){

					if (r < 2 && c < 2){
						P_VkVj.submat(3 * r, 3 * c, 
							3 * r + 2, 3 * c + 2) = std::pow(this -> stdev_Xtilde,2) * arma::eye<arma::mat>(3,3);
					}
					else if (r > 2 && c > 2){
						P_VkVj.submat(3 * r, 3 * c, 
							3 * r + 2, 3 * c + 2) = std::pow(this -> stdev_sigmatilde,2) * arma::eye<arma::mat>(3,3);
					}

				}

			}

		}

	}

	return P_VkVj;

}




void IODFinder::build_normal_equations(
	arma::mat & info_mat,
	arma::vec & normal_mat,
	arma::vec & residual_vector,
	arma::vec & apriori_state,
	const std::vector<arma::vec> & positions,
	const std::vector<arma::mat> & stms) const{

	residual_vector.fill(0);

	arma::mat H = arma::zeros<arma::mat>(3 * this -> rigid_transforms -> size(), 7);
	
	// H matrices
	// y vector
	// R matrix
	for (int k = 0; k < this -> rigid_transforms -> size(); ++ k){

		arma::mat Mkp1 = this -> rigid_transforms -> at(k).M_k;
		arma::vec Xkp1 = this -> rigid_transforms -> at(k).X_k;

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
	arma::vec residual_vector = arma::vec(3 * this -> rigid_transforms -> size());


	std::vector<arma::vec> positions;
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
	std::vector<arma::vec> & positions,
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

	for (int i = 0; i < this -> rigid_transforms -> size(); ++i){
		times.push_back(this -> rigid_transforms -> at(i).t_k);
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




void IODFinder::compute_W(const std::vector<arma::vec> & positions){

	arma::mat R = arma::zeros<arma::mat>(3* this -> rigid_transforms -> size(), 
		3 * this -> rigid_transforms -> size());

	for (int k = 0 ; k < this -> rigid_transforms -> size(); ++k){

		for (int j = 0 ; j <= k; ++j){

			arma::mat Rkj = this -> compute_Rkj(k,j,positions);

			R.submat(3 * k, 3 * j, 3 * k + 2, 3 * j + 2) = Rkj;
			R.submat(3 * j, 3 * k, 3 * j + 2, 3 * k + 2) = Rkj.t();

		}

	}
	
	R.save("../output/R_mat.txt",arma::raw_ascii);
	this -> W = arma::inv(R);


}


arma::mat::fixed<3,3> IODFinder::compute_Rkj(int k,int j,const std::vector<arma::vec> & positions) const{

	return (IODFinder::compute_J_k(k,positions) 
		* IODFinder::compute_P_Ik_Ij(k,j) 
		* IODFinder::compute_J_k(j,positions).t());

}





void IODFinder::debug_rigid_transforms(){



	// RigidTransform epoch_transform_k_nom;
	// RigidTransform epoch_transform_km1_nom;


	// RigidTransform epoch_transform_k_perp;
	// RigidTransform epoch_transform_km1_perp;

	// RigidTransform seq_transform_k_nom;
	// RigidTransform seq_transform_k_perp;


	// std::vector<arma::vec> mrps_LN;

	// mrps_LN.push_back(arma::randn<arma::vec>(3));
	// mrps_LN.push_back(arma::randn<arma::vec>(3));

	// arma::vec mrp_k = {0.4,0.2,0.1};
	// arma::vec mrp_km1 = {-0.4,-0.2,0.1};

	// epoch_transform_k_nom.X_k = {1,0,0};
	// epoch_transform_k_nom.M_k = RBK::mrp_to_dcm(mrp_k);

	// epoch_transform_km1_nom.X_k = {-1,0.5,0};
	// epoch_transform_km1_nom.M_k = RBK::mrp_to_dcm(mrp_k);


	// IODFinder::seq_transform_from_epoch_transform(1, 
	// 	seq_transform_k_nom,
	// 	epoch_transform_k_nom,
	// 	epoch_transform_km1_nom, 
	// 	mrps_LN);

	// arma::vec dX_epoch_k = 0.001 * epoch_transform_k_nom.X_k;
	// arma::vec dX_epoch_km1 = 0.001 * epoch_transform_km1_nom.X_k;

	// arma::vec dmrp_epoch_k = 0.0 * mrp_k;
	// arma::vec dmrp_epoch_km1 = 0.0 * mrp_km1;

	// epoch_transform_k_perp.X_k = epoch_transform_k_nom.X_k +  dX_epoch_k;
	// epoch_transform_km1_perp.X_k = epoch_transform_km1_nom.X_k + dX_epoch_km1;

	// epoch_transform_k_perp.M_k = epoch_transform_k_nom.M_k *RBK::mrp_to_dcm(dmrp_epoch_k) ;
	// epoch_transform_km1_perp.M_k = epoch_transform_km1_nom.M_k * RBK::mrp_to_dcm(dmrp_epoch_km1) ;


	// IODFinder::seq_transform_from_epoch_transform(1, 
	// 	seq_transform_k_perp,
	// 	epoch_transform_k_perp,
	// 	epoch_transform_km1_perp, 
	// 	mrps_LN);

	// arma::vec dX_seq = seq_transform_k_perp.X_k - seq_transform_k_nom.X_k;


	// arma::mat partial = IODFinder::compute_dIprime_k_dVtilde_k(
	// 	epoch_transform_k_nom.M_k,
	// 	epoch_transform_k_nom.X_k,
	// 	epoch_transform_km1_nom.M_k,
	// 	epoch_transform_km1_nom.X_k,
	// 	RBK::mrp_to_dcm(mrps_LN[1]),
	// 	RBK::mrp_to_dcm(mrps_LN[0]));


	// arma::vec dV = arma::zeros<arma::vec>(12);

	// dV.subvec(0,2) = dX_epoch_k;
	// dV.subvec(3,5) = dX_epoch_km1;

	// std::cout << dX_seq.t() << std::endl;
	// std::cout << (partial * dV).t() << std::endl;

	// throw;

}











