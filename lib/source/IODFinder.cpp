#include "IODFinder.hpp"
#include "Psopt.hpp"
#include "IODBounds.hpp"
#include <RigidBodyKinematics.hpp>
#include "Args.hpp"
#include "Dynamics.hpp"
#include "Observer.hpp"
#include "SystemDynamics.hpp"



IODFinder::IODFinder(std::vector<RigidTransform> * sequential_rigid_transforms,
	std::vector<RigidTransform> * absolute_rigid_transforms,  
	std::vector<arma::vec::fixed<3> > mrps_LN,
	int N_iter, 
	int N_particles){

	this -> N_iter = N_iter;
	this -> N_particles = N_particles;
	this -> sequential_rigid_transforms = sequential_rigid_transforms;
	this -> absolute_rigid_transforms = absolute_rigid_transforms;
	this -> mrps_LN = mrps_LN;
	// this -> debug_R();

	
}

void IODFinder::run_pso(
	arma::vec lower_bounds,
	arma::vec upper_bounds,
	int verbose_level,
	const arma::vec & guess){
	
	Psopt<std::vector<RigidTransform> *> psopt(
		IODFinder::cost_function_cartesian, 
		lower_bounds,
		upper_bounds, 
		this -> N_particles,
		this -> N_iter,
		this -> sequential_rigid_transforms,
		guess);
	
	psopt.run(false,verbose_level);
	this -> state_at_epoch = psopt.get_result();


}

arma::vec IODFinder::get_result() const{
	return this -> state_at_epoch;
}

double IODFinder::cost_function_cartesian(
	const arma::vec & particle, 
	std::vector<RigidTransform> * args,
	int verbose_level){

	// Particle State ordering:
	// [x,y,z,x_dot,y_dot,z_dot,mu]
	OC::CartState cart_state(particle.subvec(0,5),particle(6));
	OC::KepState kep_state = cart_state.convert_to_kep(0);

	// There are N sequential rigid transforms
	int N = args -> size();
	arma::mat positions(3,N + 1);

	// The epoch time is taken as the time when the first point
	// cloud in the considered rigid transform was collected

	double epoch_time = args -> front().t_start;

	if (verbose_level > 1){
		std::cout << "\n - Epoch time: " << epoch_time;
		std::cout << "\n - Epoch index: " << args -> front().index_start;

	}

	positions.col(0) = cart_state.get_position_vector();

	// For all the sequential rigid transforms
	for (int i = 0; i < N ; ++i){
		
		double t = args -> at(i).t_end;
		double time_from_epoch = t - epoch_time;

		positions.col(i + 1) = kep_state.convert_to_cart(time_from_epoch).get_position_vector();

		if (verbose_level > 1){
			std::cout << " - Time from 0 : " << t << std::endl;
			std::cout << " - Time from epoch : " << time_from_epoch << std::endl << std::endl;
		}

	}


	arma::vec epsilon = arma::zeros<arma::vec>(3 * N);

	for (int k = 0; k < N; ++k ){

		arma::mat M_k = args -> at(k).M;
		arma::mat X_k = args -> at(k).X;

		epsilon.subvec( 3 * k, 3 * k + 2) = positions.col(k) - M_k * positions.col(k + 1) + X_k;
		
		if (verbose_level > 1){
			std::cout << "\t Epsilon " << k << " = " << arma::norm(epsilon.subvec( 3 * k, 3 * k + 2)) << std::endl;

		}

	}

	if (verbose_level > 2){
		std::cout << positions.col(0).t() << std::endl;
		std::cout << positions.col(1).t() << std::endl;
		std::cout << args -> at(0).M << std::endl;
		std::cout << args -> at(0).X << std::endl;
	}


	return std::sqrt(arma::dot(epsilon,epsilon)/epsilon.size());

}

arma::mat::fixed<6,12> IODFinder::compute_dIprime_k_dVtilde_k(int k) const{

	arma::vec Xk,Xkm1;
	arma::mat Mk,Mkm1,LN_km1,LN_k;
	arma::mat::fixed<6,12> partial_I_prime_k_partial_Vtilde_k = arma::zeros<arma::mat>(6,12);

	Xk = this -> absolute_rigid_transforms -> at(k).X;
	Mk =  this -> absolute_rigid_transforms -> at(k).M;
	LN_k = RBK::mrp_to_dcm(this -> mrps_LN.at(k));

	if (k > 0){
		Xkm1 = this -> absolute_rigid_transforms -> at(k- 1).X;
		Mkm1 =  this -> absolute_rigid_transforms -> at(k- 1).M;
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
		this -> sequential_rigid_transforms -> at (k - 1).M,
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
	J.submat(0,3,2,5) = - 4 * this -> sequential_rigid_transforms -> at(k).M * RBK::tilde(positions.at(k + 1));
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


arma::sp_mat IODFinder::compute_partial_y_partial_T(const std::vector<arma::vec::fixed<3>> & positions) const{

	return (this -> compute_partial_y_partial_I(positions)
		*  this -> compute_partial_I_partial_V() 
		* this -> compute_partial_V_partial_T());

}

arma::sp_mat IODFinder::compute_partial_y_partial_I(const std::vector<arma::vec::fixed<3>> & positions) const{



	arma::sp_mat dydI(3 * this -> sequential_rigid_transforms -> size(),
		6 * this -> sequential_rigid_transforms -> size());

	for (int k = 0; k < this -> sequential_rigid_transforms -> size(); ++k){
		dydI.submat(3 * k, 6 * k, 3 * k + 2, 6 * k + 5) = this -> compute_J_k(k,positions);
	}

	return dydI;
}

arma::sp_mat IODFinder::compute_partial_I_partial_V() const{

	arma::sp_mat dIdV(6 * this -> sequential_rigid_transforms -> size(),
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

arma::sp_mat IODFinder::compute_partial_V_partial_T() const{


	arma::sp_mat dVdT(6 + 12 * this -> sequential_rigid_transforms -> size() ,
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
	const std::vector<arma::vec::fixed<3>> & positions,
	const std::vector<arma::mat> & stms) const{

	residual_vector.fill(0);

	arma::mat H = arma::zeros<arma::mat>(3 * this -> sequential_rigid_transforms -> size(), 7);
	
	// H matrices
	// y vector
	// R matrix

	for (int k = 0; k < this -> sequential_rigid_transforms -> size(); ++ k){

		arma::mat Mkp1 = this -> sequential_rigid_transforms -> at(k).M;
		arma::vec Xkp1 = this -> sequential_rigid_transforms -> at(k).X;

		H.rows(3 * k, 3 * k + 2) = IODFinder::compute_H_k(stms[k],stms[k+1],Mkp1);
		residual_vector.rows(3 * k, 3 * k + 2) = IODFinder::compute_y_k(positions[k],positions[k+1],Mkp1,Xkp1);
	}

	info_mat = H.t() * this -> W * H;
	normal_mat = H.t() * this -> W * residual_vector;

}


void IODFinder::run_batch(
	arma::vec & epoch_state,
	arma::vec & final_state,
	arma::mat & epoch_cov,
	arma::mat & final_cov,
	const std::map<int, arma::mat::fixed<6,6> > & R_pcs){

	int N_iter = 1;

	arma::mat info_mat(7,7);
	arma::vec normal_mat(7);
	arma::vec residual_vector = arma::vec(3 * this -> sequential_rigid_transforms -> size());

	std::vector<arma::vec::fixed<3> > positions,velocities;
	std::vector<arma::mat> stms;

	double old_residuals = std::numeric_limits<double>::infinity();

	this -> compute_P_T(R_pcs);
	for (int i = 0; i < N_iter; ++i){

		this -> compute_state_stms(epoch_state,positions,velocities,stms);
		this -> compute_W(positions);
		this -> build_normal_equations(info_mat,normal_mat,residual_vector,positions,stms);
		
		double new_residuals = std::sqrt(arma::dot(residual_vector,residual_vector)/residual_vector.size());
		std::cout << "\tResiduals RMS: " << new_residuals<< std::endl;
		

		try{
			epoch_state += arma::solve(info_mat,normal_mat);


			final_state.subvec(0,2) = positions.back();
			final_state.subvec(3,5) = velocities.back();
			final_state(6) = epoch_state(6);

		}
		catch(std::runtime_error & e){
			e.what();
		}
		std::cout << "\tState: " << epoch_state.t() << std::endl;

		if (std::abs(new_residuals - old_residuals)/new_residuals * 100 < 1e-2){
			std::cout << "\tBatch has converged after " << i + 1 << " iterations\n";
			break;
		}
		else{
			old_residuals = new_residuals;
		}

	}



	try{
		epoch_cov = arma::inv(info_mat);
		final_cov = stms.back() * epoch_cov * stms.back().t();

	}
	catch(std::runtime_error & e){
		e.what();
	}

}


void IODFinder::compute_state_stms(const arma::vec::fixed<7> & X_hat,
	std::vector<arma::vec::fixed<3> > & positions,
	std::vector<arma::vec::fixed<3> > & velocities,
	std::vector<arma::mat> & stms) const{

	positions.clear();
	velocities.clear();
	stms.clear();

	int N_est = X_hat.n_rows;

	Args args;

	
	SystemDynamics dynamics_system(args);

	dynamics_system.add_next_state("position",3,false);
	dynamics_system.add_next_state("velocity",3,false);
	dynamics_system.add_next_state("mu",1,false);


	dynamics_system.add_dynamics("position",Dynamics::velocity,{"velocity"});
	dynamics_system.add_dynamics("velocity",Dynamics::point_mass_acceleration,{"position","mu"});

	dynamics_system.add_jacobian("position","velocity",Dynamics::identity_33,{"velocity"});
	dynamics_system.add_jacobian("velocity","position",Dynamics::point_mass_gravity_gradient_matrix,{"position","mu"});
	dynamics_system.add_jacobian("velocity","mu",Dynamics::point_mass_acceleration_unit_mu,{"position"});


	arma::vec x(N_est + N_est * N_est);
	x.rows(0,N_est - 1) = X_hat;
	x.rows(N_est,N_est + N_est * N_est - 1) = arma::vectorise(arma::eye<arma::mat>(N_est,N_est));

	std::vector<arma::vec> augmented_state_history;
	typedef boost::numeric::odeint::runge_kutta_cash_karp54< arma::vec > error_stepper_type;
	auto stepper = boost::numeric::odeint::make_controlled<error_stepper_type>( 1.0e-13 ,1.0e-16 );

	std::vector<double> times;

	times.push_back(this -> sequential_rigid_transforms -> at(0).t_start);
	for (int i = 0 ; i < this -> sequential_rigid_transforms -> size(); ++i){
		times.push_back(this -> sequential_rigid_transforms -> at(i).t_end);
	}

	auto tbegin = times.begin();
	auto tend = times.end();
	boost::numeric::odeint::integrate_times(stepper, dynamics_system, x, tbegin, tend,1e-10,
		Observer::push_back_state(augmented_state_history,
			dynamics_system.get_number_of_states(),
			dynamics_system.get_attitude_state_first_indices()));

	for (int i = 0; i < times.size(); ++i){

		arma::mat::fixed<7,7> stm = arma::reshape(augmented_state_history[i].rows(N_est,N_est + N_est * N_est - 1),N_est,N_est);
		positions.push_back(augmented_state_history[i].rows(0,2));
		velocities.push_back(augmented_state_history[i].rows(3,5));
		stms.push_back(stm);
	}

}




void IODFinder::compute_W(const std::vector<arma::vec::fixed<3>> & positions){

	arma::sp_mat dydT = IODFinder::compute_partial_y_partial_T(positions);

	arma::mat R = dydT * this -> P_T * dydT.t();

	this -> W = arma::inv(R);


}

void IODFinder::compute_P_T(const std::map<int, arma::mat::fixed<6,6> > & R_pcs){

	this -> P_T = arma::zeros<arma::mat>(6 * this -> absolute_rigid_transforms -> size() ,6 * this -> absolute_rigid_transforms -> size());

	for (int i = 1; i < this -> absolute_rigid_transforms -> size(); ++i){
		this -> P_T.submat(6 * i, 6 * i, 6 * i + 2, 6 * i + 2) = R_pcs.at(i).submat(0,0,2,2);
		this -> P_T.submat(6 * i + 3, 6 * i + 3, 6 * i + 5, 6 * i + 5) = R_pcs.at(i).submat(3,3,5,5);
	}


}


arma::rowvec::fixed<2> IODFinder::partial_rp_partial_ae(const double & a, const double & e){
	arma::rowvec::fixed<2> drp_dae = {1 - e,- a };
	return drp_dae;
}

arma::mat::fixed<2,4> IODFinder::partial_ae_partial_aevec(const double & a, const arma::vec::fixed<3> & e){

	arma::mat::fixed<2,4> partial = arma::zeros<arma::mat>(2,4);
	partial(0,0) = 1;
	partial.submat(1,1,1,3) = arma::normalise(e).t();

	return partial;
}


arma::rowvec::fixed<3> IODFinder::partial_a_partial_rvec(const double & a, const arma::vec::fixed<3> & r){
	return 2 * std::pow(a,2) * r.t() / std::pow(arma::norm(r),3);
}	


arma::rowvec::fixed<3> IODFinder::partial_a_partial_rdotvec(const double & a, const arma::vec::fixed<3> & r_dot, const double & mu){

	return 2 * std::pow(a,2) /  mu * r_dot.t();
}

double IODFinder::partial_a_partial_mu(const double & a, const arma::vec::fixed<3> & r_dot, const double & mu){
	return  (- std::pow(a/ ( mu) ,2)  * arma::dot(r_dot,r_dot));
}

arma::rowvec::fixed<7> IODFinder::partial_a_partial_state(const double & a, const arma::vec::fixed<7> & state ){


	arma::rowvec::fixed<7> partial; 
	partial.subvec(0,2) = IODFinder::partial_a_partial_rvec(a,state.subvec(0,2));
	partial.subvec(3,5) = IODFinder::partial_a_partial_rdotvec(a,state.subvec(3,5),state(6));
	partial(6) = IODFinder::partial_a_partial_mu(a,state.subvec(3,5),state(6));

	return partial;

}



arma::mat::fixed<3,3> IODFinder::partial_evec_partial_r(const arma::vec::fixed<7> & state){

	return (1./state(6) * (arma::dot(state.subvec(3,5),state.subvec(3,5)) * arma::eye<arma::mat>(3,3)
		- state.subvec(3,5) * state.subvec(3,5).t())
	+ 1./arma::norm(state.subvec(0,2)) * (state.subvec(0,2) * state.subvec(0,2).t() / arma::dot(state.subvec(0,2),state.subvec(0,2))
		- arma::eye<arma::mat>(3,3)));

}
arma::mat::fixed<3,3> IODFinder::partial_evec_partial_rdot(const arma::vec::fixed<7> & state){


	return (1./state(6) * (2 * state.subvec(0,2) * state.subvec(3,5).t() - arma::dot(state.subvec(0,2),
		state.subvec(3,5)) * arma::eye<arma::mat>(3,3) - state.subvec(3,5) * state.subvec(0,2).t()));

}
arma::vec::fixed<3> IODFinder::partial_evec_partial_mu(const arma::vec::fixed<7> & state){

	return (- arma::cross(state.subvec(3,5),arma::cross(state.subvec(0,2),state.subvec(3,5)))/std::pow(state(6),2));

}


arma::rowvec::fixed<7> IODFinder::partial_rp_partial_state(const arma::vec::fixed<7> & state ){

	OC::CartState cart_state(state.subvec(0,5),state(6));
	OC::KepState kep_state = cart_state.convert_to_kep(0);

	double a = kep_state.get_a();
	double e = kep_state.get_eccentricity();
	double mu = kep_state.get_mu();

	arma::vec evec = (arma::cross(state.subvec(3,5),arma::cross(state.subvec(0,2),state.subvec(3,5)))/mu 
		- arma::normalise(state.subvec(0,2)));

	return (IODFinder::partial_rp_partial_ae(a, e) * IODFinder::partial_ae_partial_aevec(a,evec)
		* IODFinder::partial_aevec_partial_state(a,state));
}


arma::mat::fixed<4,7> IODFinder::partial_aevec_partial_state(const double & a,const arma::vec::fixed<7> & state){

	arma::mat::fixed<4,7> partial;

	partial.row(0) = IODFinder::partial_a_partial_state(a,state);
	partial.rows(1,3) = IODFinder::partial_evec_partial_state(state);
	return partial;

}

arma::mat::fixed<3,7> IODFinder::partial_evec_partial_state(const arma::vec::fixed<7> & state){

	arma::mat::fixed<3,7> partial;
	partial.cols(0,2) = IODFinder::partial_evec_partial_r(state);
	partial.cols(3,5) = IODFinder::partial_evec_partial_rdot(state);
	partial.col(6) = IODFinder::partial_evec_partial_mu(state);

	return partial;

}


