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
	std::vector<arma::vec::fixed<3> > mrps_LN,
	double stdev_Xtilde,
	double stdev_sigmatilde,
	int N_iter, 
	int particles){

	this -> N_iter = N_iter;
	this -> N_particles = particles;
	this -> sequential_rigid_transforms = sequential_rigid_transforms;
	this -> absolute_rigid_transforms = absolute_rigid_transforms;
	this -> mrps_LN = mrps_LN;
	this -> stdev_Xtilde = stdev_Xtilde;
	this -> stdev_sigmatilde = stdev_sigmatilde;
	// this -> debug_R();

	this -> compute_P_T();

}

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
	arma::vec particle, 
	std::vector<RigidTransform> * args,
	int verbose_level){

	// Particle State ordering:
	// [x,y,z,x_dot,y_dot,z_dot,mu]
	OC::CartState cart_state(particle.subvec(0,5),particle(6));
	OC::KepState kep_state = cart_state.convert_to_kep(0);

	// There are N sequential rigid transforms
	int N = args -> size();
	arma::mat positions(3,N + 1);

	// The epoch time is taken as the time when the last point
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
	arma::vec & state,
	arma::mat & cov,
	const std::map<int, arma::mat::fixed<6,6> > & R_pcs){

	int N_iter = 10;

	arma::mat info_mat(7,7);
	arma::vec normal_mat(7);
	arma::vec residual_vector = arma::vec(3 * this -> sequential_rigid_transforms -> size());

	std::vector<arma::vec::fixed<3> > positions;
	std::vector<arma::mat> stms;

	this -> compute_P_T(R_pcs);
	for (int i = 0; i < N_iter; ++i){

		this -> compute_state_stms(state,positions,stms);
		this -> compute_W(positions);
		this -> build_normal_equations(info_mat,normal_mat,residual_vector,positions,stms);
		
		std::cout << "\tResiduals RMS: " << std::sqrt(arma::dot(residual_vector,residual_vector)/residual_vector.size())<< std::endl;

		try{
			arma::vec deviation = arma::solve(info_mat,normal_mat);

			state += deviation;

		}
		catch(std::runtime_error & e){
			e.what();
		}
		std::cout << "\tState: " << state.t() << std::endl;

	}

	try{
		cov = arma::inv(info_mat);
	}
	catch(std::runtime_error & e){
		e.what();
	}

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
	auto stepper = boost::numeric::odeint::make_controlled<error_stepper_type>( 1.0e-13 ,1.0e-16 );

	std::vector<double> times;

	times.push_back(this -> sequential_rigid_transforms -> at(0).t_start);
	for (int i = 0 ; i < this -> sequential_rigid_transforms -> size(); ++i){
		times.push_back(this -> sequential_rigid_transforms -> at(i).t_end);
	}

	auto tbegin = times.begin();
	auto tend = times.end();
	boost::numeric::odeint::integrate_times(stepper, dynamics, x, tbegin, tend,1e-10,
		Observer::push_back_augmented_state_no_mrp(augmented_state_history));

	for (int i = 0; i < times.size(); ++i){

		arma::mat::fixed<7,7> stm = arma::reshape(augmented_state_history[i].rows(N_est,N_est + N_est * N_est - 1),N_est,N_est);
		positions.push_back(augmented_state_history[i].rows(0,2));
		stms.push_back(stm);
	}

}




void IODFinder::compute_W(const std::vector<arma::vec::fixed<3>> & positions){

	arma::sp_mat dydT = IODFinder::compute_partial_y_partial_T(positions);

	arma::mat R = dydT * this -> P_T * dydT.t();

	this -> W = arma::inv(R);

	std::cout << "DEBUG\n";
	this -> W = arma::eye<arma::mat>(this -> W.n_rows,this -> W.n_cols);



}



void IODFinder::debug_R() const{

	// arma::vec true_particle = { 6.4514e+02,1.9006e+02,3.3690e+02,-2.7270e-02, 4.0806e-03,5.4598e-02,2.25535};

	// std::vector<arma::vec::fixed<3> > positions;
	// std::vector<arma::mat> stms;
	// this -> compute_state_stms(true_particle,positions,stms);



	// int N_rigid_transforms = this -> sequential_rigid_transforms -> size();

	// std::cout << "Number of sequential transforms: " << this -> sequential_rigid_transforms -> size() << std::endl;
	// std::cout << "Number of absolute transforms: " << this -> absolute_rigid_transforms -> size() << std::endl;

	
	// // Nominal sequential transforms
	// arma::vec I_nom(6 * N_rigid_transforms);
	// arma::vec V_nom(12 * N_rigid_transforms + 6);


	// V_nom.subvec(0,2) = this -> absolute_rigid_transforms -> at(0).X;
	// V_nom.subvec(3,5) = RBK::dcm_to_mrp(this -> absolute_rigid_transforms -> at(0).M);

	// for (int k = 1 ; k < N_rigid_transforms + 1; ++ k){


	// 	I_nom.subvec(6 * (k-1), 6 * (k-1) + 2) = this -> sequential_rigid_transforms -> at(k-1).X;
	// 	I_nom.subvec(6 * (k-1) + 3, 6 * (k-1) + 5) = RBK::dcm_to_mrp(this -> sequential_rigid_transforms -> at(k-1).M) ;

	// 	V_nom.subvec(6 + 12 * (k - 1) , 6 + 12 * (k - 1) + 2) = this -> absolute_rigid_transforms -> at(k-1).X;
	// 	V_nom.subvec(6 + 12 * (k - 1) + 3, 6 + 12 * (k - 1) + 5) = RBK::dcm_to_mrp(this -> absolute_rigid_transforms -> at(k-1).M);

	// 	V_nom.subvec(6 + 12 * (k - 1) + 6, 6 + 12 * (k - 1) + 8) = this -> absolute_rigid_transforms -> at(k).X;
	// 	V_nom.subvec(6 + 12 * (k - 1) + 9, 6 + 12 * (k - 1) + 11) = RBK::dcm_to_mrp(this -> absolute_rigid_transforms -> at(k).M);


	// }

	// // Nominal y
	// arma::vec y_nom = arma::zeros<arma::vec>(3 * this -> sequential_rigid_transforms -> size());

	// for (int k = 0; k < this -> sequential_rigid_transforms -> size(); ++ k){

	// 	arma::mat Mkp1 = this -> sequential_rigid_transforms ->  at(k).M;
	// 	arma::vec Xkp1 = this -> sequential_rigid_transforms ->  at(k).X;

	// 	y_nom.rows(3 * k, 3 * k + 2) = IODFinder::compute_y_k(
	// 		positions[k],
	// 		positions[k+1],
	// 		Mkp1,
	// 		Xkp1);
	// }

	// arma::vec dT = arma::zeros<arma::vec>(6 * this -> sequential_rigid_transforms -> size() + 6);
	// arma::vec dI = arma::zeros<arma::vec>(6 * this -> sequential_rigid_transforms -> size());


	// for (int i = 1; i < this -> sequential_rigid_transforms -> size(); ++i){
	// 	dT.subvec(6 * i, 6 * i + 2) =  1 * arma::randn<arma::vec>(3);
	// 	dT.subvec(6 * i + 3, 6 * i + 2 + 3) = 0.001 * arma::randn<arma::vec>(3);
	// }

	

	// // Perturbed sequential transforms
	// std::vector<RigidTransform> rigid_transforms_seq_p;
	// arma::vec I_p(6 * N_rigid_transforms);
	// arma::vec V_p(12 * N_rigid_transforms + 6);
	// arma::vec dV_non_lin = arma::zeros<arma::vec>(12 * N_rigid_transforms + 6);
	// arma::vec dI_non_lin = arma::zeros<arma::vec>(6 * N_rigid_transforms);


	// V_p.subvec(0,2) = this -> absolute_rigid_transforms -> at(0).X;
	// V_p.subvec(3,5) = RBK::dcm_to_mrp(this -> absolute_rigid_transforms -> at(0).M);

	// for (int k = 1 ; k < N_rigid_transforms + 1; ++ k){

	// 	arma::vec X_tilde_km1_p = this -> absolute_rigid_transforms -> at(k-1).X + dT.subvec(6 * (k - 1) , 6 * (k - 1) + 2);
	// 	arma::vec X_tilde_k_p = this -> absolute_rigid_transforms -> at(k).X + dT.subvec(6 * k , 6 * k + 2);

	// 	arma::mat M_tilde_km1_p = this -> absolute_rigid_transforms -> at(k-1).M * RBK::mrp_to_dcm(dT.subvec(6 *(k - 1) + 3, 6 * (k - 1) + 5));
	// 	arma::mat M_tilde_k_p = this -> absolute_rigid_transforms -> at(k).M * RBK::mrp_to_dcm(dT.subvec(6 * k + 3, 6 * k + 5));

	// 	arma::mat M_p_k = RBK::mrp_to_dcm(this -> mrps_LN[k - 1]).t() * M_tilde_km1_p .t() * M_tilde_k_p * RBK::mrp_to_dcm(this -> mrps_LN[k]);
	// 	arma::vec X_p_k = RBK::mrp_to_dcm(this -> mrps_LN[k - 1]).t() * M_tilde_km1_p .t() * (X_tilde_k_p - X_tilde_km1_p);


	// 	X_p_k = X_p_k + dI.subvec(6 * (k - 1) , 6 * (k - 1) + 2 );
	// 	M_p_k = M_p_k * RBK::mrp_to_dcm(dI.subvec(6 * (k - 1) + 3, 6 * (k - 1) + 2 + 3));


	// 	RigidTransform rigid_transform;
	// 	rigid_transform.M = M_p_k;
	// 	rigid_transform.X = X_p_k;
	// 	rigid_transform.t_k = this -> sequential_rigid_transforms -> at(k - 1).t_k;
	// 	rigid_transforms_seq_p.push_back(rigid_transform);

	// 	I_p.subvec(6 * (k-1), 6 * (k-1) + 2) = X_p_k;
	// 	I_p.subvec(6 * (k-1) + 3, 6 * (k-1) + 5) = RBK::dcm_to_mrp(M_p_k) ;

	// 	V_p.subvec(6 + 12 * (k - 1) , 6 + 12 * (k - 1) + 2) = X_tilde_km1_p;
	// 	V_p.subvec(6 + 12 * (k - 1) + 3, 6 + 12 * (k - 1) + 5) = RBK::dcm_to_mrp(M_tilde_km1_p);

	// 	V_p.subvec(6 + 12 * (k - 1) + 6, 6 + 12 * (k - 1) + 8) = X_tilde_k_p;
	// 	V_p.subvec(6 + 12 * (k - 1) + 9, 6 + 12 * (k - 1) + 11) = RBK::dcm_to_mrp(M_tilde_k_p);

	// 	dV_non_lin.subvec(6 + 12 * (k - 1) , 6 + 12 * (k - 1) + 2) = V_p.subvec(6 + 12 * (k - 1) , 6 + 12 * (k - 1) + 2) - V_nom.subvec(6 + 12 * (k - 1) , 6 + 12 * (k - 1) + 2) ;
	// 	dV_non_lin.subvec(6 + 12 * (k - 1) + 3, 6 + 12 * (k - 1) + 5) = RBK::dcm_to_mrp(RBK::mrp_to_dcm(V_nom.subvec(6 + 12 * (k - 1) + 3, 6 + 12 * (k - 1) + 5)).t() * RBK::mrp_to_dcm(V_p.subvec(6 + 12 * (k - 1) + 3, 6 + 12 * (k - 1) + 5)));

	// 	dV_non_lin.subvec(6 + 12 * (k - 1) + 6, 6 + 12 * (k - 1) + 8) = V_p.subvec(6 + 12 * (k - 1) + 6, 6 + 12 * (k - 1) + 8) - V_nom.subvec(6 + 12 * (k - 1) + 6, 6 + 12 * (k - 1) + 8);
	// 	dV_non_lin.subvec(6 + 12 * (k - 1) + 9, 6 + 12 * (k - 1) + 11) = RBK::dcm_to_mrp(RBK::mrp_to_dcm(V_nom.subvec(6 + 12 * (k - 1) + 9, 6 + 12 * (k - 1) + 11)).t() * RBK::mrp_to_dcm(V_p.subvec(6 + 12 * (k - 1) + 9, 6 + 12 * (k - 1) + 11)));

	// 	dI_non_lin.subvec(6 * (k-1), 6 * (k-1) + 2) = I_p.subvec(6 * (k-1), 6 * (k-1) + 2) - I_nom.subvec(6 * (k-1), 6 * (k-1) + 2);
	// 	dI_non_lin.subvec(6 * (k-1) + 3 , 6 * (k-1) + 5) = RBK::dcm_to_mrp(RBK::mrp_to_dcm(I_nom.subvec(6 * (k-1) + 3, 6 * (k-1) + 5)).t() * RBK::mrp_to_dcm( I_p.subvec(6 * (k-1) + 3, 6 * (k-1) + 5)));


	// }	


	// // perturbed y
	// arma::vec y_p = arma::zeros<arma::vec>(3 * rigid_transforms_seq_p. size());

	// for (int k = 0; k < rigid_transforms_seq_p. size(); ++ k){

	// 	arma::mat Mkp1 = rigid_transforms_seq_p.at(k).M;
	// 	arma::vec Xkp1 = rigid_transforms_seq_p.at(k).X;

	// 	y_p.rows(3 * k, 3 * k + 2) = IODFinder::compute_y_k(
	// 		positions[k],
	// 		positions[k+1],
	// 		Mkp1,
	// 		Xkp1);
	// }



	// arma::sp_mat dVdT =  this -> compute_partial_V_partial_T();


	// arma::vec dV_lin = dVdT * dT;

	// std::cout <<  "\tdV_non_lin" << std::endl << std::endl;
	// std::cout << dV_non_lin << std::endl;

	// std::cout <<  "\tdV_lin" << std::endl << std::endl;

	// std::cout << dV_lin << std::endl;
	// std::cout <<  "\t dV_non_lin - dV_lin = \n";
	// std::cout << dV_non_lin - dV_lin << std::endl;

	// std::cout <<  "\t Max dV_non_lin - dV_lin (%) = " << arma::max(arma::abs((dV_non_lin - dV_lin))/dV_non_lin) * 100 << std::endl;

	// arma::sp_mat dIdT = this -> compute_partial_I_partial_V() * dVdT;

	// arma::vec dI_lin = dIdT * dT;

	// std::cout <<  "\tdI_non_lin" << std::endl << std::endl;
	// std::cout << dI_non_lin << std::endl;

	// std::cout <<  "\tdI_lin" << std::endl << std::endl;

	// std::cout << dI_lin << std::endl;

	// std::cout <<  "\t dI_non_lin - dI_lin (%) = \n";
	// std::cout << (dI_non_lin - dI_lin)/dI_lin * 100 << std::endl;

	// std::cout <<  "\t Max dI_non_lin - dI_lin (%) = " << arma::max(arma::abs((dI_non_lin - dI_lin))/dI_lin) * 100 << std::endl;

	// arma::sp_mat dydT = IODFinder::compute_partial_y_partial_T(positions);



	// arma::vec dy_non_lin =  y_p - y_nom;
	// arma::vec dy_lin = dydT * dT;

	// std::cout <<  "\tdy_non_lin" << std::endl << std::endl;
	// std::cout << dy_non_lin << std::endl;

	// std::cout <<  "\tdy_lin" << std::endl << std::endl;

	// std::cout << dy_lin << std::endl;
	// std::cout <<  "\t dy_non_lin - dy_lin \n";
	// std::cout << dy_non_lin - dy_lin << std::endl;

	// std::cout <<  "\t dy_non_lin - dy_lin (%)\n";
	// std::cout << arma::abs(dy_non_lin - dy_lin)/dy_lin << std::endl;

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


void IODFinder::debug_rp_partial() {

	arma::vec kep_elements = {1000,0.1,0.2,0.3,0.5,0.1};
	double mu = 5;

	OC::KepState kep_state_0(kep_elements,mu);

	// State 0
	arma::vec state_0(7);
	state_0.subvec(0,5) = kep_state_0.convert_to_cart(0).get_state();
	state_0(6) = mu;

	arma::vec evec_0 = (arma::cross(state_0.subvec(3,5),arma::cross(state_0.subvec(0,2),state_0.subvec(3,5)))/state_0(6) 
		- arma::normalise(state_0.subvec(0,2)));

	// rp 0
	double rp_0 = kep_state_0.get_a() * ( 1 - kep_state_0.get_eccentricity());


	// dx
	arma::vec dx(7);
	dx.subvec(0,2) = 0.001 * state_0.subvec(0,2);
	dx.subvec(3,5) = 0.001 * state_0.subvec(3,5);
	dx(6) = 0.001 * state_0(6);

	arma::vec state_1 = state_0 + dx;

	// State 1
	OC::CartState cart_state_1(state_1.subvec(0,5),state_1(6));
	OC::KepState kep_state_1 = cart_state_1.convert_to_kep(0);


	arma::vec evec_1 = (arma::cross(state_1.subvec(3,5),arma::cross(state_1.subvec(0,2),state_1.subvec(3,5)))/state_1(6) 
		- arma::normalise(state_1.subvec(0,2)));

	// rp 1
	double rp_1 = kep_state_1.get_a() * ( 1 - kep_state_1.get_eccentricity());

	
	// devec
	arma::vec devec_non_lin = evec_1 - evec_0;
	arma::vec devec_lin = IODFinder::partial_evec_partial_state(state_0)*dx;

	std::cout << "evec: \n";
	std::cout << devec_non_lin.t() << std::endl;
	std::cout << devec_lin.t() << std::endl;

	// de
	double de_non_lin = kep_state_1.get_eccentricity() -  kep_state_0.get_eccentricity() ;
	arma::vec dae_lin = (IODFinder::partial_ae_partial_aevec(kep_state_0.get_a(), 
		evec_0)* IODFinder::partial_aevec_partial_state(kep_state_0.get_a(),state_0) * dx);
	double de_lin = dae_lin(1);

	std::cout << "e: \n";
	std::cout << de_non_lin << std::endl;
	std::cout << de_lin << std::endl;

	// da
	double da_non_lin = kep_state_1.get_a() -  kep_state_0.get_a() ;
	double da_lin = dae_lin(0);

	std::cout << "a: \n";
	std::cout << da_non_lin << std::endl;
	std::cout << da_lin << std::endl;


	// drp
	double drp_non_lin = rp_1 - rp_0;	
	double drp_lin = arma::dot(IODFinder::partial_rp_partial_state(state_0).t(), dx);



	std::cout << "rp: \n";
	std::cout << drp_non_lin << std::endl;
	std::cout << drp_lin << std::endl;
	throw;
















}

