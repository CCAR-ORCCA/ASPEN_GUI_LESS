#include "BatchFilter.hpp"
#include "ExtendedKalmanFilter.hpp"

#include "Wrapper.hpp"

int main(){


	double a = 10000.;
	double tau = std::sqrt(std::pow(a,3)/398600.);
	double earth_mass = 1./arma::datum::G;
	double earth_radius = 6371./a;
	double rotation_rate = 2 * arma::datum::pi / (86400.) * tau;


	arma::vec P0_diag = {0.001,0.001,0.001,0.001,0.001,0.001};
	arma::mat P0 = arma::diagmat(P0_diag);

	arma::vec omega = {0,0,rotation_rate};
	arma::vec station_coords = {48.8566, 2.3522};
	station_coords = station_coords / 180. * arma::datum::pi;

	Args args;
	DynamicAnalyses dyn_an(nullptr);
	args.set_mass(earth_mass);
	args.set_dyn_analyses(&dyn_an);
	args.set_constant_omega(omega);
	args.set_ref_radius(earth_radius);
	args.set_coords_station(station_coords);

	ExtendedKalmanFilter filter(args);
	filter.set_observations_fun(Wrapper::obs_long_lat,
		Wrapper::obs_jac_long_lat);
	filter.set_true_dynamics_fun(Wrapper::point_mass_dxdt_wrapper_odeint);
	filter.set_estimate_dynamics_fun(Wrapper::point_mass_dxdt_wrapper_odeint,
		Wrapper::point_mass_jac_wrapper_odeint);
	filter.set_initial_information_matrix(arma::inv(P0));

	filter.set_gamma_fun(Wrapper::gamma_OD);

	arma::mat Q = std::pow(1e-6 / (a / (tau * tau)),2) * arma::eye<arma::mat>(3,3);


	double N_orbits = 1;
	unsigned N = 1000;
	std::vector<double> times;

	for (unsigned int i = 0; i < N; ++i){

		times.push_back( 2 * arma::datum::pi * double(i) / (N - 1) * N_orbits  );

	}

	arma::vec X0_true = {0,0,1.1,1,0,0.01};
	arma::vec X_bar_0 = {0,0,1.1,1,0,0.01};

	arma::mat R = std::pow(1./3600 * arma::datum::pi / 180,2) * arma::eye<arma::mat>(2,2);

	int iter = filter.run(1,X0_true,X_bar_0,times,R,Q,true);
	
	filter.write_estimated_state("./X_hat.txt");
	filter.write_true_obs("./Y_true.txt");
	filter.write_true_state("./X_true.txt");
	filter.write_T_obs(times,"./T_obs.txt");
	filter.write_residuals("./residuals.txt",R);
	filter.write_estimated_covariance("./covariances.txt");


	return 0;

}