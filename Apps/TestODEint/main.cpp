#include <iostream>
#include <vector>

#include <boost/numeric/odeint.hpp>
#include <armadillo>
#include "Args.hpp"
#include "System.hpp"
#include "SystemNew.hpp"
#include "Dynamics.hpp"
#include "Observer.hpp"

// typedef arma::vec::fixed<6> arma::vec;

// 

int main( ){

    Args args;

    SystemNew system(args);



    arma::vec x0 = arma::zeros<arma::vec>(7 + 49);
    arma::vec initial_state = {0.1,0.05,1,1,0.01,0.1,1};
    x0.subvec(0,6) = initial_state;
    x0.subvec(7, 7 + 48) = arma::vectorise(arma::eye<arma::mat>(7,7));


    system.add_next_state("spacecraft_position",3);
    system.add_next_state("spacecraft_velocity",3);
    system.add_next_state("mu",1);



    system.add_dynamics("spacecraft_position",Dynamics::velocity,{"spacecraft_velocity"});
    system.add_dynamics("spacecraft_velocity",Dynamics::point_mass_acceleration,{"spacecraft_position","mu"});

    system.add_jacobian("spacecraft_position","spacecraft_velocity",Dynamics::identity_33,{"spacecraft_velocity"});
    system.add_jacobian("spacecraft_velocity","spacecraft_position",Dynamics::point_mass_gravity_gradient_matrix,{"spacecraft_position","mu"});
    system.add_jacobian("spacecraft_velocity","mu",Dynamics::point_mass_acceleration_unit_mu,{"spacecraft_position"});


    std::vector<arma::vec> states;
    std::vector<double> energy;
    std::vector<double> times;

    double tf = 1;
    unsigned N_times = 10000;

    for (unsigned int i = 0; i < N_times; ++i){

        if (i == 0){
            times.push_back(0);
        }
        else{
            times.push_back( 2 * arma::datum::pi * double(i) / (N_times - 1) * tf  );
        }

    }

    typedef boost::numeric::odeint::runge_kutta_cash_karp54< arma::vec  > error_stepper_type;
    auto stepper = boost::numeric::odeint::make_controlled<error_stepper_type>( 1.0e-10 , 1.0e-16 );

    arma::vec x0_copy(x0);

    boost::numeric::odeint::integrate_times(stepper, system, x0_copy, times.begin(), times.end(), 
        1e-10,Observer::push_back_state(states));

    
    arma::mat orbit(7,states.size());
    std::vector<arma::mat> stms;
    for (int i = 0 ; i < states.size(); ++i) {
        orbit.col(i) = states[i].subvec(0,6);
        stms.push_back(arma::reshape(states[i].subvec(7, 7 + 48),7,7));
    }



    // Perturbed orbit
    arma::vec dx0 = 1e-3 * x0.subvec(0,6);
    arma::vec x0_p(x0);
    x0_p.subvec(0,6) += dx0;

    arma::vec x0_p_copy(x0_p);

    std::vector<arma::vec> states_perturbed; 
    boost::numeric::odeint::integrate_times(stepper, system, x0_p_copy, times.begin(), times.end(), 
        1e-10,Observer::push_back_state(states_perturbed));


    arma::mat orbit_perturbed(7,states_perturbed.size());
    arma::mat linear_perturbation(7,states_perturbed.size());


    for (int i = 0 ; i < states_perturbed.size(); ++i) {
        orbit_perturbed.col(i) = states_perturbed[i].subvec(0,6);
        linear_perturbation.col(i) = stms[i] * dx0;
    }

    orbit.save("orbit.txt",arma::raw_ascii);
    orbit_perturbed.save("orbit_perturbed.txt",arma::raw_ascii);
    linear_perturbation.save("linear_perturbation.txt",arma::raw_ascii);







}
