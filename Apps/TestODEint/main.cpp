#include <iostream>
#include <vector>

#include <boost/numeric/odeint.hpp>
#include <armadillo>
#include "Args.hpp"
#include "SystemDynamics.hpp"
#include "Dynamics.hpp"
#include "Observer.hpp"
#include "OrbitConversions.hpp"
// typedef arma::vec::fixed<6> arma::vec;

// 

int main( ){

    Args args;
    args.set_distance_from_sun_AU(1.);

    SystemDynamics system(args);

    OC::KepState kep_state;
    kep_state.set_state({1e3,0.1,0.1,0.1,0.2,0});
    kep_state.set_mu(2.35);


    arma::vec initial_state(8);
    initial_state.subvec(0,5) = kep_state.convert_to_cart(0).get_state();
    initial_state(6) = kep_state.get_mu();
    initial_state(7) = 1.2;

    int number_of_states = initial_state.size();
    
    arma::vec x0 = arma::zeros<arma::vec>(number_of_states + number_of_states * number_of_states);

    x0.subvec(0,number_of_states - 1) = initial_state;
    x0.subvec(number_of_states, number_of_states + number_of_states * number_of_states - 1) = arma::vectorise(arma::eye<arma::mat>(number_of_states,number_of_states));


    system.add_next_state("spacecraft_position",3,false);
    system.add_next_state("spacecraft_velocity",3,false);
    system.add_next_state("mu",1,false);
    system.add_next_state("C",1,false);


    system.add_dynamics("spacecraft_position",Dynamics::velocity,{"spacecraft_velocity"});
    system.add_dynamics("spacecraft_velocity",Dynamics::point_mass_acceleration,{"spacecraft_position","mu"});
    system.add_dynamics("spacecraft_velocity",Dynamics::SRP_cannonball,{"C"});

    system.add_jacobian("spacecraft_position","spacecraft_velocity",Dynamics::identity_33,{"spacecraft_velocity"});
    system.add_jacobian("spacecraft_velocity","spacecraft_position",Dynamics::point_mass_gravity_gradient_matrix,{"spacecraft_position","mu"});
    system.add_jacobian("spacecraft_velocity","mu",Dynamics::point_mass_acceleration_unit_mu,{"spacecraft_position"});
    system.add_jacobian("spacecraft_velocity","C",Dynamics::SRP_cannonball_unit_C,{"C"});

    std::vector<arma::vec> states;
    std::vector<double> times;

    double tf = 2 * arma::datum::pi / kep_state.get_n();
    unsigned N_times = 10000;

    for (unsigned int i = 0; i < N_times; ++i){

        if (i == 0){
            times.push_back(0);
        }
        else{
            times.push_back(  double(i) / (N_times - 1) * tf  );
        }

    }

    typedef boost::numeric::odeint::runge_kutta_cash_karp54< arma::vec  > error_stepper_type;
    auto stepper = boost::numeric::odeint::make_controlled<error_stepper_type>( 1.0e-10 , 1.0e-16 );

    arma::vec x0_copy(x0);
    std::cout << "integrating\n";


    boost::numeric::odeint::integrate_times(stepper, system, x0_copy, times.begin(), times.end(), 
        1e-10,Observer::push_back_state(states,system.get_number_of_states(),
            system.get_attitude_state_first_indices()));

    std::cout << "done integrating\n";
    
    arma::mat orbit(number_of_states,states.size());
    std::vector<arma::mat> stms;
    for (int i = 0 ; i < states.size(); ++i) {
        orbit.col(i) = states[i].subvec(0,number_of_states - 1);
        stms.push_back(arma::reshape(states[i].subvec(number_of_states,
         number_of_states + number_of_states * number_of_states - 1),
        number_of_states,number_of_states));
    }


    // Perturbed orbit
    arma::vec dx0 = {
        0e-1,
        0e-2,
        0,
        0,
        0e-4,
        0,
        1e-3,
        1e-3
    };
    arma::vec x0_p(x0);
    x0_p.subvec(0,number_of_states - 1) += dx0;

    arma::vec x0_p_copy(x0_p);

    std::vector<arma::vec> states_perturbed; 
    boost::numeric::odeint::integrate_times(stepper, system, x0_p_copy, times.begin(), times.end(), 
        1e-10,Observer::push_back_state(states_perturbed,system.get_number_of_states(),
            system.get_attitude_state_first_indices()));


    arma::mat orbit_perturbed(number_of_states,states_perturbed.size());
    arma::mat linear_perturbation(number_of_states,states_perturbed.size());


    for (int i = 0 ; i < states_perturbed.size(); ++i) {
        orbit_perturbed.col(i) = states_perturbed[i].subvec(0,number_of_states - 1);
        linear_perturbation.col(i) = stms[i] * dx0;
    }

    orbit.save("orbit.txt",arma::raw_ascii);
    orbit_perturbed.save("orbit_perturbed.txt",arma::raw_ascii);
    linear_perturbation.save("linear_perturbation.txt",arma::raw_ascii);







}
