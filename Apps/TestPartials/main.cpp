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
    args.set_inertia_estimate(arma::eye<arma::mat>(3,3));

    OC::KepState kep_state;
    kep_state.set_state({1e3,0.1,0.1,0.1,0.2,0});
    kep_state.set_mu(2.35);

    arma::vec attitude_state = {1e-1,1e-2,-1e-1,0,0,1e-5};

    arma::vec::fixed<3> dpos, dvel,dmrp, domega;
    double dmu,dC;


    dpos = {1e-2,0,1e-1};
    dvel = {1e-3,0e-3,1e-3};
    dmrp = {1e-3,1e-3,0};
    domega = {1e-6,0,0};

    dmu = 1e-3;
    dC = 1e-3;




    arma::vec initial_state = arma::zeros<arma::vec>(14);
    int number_of_states = initial_state.size();

    initial_state.subvec(0,5) = kep_state.convert_to_cart(0).get_state();
    std::cout << "Unperturbed initial state: \n";

    initial_state.subvec(6,11) = attitude_state;

    initial_state(12) = kep_state.get_mu();
    initial_state(13) = 1.2;
    std::cout << initial_state.subvec(0,number_of_states - 1).t() << std::endl;

    
    arma::vec x0 = arma::zeros<arma::vec>(number_of_states + number_of_states * number_of_states);

    x0.subvec(0,number_of_states - 1) = initial_state;
    x0.subvec(number_of_states, number_of_states + number_of_states * number_of_states - 1) = arma::vectorise(arma::eye<arma::mat>(number_of_states,number_of_states));

    SystemDynamics system(args);

    system.add_next_state("spacecraft_position",3,false);
    system.add_next_state("spacecraft_velocity",3,false);
    system.add_next_state("sigma",3,true);
    system.add_next_state("omega",3,false);

    system.add_next_state("mu",1,false);
    system.add_next_state("C",1,false);

    system.add_dynamics("spacecraft_position",Dynamics::velocity,{"spacecraft_velocity"});
    system.add_dynamics("spacecraft_velocity",Dynamics::point_mass_acceleration,{"spacecraft_position","mu"});
    system.add_dynamics("spacecraft_velocity",Dynamics::SRP_cannonball,{"C"});

    system.add_dynamics("sigma",Dynamics::dmrp_dt,{"sigma","omega"});
    system.add_dynamics("omega",Dynamics::domega_dt_estimate,{"sigma","omega"});

    system.add_jacobian("sigma","sigma",Dynamics::partial_mrp_dot_partial_mrp,{"sigma","omega"});
    system.add_jacobian("sigma","omega",Dynamics::partial_mrp_dot_partial_omega,{"sigma"});
    
    system.add_jacobian("omega","omega",Dynamics::partial_omega_dot_partial_omega_estimate,{"sigma","omega"});

    system.add_jacobian("spacecraft_position","spacecraft_velocity",Dynamics::identity_33,{"spacecraft_velocity"});
    system.add_jacobian("spacecraft_velocity","spacecraft_position",Dynamics::point_mass_gravity_gradient_matrix,{"spacecraft_position","mu"});
    system.add_jacobian("spacecraft_velocity","mu",Dynamics::point_mass_acceleration_unit_mu,{"spacecraft_position"});
    system.add_jacobian("spacecraft_velocity","C",Dynamics::SRP_cannonball_unit_C,{"C"});



    arma::vec state_derivative = arma::zeros<arma::vec>(x0.size());


    system.evaluate_state_derivative(x0 , state_derivative , 0);

    std::cout << "Unperturbed state derivative: \n";
    std::cout << state_derivative.subvec(0,number_of_states - 1 ).t();


    // Perturbed orbit
    arma::vec dx0 = arma::zeros<arma::vec>(initial_state.size());

    dx0.subvec(0,2) = dpos;
    dx0.subvec(3,5) = dvel;
    dx0.subvec(6,8) = dmrp;
    dx0.subvec(9,11) = domega;
    dx0(12) = dmu;
    dx0(13) = dC;



    std::cout << "State perturbation: \n";
    std::cout << dx0.t();

    arma::vec x0_p(x0);
    x0_p.subvec(0,number_of_states - 1) += dx0;

    arma::vec perturbed_state_derivative = arma::zeros<arma::vec>(x0.size());


    system.evaluate_state_derivative(x0_p , perturbed_state_derivative , 0);

    
    std::cout << "Perturbed state derivative: \n";
    std::cout << perturbed_state_derivative.subvec(0,number_of_states - 1 ).t();


    arma::vec state_derivative_difference = (perturbed_state_derivative - state_derivative);

    std::cout << "Difference in state derivative: \n";
    std::cout << state_derivative_difference.subvec(0,number_of_states - 1).t();



    auto A_mat = system.compute_A_matrix(x0 , 0);
    arma::vec state_derivative_difference_linear = (A_mat * dx0);
  
    std::cout << "Difference in state derivative, linear: \n";
    std::cout << state_derivative_difference_linear.t();


    std::cout << "Error (%)\n";
    std::cout << (state_derivative_difference.subvec(0,number_of_states - 1) - state_derivative_difference_linear).t()/state_derivative_difference.subvec(0,number_of_states - 1).t() * 100;



}
