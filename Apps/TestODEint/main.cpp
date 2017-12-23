#include <iostream>
#include <vector>

#include <boost/numeric/odeint.hpp>
#include <armadillo>
#include "Args.hpp"
#include "System.hpp"
#include "Dynamics.hpp"
#include "DynamicAnalyses.hpp"
#include "Observer.hpp"

// typedef arma::vec::fixed<6> arma::vec;

// 

int main( ){

    Args args;
    DynamicAnalyses dyn_an(nullptr);
    args.set_mass(1./arma::datum::G);
    args.set_dyn_analyses(&dyn_an);

    System dynamics(args,6, Dynamics::point_mass_dxdt_odeint);

    arma::vec x0 =  {0,0,1,1,0.,0};

    double R = 10000;
    double tau = std::sqrt(std::pow(R,3)/398600.);

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

    boost::numeric::odeint::integrate_times(stepper, dynamics, x0, times.begin(), times.end(), 
        1e-10,Observer::push_back_state_and_energy(states,energy));

    arma::mat states_mat = arma::zeros<arma::mat> (states.size(),x0.n_rows);
    arma::mat energy_vec = arma::zeros (states.size());

    for (unsigned int i = 0; i < states.size(); ++i){
        states_mat.row(i) = states[i].t();

    }

    states_mat.save("./states.txt",arma::raw_ascii);

    for (unsigned int i = 0; i < states.size(); ++i){
        energy_vec(i) = energy[i];

    }

    states_mat.save("./states.txt",arma::raw_ascii);
    energy_vec.save("./energy.txt",arma::raw_ascii);


}
