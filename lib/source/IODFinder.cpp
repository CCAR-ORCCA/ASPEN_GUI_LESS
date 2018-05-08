#include "IODFinder.hpp"
#include "Psopt.hpp"



IODFinder::IODFinder(std::vector<RigidTransform> * rigid_transforms, 
	int N_iter, 
	int particles,
	bool pedantic){

	this -> N_iter = N_iter;
	this -> particles = particles;
	this -> rigid_transforms = rigid_transforms;
	this -> pedantic = pedantic;
}



void IODFinder::run(const arma::vec & lower_bounds,const arma::vec & upper_bounds){

	Psopt<std::vector<RigidTransform> *> psopt(IODFinder::cost_function, 
		lower_bounds,
		upper_bounds, 
		this -> particles,
		this -> N_iter,
		this -> rigid_transforms);

	std::cout << "Running IODFinder\n";

	psopt.run(false,this -> pedantic);

	arma::vec elements = psopt.get_result();
	std::cout << "Minimum of cost function : " << IODFinder::cost_function(elements,this-> rigid_transforms) << std::endl;

	this -> keplerian_state_at_epoch = OC::KepState(elements.subvec(0,5),elements(6));


}

OC::KepState IODFinder::get_result() const{
	return this -> keplerian_state_at_epoch;
}

double IODFinder::cost_function(arma::vec particle, std::vector<RigidTransform> * args){

	// Particle State ordering:
	// [a,e,i,Omega,omega,M0_0,mu]

	OC::KepState kep_state(particle.subvec(0,5),particle(6));

	int N =  args -> size();
	arma::mat positions(3,N + 1);


	// Since the dt at which images are captured is constant, the epoch can be inferred
	// by subtracting dt to the first rigid transform's t_k

	double dt = args -> at(1).t_k -  args -> at(0).t_k;
	assert(dt == args -> at(2).t_k -  args -> at(1).t_k);
	double epoch_time = args -> at(1).t_k - dt;
	assert(epoch_time == 0);

	positions.col(0) = kep_state.convert_to_cart(epoch_time).get_position_vector();

	for (int k = 1; k < N + 1; ++k){
		double t_k = args -> at(k - 1).t_k;
		positions.col(k) = kep_state.convert_to_cart(t_k).get_position_vector();
	}

	arma::vec epsilon = arma::zeros<arma::vec>(3 * N);
	for (int k = 0; k < N; ++k ){

		arma::mat M_k = args -> at(k).M_k;
		arma::mat X_k = args -> at(k).X_k;

		epsilon.subvec( 3 * k, 3 * k + 2) = positions.col(k) - M_k * positions.col(k + 1) + X_k;
	}

	return arma::norm(epsilon);

}






