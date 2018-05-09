#ifndef HEADER_IOD_FINDER
#define HEADER_IOD_FINDER

#include <armadillo>
#include <OrbitConversions.hpp>

struct RigidTransform{

	arma::mat::fixed<3,3> M_k;
	arma::vec::fixed<3> X_k;
	double t_k;

};



class IODFinder{


public:

	IODFinder(std::vector<RigidTransform> * rigid_transforms, 
		int N_iter, 
		int particles);

	static double cost_function(arma::vec particle, std::vector<RigidTransform> * args,int verbose_level = 0);

	void run(const arma::vec & lower_bounds,const arma::vec & upper_bounds,int verbose_level = 0);
	OC::KepState get_result() const;

protected:

	int particles;
	int N_iter;
	std::vector<RigidTransform> * rigid_transforms;
	OC::KepState keplerian_state_at_epoch;


};




#endif