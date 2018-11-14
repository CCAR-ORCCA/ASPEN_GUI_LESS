#ifndef BATCHATTITUDE_HEADER
#define BATCHATTITUDE_HEADER
#include <IODFinder.hpp>

class BatchAttitude{
	

public:


	BatchAttitude(const arma::vec & times, const std::map<int,arma::mat::fixed<3,3> > & M_pcs);


	void run(const std::map<int, arma::mat::fixed<6,6> > & R_pcs,
		const std::vector<arma::vec::fixed<3> > & mrps_LN);

	arma::vec::fixed<6> get_state_estimate_at_epoch() const;
	arma::mat::fixed<6,6> get_state_covariance_at_epoch() const;

	std::vector<arma::vec::fixed<6> > get_attitude_state_history() const;
	std::vector<arma::mat::fixed<6,6> > get_attitude_state_covariances_history() const;


	void set_a_priori_state(const arma::vec::fixed<6> & initial_state);

protected:	


	void compute_state_stms(std::vector<arma::vec::fixed<6> > & state_history,
		std::vector<arma::mat::fixed<6,6> > & stms) const;
	

	void build_normal_equations(
		arma::mat & info_mat,
		arma::vec & normal_mat,
		arma::vec & residual_vector,
		const std::vector<arma::vec::fixed<6> > & state_history,
		const std::vector<arma::mat::fixed<6,6> > & stms,
		const std::vector<arma::vec::fixed<3> > & mrps_LN,
		const std::map<int, arma::mat::fixed<6,6> > & R_pcs) const;



	std::vector<arma::vec::fixed<6> > attitude_state_history;
	std::vector<arma::mat::fixed<6,6> > attitude_state_covariances_history;


	std::vector<RigidTransform> absolute_rigid_transforms;


	arma::vec::fixed<6> state_estimate_at_epoch;
	arma::mat::fixed<6,6> state_covariance_at_epoch;




};

#endif