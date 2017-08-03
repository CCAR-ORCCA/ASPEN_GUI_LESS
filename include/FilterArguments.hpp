#ifndef HEADER_FILTERARGS
#define HEADER_FILTERARGS


/**
Class storing the filter parameters
	*/
class FilterArguments {
public:



	/**
	Constructor
	@param t0 Initial time (s)
	@param t1 Final time (s)
	@param orbit_rate Angular rate of the instrument about the target (rad/s), 313
	@param body_spin_rate Angular rate of the instrument about the target (rad/s), RBK::M3
	@param max_ray_incidence maximum incidence w/r to surface normal beyond which a ray is discarded (rad)
	@param min_facet_normal_angle_difference Minimum angle separating to normals associated with the same vertex
	@param ridge_coef Non-zero value regularizes the information matrix by introducing a bias
	estimation process
	@param min_facet_angle Minimum angle below which a facet's corner angle indicates facet degeneracy
	@param min_edge_angle Minimum angle separating to facets around an edge for the facets not
	to be flagged as degenerated
	@param minimum_ray_per_facet Minimum number of rays per facet to include the facet in the
	@param max_split_count Maximum time a facet (and its children) can get split

	@param reject_outliers True if facet residuals differing from the mean by more than one sigma should be excluded
	@param split_status True if the shape model is to be split
	@param use_cholesky True is Cholesky decomposition should be used to solve the normal equation
	@param merge_shrunk_facets True if facets that are degenerated should be removed
	*/

	FilterArguments() {

	}


	double get_max_ray_incidence() const {
		return this -> max_ray_incidence;

	}

	double get_min_facet_normal_angle_difference() const {
		return this -> min_facet_normal_angle_difference;

	}

	bool get_split_status() const {
		return this -> split_status;

	}

	unsigned int get_minimum_ray_per_facet() const {
		return this -> minimum_ray_per_facet;

	}
	double get_ridge_coef() const {
		return this -> ridge_coef;

	}
	

	bool get_use_cholesky() const {
		return this -> use_cholesky;
	}

	bool get_merge_shrunk_facets() const {
		return this -> merge_shrunk_facets;
	}
	double get_max_split_count() const {
		return this -> max_split_count;
	}
	double get_min_facet_angle() const {
		return this -> min_facet_angle;
	}
	double get_min_edge_angle() const {
		return this -> min_edge_angle;
	}

	//

	void set_max_ray_incidence(double max_ray_incidence) {
		this -> max_ray_incidence = max_ray_incidence;

	}


	void set_min_facet_normal_angle_difference(double min_facet_normal_angle_difference)  {
		this -> min_facet_normal_angle_difference = min_facet_normal_angle_difference;

	}

	void set_split_facets(bool split_status)  {
		this -> split_status = split_status;

	}

	void set_min_ray_per_facet(unsigned int minimum_ray_per_facet)  {
		this -> minimum_ray_per_facet = minimum_ray_per_facet;

	}

	void set_max_recycled_facets(unsigned int max_recycled_facets) {
		this -> max_recycled_facets = max_recycled_facets;
	}

	void set_convergence_facet_residuals(double convergence_facet_residuals) {
		this -> convergence_facet_residuals = convergence_facet_residuals;
	}

	double get_convergence_facet_residuals() const {
		return this -> convergence_facet_residuals;
	}

	unsigned int get_max_recycled_facets() const {
		return this -> max_recycled_facets;
	}

	void set_N_iterations(unsigned int N_iterations) {
		this -> N_iterations = N_iterations;
	}

	unsigned int get_N_iterations() const {
		return this -> N_iterations;
	}

	void set_ridge_coef(double ridge_coef )  {
		this -> ridge_coef = ridge_coef;

	}


	void set_use_cholesky(bool use_cholesky)  {
		this -> use_cholesky = use_cholesky;
	}

	void set_merge_shrunk_facets(bool merge_shrunk_facets)  {
		this -> merge_shrunk_facets = merge_shrunk_facets;
	}

	void set_max_split_count(double max_split_count)  {
		this -> max_split_count = max_split_count;
	}

	void set_min_facet_angle(double min_facet_angle)  {
		this -> min_facet_angle = min_facet_angle;
	}

	void set_min_edge_angle(double min_edge_angle)  {
		this -> min_edge_angle  = min_edge_angle;
	}

	double get_maximum_J_rms_shape() const {
		return this -> maximum_J_rms_shape;
	}

	void set_maximum_J_rms_shape(double maximum_J_rms) {
		this -> maximum_J_rms_shape = maximum_J_rms;
	}

	/**
	Center of mass
	*/
	void set_cm_bar_0(arma::vec cm_bar) {
		this -> cm_hat_history.clear();
		this -> cm_bar = cm_bar;
		this -> cm_hat_history.push_back(cm_bar);
	}


	arma::vec get_latest_cm_hat() const {
		return *(--this -> cm_hat_history.end());
	}


	void set_P_cm_0(arma::mat P_cm_0) {
		this -> P_cm_hat_history.clear();
		this -> P_cm_0 = P_cm_0;
		this -> P_cm_hat_history.push_back(P_cm_0);
	}


	arma::mat get_latest_P_cm_hat() const {
		return *(--this -> P_cm_hat_history.end()) ;
	}



	void set_Q_cm(arma::mat Q_cm) {
		this -> Q_cm = Q_cm;
	}

	arma::mat get_Q_cm() const {
		return this -> Q_cm;
	}

	void set_estimate_shape(bool estim_shape) {
		this -> estimate_shape = estim_shape;
	}

	bool get_estimate_shape() const {
		return this -> estimate_shape;
	}

	bool get_has_transitioned_to_shape() const {
		return this -> has_transitioned_to_shape;
	}

	void set_has_transitioned_to_shape(bool transition) {
		this -> has_transitioned_to_shape = transition;
	}

	double get_shape_estimation_cm_trigger_thresh() const {
		return this -> shape_estimation_cm_trigger_thresh;
	}

	void set_shape_estimation_cm_trigger_thresh(double threshold) {
		this -> shape_estimation_cm_trigger_thresh = threshold;
	}

	void append_cm_hat(arma::vec cm) {
		this -> cm_hat_history.push_back(cm);
	}

	void append_P_cm_hat(arma::mat P) {
		this -> P_cm_hat_history.push_back(P);
	}


	std::vector<arma::vec> * get_cm_hat_history() {
		return &this -> cm_hat_history;
	}

	/**
	MRP
	*/


	void append_mrp_mes(arma::vec mrp_mes) {
		this -> mrp_mes_history.push_back(mrp_mes);
	}

	void append_mrp_true(arma::vec mrp_true) {
		this -> mrp_true_history.push_back(mrp_true);
	}

	arma::vec get_latest_mrp_mes() const {
		if (this -> mrp_mes_history.size() == 0) {
			return arma::zeros<arma::vec>(3);
		}
		else
			return * (--this ->mrp_mes_history.end());

	}

	/**
	Spin axis
	*/

	void append_spin_axis_mes(arma::vec spin_axis_mes) {
		this -> spin_axis_mes_history.push_back(spin_axis_mes);
	}

	arma::vec get_latest_spin_axis_mes() const {


		return * (--this -> spin_axis_mes_history.end());
	}



	/**
	Omega
	*/

	void set_omega_bar_0(arma::vec omega_bar) {
		this -> omega_hat_history.clear();
		this -> omega_bar = omega_bar;
		this -> omega_hat_history.push_back(omega_bar);
	}

	void set_P_omega_0(arma::mat P_omega_0) {
		this -> P_omega_hat_history.clear();
		this -> P_omega_0 = P_omega_0;
		this -> P_omega_hat_history.push_back(P_omega_0);
	}


	arma::vec get_latest_omega_hat() const {
		return *(--this -> omega_hat_history.end());
	}

	arma::mat get_latest_P_omega_hat() const {
		return *(--this -> P_omega_hat_history.end()) ;
	}


	void append_R_omega(arma::mat R_omega)  {
		this -> R_omega.push_back(R_omega);
	}

	arma::mat get_latest_R_omega() const {
		return *(--this -> R_omega.end());
	}

	void set_Q_omega(arma::mat Q_omega) {
		this -> Q_omega = Q_omega;
	}

	arma::mat get_Q_omega() const {
		return this -> Q_omega;
	}


	void append_omega_true(arma::vec omega) {
		this -> omega_true_history.push_back(omega);
	}


	void append_omega_hat(arma::vec omega) {
		this -> omega_hat_history.push_back(omega);
	}

	void append_omega_mes(arma::vec omega) {
		this -> omega_mes_history.push_back(omega);
	}

	arma::vec get_latest_omega_mes() const {
		return *(--this -> omega_mes_history.end());
	}


	void append_P_omega_hat(arma::mat P) {
		this -> P_omega_hat_history.push_back(P);
	}

	void append_time(double time) {
		this -> time_history.push_back(time);
	}

	double get_latest_time() const {
		return *(--this -> time_history.end()) ;
	}

	unsigned int get_number_of_shape_passes() const {
		return this -> number_of_shape_passes;
	}

	void set_number_of_shape_passe(unsigned int shape_passes)  {
		this -> number_of_shape_passes = shape_passes;
	}



	void save_estimate_time_history() const {

		arma::mat cm_hat_time_history_mat = arma::mat(3, this -> cm_hat_history.size());
		arma::mat P_cm_hat_time_history_mat = arma::mat(9, this -> P_cm_hat_history.size());


		arma::vec time_mat = arma::vec(this -> omega_mes_history.size());
		arma::mat omega_mes_time_history_mat = arma::mat(3, this -> omega_mes_history.size());
		arma::mat omega_true_time_history_mat = arma::mat(3, this -> omega_true_history.size());

		arma::mat mrp_mes_time_history_mat = arma::mat(3, this -> mrp_mes_history.size());
		arma::mat mrp_true_time_history_mat = arma::mat(3, this -> mrp_true_history.size());




		for (unsigned int i = 0; i < this -> cm_hat_history.size() ; ++i) {

			cm_hat_time_history_mat.col(i) = this -> cm_hat_history[i];
			P_cm_hat_time_history_mat.col(i) = arma::vectorise(this -> P_cm_hat_history[i]);

		}


		for (unsigned int i = 0; i < this -> omega_mes_history.size() ; ++i) {

			omega_mes_time_history_mat.col(i) = this -> omega_mes_history[i];
			omega_true_time_history_mat.col(i) = this -> omega_true_history[i];
			time_mat(i) = this -> time_history[i];


		}

		for (unsigned int i = 0; i < this -> mrp_mes_history.size() ; ++i) {
			mrp_mes_time_history_mat.col(i) = this -> mrp_mes_history[i];
		}

		for (unsigned int i = 0; i < this -> mrp_true_history.size() ; ++i) {
			mrp_true_time_history_mat.col(i) = this -> mrp_true_history[i];
		}


		cm_hat_time_history_mat.save("../output/attitude/cm_time_history_mat.txt", arma::raw_ascii);
		P_cm_hat_time_history_mat.save("../output/attitude/P_cm_hat_time_history_mat.txt", arma::raw_ascii);
		omega_mes_time_history_mat.save("../output/attitude/omega_mes_time_history_mat.txt", arma::raw_ascii);
		mrp_mes_time_history_mat.save("../output/attitude/mrp_mes_time_history_mat.txt", arma::raw_ascii);
		mrp_true_time_history_mat.save("../output/attitude/mrp_true_time_history_mat.txt", arma::raw_ascii);


		omega_true_time_history_mat.save("../output/attitude/omega_true_time_history_mat.txt", arma::raw_ascii);
		time_mat.save("../output/attitude/time_history.txt", arma::raw_ascii);



	}


protected:


	double max_ray_incidence;
	double min_facet_normal_angle_difference;
	double ridge_coef;
	double min_facet_angle;
	double min_edge_angle;
	double shape_estimation_cm_trigger_thresh;
	double convergence_facet_residuals;
	double maximum_J_rms_shape = 2;



	unsigned int minimum_ray_per_facet = 1;
	unsigned int max_split_count = 1000;
	unsigned int N_iterations;
	unsigned int max_recycled_facets;
	unsigned int number_of_shape_passes = 5;

	bool split_status;
	bool use_cholesky;
	bool merge_shrunk_facets;
	bool estimate_shape;
	bool has_transitioned_to_shape = false;



	arma::vec cm_bar;
	arma::vec omega_bar;

	arma::mat P_cm_0;
	arma::mat P_omega_0;
	arma::mat Q_cm;
	arma::mat Q_omega;

	std::vector<arma::vec> cm_hat_history;
	std::vector<arma::vec> spin_axis_mes_history;
	std::vector<arma::vec> omega_mes_history;
	std::vector<arma::vec> omega_hat_history;
	std::vector<arma::vec> omega_true_history;
	std::vector<arma::vec> mrp_mes_history;
	std::vector<arma::vec> mrp_true_history;

	std::vector<arma::mat > R_omega;
	std::vector<arma::mat> P_cm_hat_history;
	std::vector<arma::mat> P_omega_hat_history;


	std::vector<double> time_history;








};

#endif

