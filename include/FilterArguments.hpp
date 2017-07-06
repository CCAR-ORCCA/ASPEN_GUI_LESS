#ifndef HEADER_FILTERARGS
#define HEADER_FILTERARGS


/**
Class storing the filter parameters
	*/
class FilterArguments {
public:

	/**
	Default constructor
	*/
	FilterArguments() {

	}

	/**
	Constructor
	@param t0 Initial time (s)
	@param t1 Final time (s)
	@param orbit_rate Angular rate of the instrument about the target (rad/s), 313
	@param body_spin_rate Angular rate of the instrument about the target (rad/s), M3
	@param min_normal_observation_angle Minimum angle for a ray to be used (rad)
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
	@param recycle_shrunk_facets True if facets that are degenerated should be removed
	*/

	FilterArguments(double t0,
	                double tf,
	                double min_normal_observation_angle,
	                double orbit_rate,
	                double inclination,
	                double body_spin_rate,
	                double min_facet_normal_angle_difference,
	                double ridge_coef,
	                double min_facet_angle,
	                double min_edge_angle,
	                unsigned int minimum_ray_per_facet,
	                unsigned int max_split_count,
	                bool reject_outliers,
	                bool split_status,
	                bool use_cholesky,
	                bool recycle_shrunk_facets) {

		this -> t0 = t0;
		this -> tf = tf;
		this -> min_normal_observation_angle = min_normal_observation_angle;
		this -> orbit_rate = orbit_rate;
		this -> inclination = inclination;
		this -> body_spin_rate = body_spin_rate;
		this -> min_facet_normal_angle_difference = min_facet_normal_angle_difference;
		this -> minimum_ray_per_facet = minimum_ray_per_facet;
		this -> max_split_count = max_split_count;
		this -> ridge_coef = ridge_coef;
		this -> min_facet_angle = min_facet_angle;
		this -> min_edge_angle = min_edge_angle;
		this -> reject_outliers = reject_outliers;
		this -> split_status = split_status;
		this -> use_cholesky = use_cholesky;
		this -> recycle_shrunk_facets = recycle_shrunk_facets;
	}

	double get_t0() const {
		return this -> t0;

	}
	double get_tf() const {
		return this -> tf;

	}
	double get_min_normal_observation_angle() const {
		return this -> min_normal_observation_angle;

	}
	double get_orbit_rate() const {
		return this -> orbit_rate;

	}
	double get_body_spin_rate() const {
		return this -> body_spin_rate;
	}

	double get_inclination() const {
		return this -> inclination;
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
	bool get_reject_outliers() const {
		return this -> reject_outliers;

	}

	bool get_use_cholesky() const {
		return this -> use_cholesky;
	}

	bool get_recycle_shrunk_facets() const {
		return this -> recycle_shrunk_facets;
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

	void set_t0(double t0)  {
		this -> t0 = t0;

	}

	void set_tf(double tf)  {
		this -> tf = tf;

	}

	void set_min_normal_observation_angle(double min_normal_observation_angle) {
		this -> min_normal_observation_angle = min_normal_observation_angle;

	}

	void set_orbit_rate(double orbit_rate )  {
		this -> orbit_rate = orbit_rate;

	}

	void set_body_spin_rate(double body_spin_rate) {
		this -> body_spin_rate = body_spin_rate;
	}

	void set_inclination(double inclination)  {
		this -> inclination = inclination;
	}

	void set_min_facet_normal_angle_difference(double min_facet_normal_angle_difference)  {
		this -> min_facet_normal_angle_difference = min_facet_normal_angle_difference;

	}

	void set_split_status(bool split_status)  {
		this -> split_status = split_status;

	}

	void set_minimum_ray_per_facet(unsigned int minimum_ray_per_facet)  {
		this -> minimum_ray_per_facet = minimum_ray_per_facet;

	}

	void set_ridge_coef(double ridge_coef )  {
		this -> ridge_coef = ridge_coef;

	}

	void set_reject_outliers(bool reject_outliers)  {
		this -> reject_outliers = reject_outliers;

	}

	void set_use_cholesky(bool use_cholesky)  {
		this -> use_cholesky = use_cholesky;
	}

	void set_recycle_shrunk_facets(bool recycle_shrunk_facets)  {
		this -> recycle_shrunk_facets = recycle_shrunk_facets;
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

	void set_cm_bar_0(arma::vec cm_bar) {
		this -> cm_hat_history.clear();
		this -> cm_bar = cm_bar;
		this -> cm_hat_history.push_back(cm_bar);
	}

	void set_omega_bar_0(arma::vec omega_bar) {
		this -> omega_hat_history.clear();
		this -> omega_bar = omega_bar;
		this -> omega_hat_history.push_back(omega_bar);
	}

	/**
	Center of mass
	*/
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


	double get_R_cm() const {
		return this -> R_cm;
	}

	unsigned int get_latest_index_cm() const {
		return this -> cm_hat_history.size();
	}


	void set_R_cm(double R_cm) {
		this -> R_cm = R_cm;
	}


	void set_Q_cm(arma::mat Q_cm) {
		this -> Q_cm = Q_cm;
	}

	arma::mat get_Q_cm() const {
		return this -> Q_cm;
	}


	void append_cm_hat(arma::vec cm) {
		this -> cm_hat_history.push_back(cm);
	}

	void append_P_cm_hat(arma::mat P) {
		this -> P_cm_hat_history.push_back(P);
	}


	std::vector<arma::vec> * get_cm_hat_history(){
		return &this -> cm_hat_history;
	}


	void append_latest_index_cm(unsigned int index) {
		this -> latest_index_history_cm.push_back(index);
	}

	/**
	Omega
	*/

	void set_P_omega_0(arma::mat P_omega_0) {
		this -> P_omega_hat_history.clear();
		this -> P_omega_0 = P_omega_0;
		this -> P_omega_hat_history.push_back(P_omega_0);
	}


	arma::vec get_latest_omega_hat() const {
		return *(--this -> omega_hat_history.end());
	}


	unsigned int get_latest_index_omega() const {
		return this -> omega_hat_history.size();
	}

	
	arma::mat get_latest_P_omega_hat() const {
		return *(--this -> P_omega_hat_history.end()) ;
	}



	void set_R_omega(double R_omega) {
		this -> R_omega = R_omega;
	}


	void set_Q_omega(arma::mat Q_omega) {
		this -> Q_omega = Q_omega;
	}

	arma::mat get_Q_omega() const {
		return this -> Q_omega;
	}
	
	void append_latest_index_omega(unsigned int index) {
		this -> latest_index_history_omega.push_back(index);
	}




	void save_estimate_time_history() const {



		// Estimate
		arma::mat cm_hat_time_history_mat = arma::mat(3, this -> cm_hat_history.size());
		arma::mat P_cm_hat_time_history_mat = arma::mat(9, this -> P_cm_hat_history.size());
		arma::vec latest_index_history_cm_vec = arma::vec(this -> latest_index_history_cm.size());

		for (unsigned int i = 0; i < this -> cm_hat_history.size() ; ++i) {
			cm_hat_time_history_mat.col(i) = this -> cm_hat_history[i];
			P_cm_hat_time_history_mat.col(i) = arma::vectorise(this -> P_cm_hat_history[i]);

		}

		for (unsigned int i = 0; i < this -> latest_index_history_cm.size() ; ++i) {
			latest_index_history_cm_vec(i) = this -> latest_index_history_cm[i];
		}


		cm_hat_time_history_mat.save("cm_time_history_mat.txt", arma::raw_ascii);
		P_cm_hat_time_history_mat.save("P_cm_hat_time_history_mat.txt", arma::raw_ascii);
		latest_index_history_cm_vec.save("latest_index_history_cm_vec.txt", arma::raw_ascii);

	}


protected:

	double t0;
	double tf;
	double min_normal_observation_angle;
	double orbit_rate;
	double inclination;
	double body_spin_rate;
	double min_facet_normal_angle_difference;
	double ridge_coef;
	double min_facet_angle;
	double min_edge_angle;

	unsigned int minimum_ray_per_facet;
	unsigned int max_split_count;

	bool reject_outliers;
	bool split_status;
	bool use_cholesky;
	bool recycle_shrunk_facets;

	arma::vec cm_bar;
	arma::vec omega_bar;
	arma::mat P_cm_0;
	arma::mat P_omega_0;


	double R_cm;
	arma::mat R_omega;
	arma::mat Q_cm;
	arma::mat Q_omega;


	std::vector<arma::vec> cm_hat_history;
	std::vector<arma::vec> omega_hat_history;

	std::vector<arma::mat> P_cm_hat_history;
	std::vector<arma::mat> P_omega_hat_history;

	std::vector<unsigned int> latest_index_history_cm;
	std::vector<unsigned int> latest_index_history_omega;






};

#endif

