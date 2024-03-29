#ifndef HEADER_FILTERARGS
#define HEADER_FILTERARGS
#include <cassert>

/**
Class storing the filter parameters
	*/
class ShapeBuilderArguments {
public:

	/**
	Constructor
	@param t0 Initial time (s)
	@param t1 Final time (s)
	@param orbit_rate Angular rate of the instrument about the target (rad/s), 313
	@param body_spin_rate Angular rate of the instrument about the target (rad/s), RBK::M3
	@param max_ray_incidence maximum incidence w/r to surface normal beyond which a ray is discarded (rad)
	@param min_normal_angle_difference Minimum angle separating to normals associated with the same vertex
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

	ShapeBuilderArguments() {

	}

	double get_max_ray_incidence() const {
		return this -> max_ray_incidence;

	}

	double get_min_normal_angle_difference() const {
		return this -> min_normal_angle_difference;

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

	void set_use_icp(bool use_icp){
		this -> use_icp = use_icp;
	}

	bool get_use_icp() const{
		return this -> use_icp;
	}

	//

	void set_max_ray_incidence(double max_ray_incidence) {
		this -> max_ray_incidence = max_ray_incidence;

	}


	void set_min_normal_angle_difference(double min_normal_angle_difference)  {
		this -> min_normal_angle_difference = min_normal_angle_difference;

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

	


	// void append_relative_pos_mes(const arma::vec & relative_pos_mes){
	// 	this -> relative_pos_mes_history.push_back(relative_pos_mes);
	// }


	// void append_relative_pos_true(const arma::vec & relative_pos_true){
	// 	this -> relative_pos_true_history.push_back(relative_pos_true);
	// }



	// arma::vec get_latest_relative_pos_mes() const{
	// 	return this -> relative_pos_mes_history.back();
	// }


	// arma::vec get_latest_relative_pos_true() const{
	// 	return this -> relative_pos_true_history.back();
	// }



	void set_index_init(unsigned int init){
		this -> index_init = init;
	}

	unsigned int get_index_init() const{
		return this -> index_init;
	}

	void set_index_end(unsigned int end){
		this -> index_end = end;
	}

	unsigned int get_index_end() const{
		return this -> index_end;
	}
	
	

	void set_estimate_shape(bool estim_shape) {
		this -> estimate_shape = estim_shape;
	}

	void set_N_iter_bundle_adjustment(int n){
		this -> N_iter_bundle_adjustment = n;
	}

	int get_N_iter_bundle_adjustment() const{
		return this -> N_iter_bundle_adjustment;
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

	
	/**
	MRP
	*/


	// void append_mrp_mes(arma::vec mrp_mes) {
	// 	this -> mrp_mes_history.push_back(mrp_mes);
	// }

	// void append_mrp_true(arma::vec mrp_true) {
	// 	this -> mrp_true_history.push_back(mrp_true);
	// }

	// arma::vec get_latest_mrp_mes() const {
	// 	return this -> mrp_mes_history.back();

	// }

	// arma::vec get_latest_mrp_true() const {
	// 	return this ->mrp_true_history.back();

	// }


	void set_los_noise_sd_baseline(double los_noise_sd_baseline){
		this -> los_noise_sd_baseline = los_noise_sd_baseline;
	}
	double get_los_noise_sd_baseline(){
		return this -> los_noise_sd_baseline;
	}


	void set_shape_degree(unsigned int shape_degree){
		this -> shape_degree = shape_degree;
	}
	unsigned int get_shape_degree() const{
		return this -> shape_degree;
	}


	/**
	Spin axis
	*/

	// void append_spin_axis_mes(arma::vec spin_axis_mes) {
	// 	this -> spin_axis_mes_history.push_back(spin_axis_mes);
	// }

	// arma::vec get_latest_spin_axis_mes() const {
	// 	return this -> spin_axis_mes_history.back();
	// }



	/**
	Omega
	*/



	// void append_omega_true(arma::vec omega) {
	// 	this -> omega_true_history.push_back(omega);
	// }


	
	// void append_omega_mes(arma::vec omega) {
	// 	this -> omega_mes_history.push_back(omega);
	// }

	// arma::vec get_latest_omega_mes() const {
	// 	return this -> omega_mes_history.back();
	// }


	// void append_time(double time) {
	// 	this -> time_history.push_back(time);
	// }


	// double get_latest_time() const {
	// 	return this -> time_history.back() ;
	// }

	// unsigned int get_number_of_measurements() const{
	// 	return this -> time_history.size();
	// }


	unsigned int get_N_iter_shape_filter() const {
		return this -> iter_filter;
	}


	void set_N_iter_shape_filter(unsigned int iter)  {
		this -> iter_filter = iter;
	}





	// void save_results() const {

	// 	arma::vec time_mat = arma::vec(this -> omega_mes_history.size());

	// 	arma::mat omega_mes_time_history_mat = arma::mat(3, this -> omega_mes_history.size());
	// 	arma::mat omega_true_time_history_mat = arma::mat(3, this -> omega_true_history.size());

	// 	arma::mat mrp_mes_time_history_mat = arma::mat(3, this -> mrp_mes_history.size());
	// 	arma::mat mrp_true_time_history_mat = arma::mat(3, this -> mrp_true_history.size());

	// 	arma::mat relative_pos_mes_time_history_mat = arma::mat(3, this -> relative_pos_mes_history.size());
	// 	arma::mat relative_pos_true_time_history_mat = arma::mat(3, this -> relative_pos_true_history.size());

	// 	assert(this -> mrp_mes_history.size() == this -> mrp_true_history.size());
	// 	assert(this -> omega_mes_time_history_mat.size() == this -> omega_true_time_history_mat.size());
	// 	assert(this -> relative_pos_mes_time_history_mat.size() == this -> relative_pos_true_time_history_mat.size());
	// 	assert(this -> omega_mes_history.size() == this -> mrp_true_history.size());



	// 	for (unsigned int i = 0; i < this -> omega_mes_history.size() ; ++i) {

	// 		omega_mes_time_history_mat.col(i) = this -> omega_mes_history[i];
	// 		omega_true_time_history_mat.col(i) = this -> omega_true_history[i];
	// 		time_mat(i) = this -> time_history[i];

	// 	}

	// 	for (unsigned int i = 0; i < this -> mrp_mes_history.size() ; ++i) {
	// 		mrp_mes_time_history_mat.col(i) = this -> mrp_mes_history[i];
	// 		mrp_true_time_history_mat.col(i) = this -> mrp_true_history[i];
	// 		relative_pos_true_time_history_mat.col(i) = this -> relative_pos_true_history[i];
	// 		relative_pos_mes_time_history_mat.col(i) = this -> relative_pos_mes_history[i];
	// 	}


	// 	omega_mes_time_history_mat.save("../output/attitude/omega_mes_time_history_mat.txt", arma::raw_ascii);
	// 	mrp_mes_time_history_mat.save("../output/attitude/mrp_mes_time_history_mat.txt", arma::raw_ascii);
	// 	mrp_true_time_history_mat.save("../output/attitude/mrp_true_time_history_mat.txt", arma::raw_ascii);

	// 	relative_pos_mes_time_history_mat.save("../output/attitude/relative_pos_mes_time_history_mat.txt", arma::raw_ascii);
	// 	relative_pos_true_time_history_mat.save("../output/attitude/relative_pos_true_time_history_mat.txt", arma::raw_ascii);

	// 	omega_true_time_history_mat.save("../output/attitude/omega_true_time_history_mat.txt", arma::raw_ascii);
	// 	time_mat.save("../output/attitude/time_history.txt", arma::raw_ascii);
	// 	// index_time_pc_vec.save("../output/attitude/index_time_pc_vec_time_history.txt", arma::raw_ascii);
	// 	// index_time_shape_vec.save("../output/attitude/index_time_shape_vec_time_history.txt", arma::raw_ascii);



	// }


	bool get_use_ba() const{
		return this -> use_ba;
	}
	void set_use_ba(bool use_ba_) {
		this -> use_ba = use_ba_;
	}


	int get_iod_rigid_transforms_number() const{
		return this -> iod_rigid_transforms_number;
	}

	void set_iod_rigid_transforms_number(int n) {
		this -> iod_rigid_transforms_number = n;
	}

	int get_iod_particles() const{
		return this -> iod_particles;
	}

	void set_iod_particles(int n) {
		this -> iod_particles = n;
	}


	int get_iod_iterations() const{
		return this -> iod_iterations;
	}

	void set_iod_iterations(int n) {
		this -> iod_iterations = n;
	}


	void set_mrp_EN_final(arma::vec mrp) {
		this -> mrp_EN_final = mrp;
	}
	
	void set_omega_EN_final(arma::vec omega) {
		this -> omega_EN_final = omega;
	}

	void set_position_final(arma::vec pos) {
		this -> position_final = pos;
	}
	
	void set_velocity_final(arma::vec vel) {
		this -> velocity_final = vel;
	}


	arma::vec get_mrp_EN_final() const{
		return this -> mrp_EN_final;
	}

	arma::vec get_omega_EN_final() const{
		return this -> omega_EN_final;
	}


	arma::vec get_position_final() const{
		return this -> position_final;
	}

	arma::vec get_velocity_final() const{
		return this -> velocity_final;
	}

	double get_rigid_transform_noise_sd(std::string component) const{
		return this -> rigid_transforms_noise_sd.at(component);
	}

	void set_rigid_transform_noise_sd(std::string component,double value){
		this -> rigid_transforms_noise_sd[component] = value;
	}	

	void set_iod_mc_iter(int iter){
		this -> iod_mc_iter = iter;
	}	

	int get_iod_mc_iter() const{
		return this -> iod_mc_iter;
	}

	bool get_use_true_rigid_transforms() const{
		return this -> use_true_rigid_transforms;
	}


	void set_use_true_rigid_transforms(bool use_true_rigid_transforms){
		this -> use_true_rigid_transforms = use_true_rigid_transforms;
	}

	double get_min_triangle_angle() const{
		return this -> min_triangle_angle;
	}
	double get_max_triangle_size() const{
		return this -> max_triangle_size;
	}
	double get_surface_approx_error() const{
		return this -> surface_approx_error;
	}

	int get_number_of_edges() const{
		return this -> number_of_edges;
	}

	void set_min_triangle_angle(double min_triangle_angle) {
		this -> min_triangle_angle = min_triangle_angle;
	}
	void set_max_triangle_size(double max_triangle_size) {
		this -> max_triangle_size = max_triangle_size;
	}
	void set_surface_approx_error(double surface_approx_error) {
		this -> surface_approx_error = surface_approx_error;
	}

	void set_number_of_edges(int number_of_edges) {
		this -> number_of_edges = number_of_edges;
	}

	int get_ba_h() const{
		return this -> ba_h;
	}

	void set_ba_h(int ba_h){
		this -> ba_h = ba_h;
	}

	bool get_use_bezier_shape() const{
		return this -> use_bezier_shape;
	}

	void set_use_bezier_shape(bool use_bezier_shape){
		this -> use_bezier_shape = use_bezier_shape;
	}

	bool get_use_target_poi() const{
		return this -> use_target_poi;
	}

	void set_use_target_poi(bool use_target_poi){
		this -> use_target_poi = use_target_poi;
	}


	void set_save_transformed_source_pc(bool flag){
		this -> save_transformed_source_pc = flag;
	}
	bool get_save_transformed_source_pc() const {
		return this -> save_transformed_source_pc ;
	}


protected:


	double max_ray_incidence;
	double min_normal_angle_difference;
	double ridge_coef = 0;
	double min_facet_angle;
	double min_edge_angle;
	double convergence_facet_residuals;
	double maximum_J_rms_shape = 2;
	double los_noise_sd_baseline;

	double min_triangle_angle;
	double max_triangle_size;
	double surface_approx_error;





	int iod_rigid_transforms_number;
	int iod_iterations;
	int iod_particles;
	int iod_mc_iter;
	int number_of_edges;
	int ba_h = 4;

	unsigned int index_init;
	unsigned int index_end;
	
	unsigned int minimum_ray_per_facet = 1;
	unsigned int max_split_count = 1000;
	unsigned int N_iterations;
	unsigned int max_recycled_facets;
	unsigned int iter_filter ;
	unsigned int shape_degree;
	int N_iter_bundle_adjustment;

	bool split_status;
	bool use_cholesky;
	bool merge_shrunk_facets;
	bool estimate_shape;
	bool has_transitioned_to_shape = false;
	bool use_icp = true;
	bool use_ba = false;
	bool use_true_rigid_transforms = false;
	bool use_bezier_shape = true;
	bool use_target_poi = false;
	bool save_transformed_source_pc = false;


	arma::vec mrp_EN_final;
	arma::vec omega_EN_final;
	arma::vec position_final;
	arma::vec velocity_final;

	std::map<std::string,double> rigid_transforms_noise_sd;











};

#endif

