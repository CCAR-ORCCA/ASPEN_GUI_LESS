#ifndef HEADER_ARGS
#define HEADER_ARGS

#include "FrameGraph.hpp"
#include "ShapeModelTri.hpp"
#include "Interpolator.hpp"
#include "DynamicAnalyses.hpp"
#include "Lidar.hpp"

class Args {

public:

	void set_frame_graph(FrameGraph * frame_graph) {
		this -> frame_graph = frame_graph;
	}

	void set_estimated_shape_model(ShapeModelTri * shape_model) {
		this -> estimated_shape_model = shape_model;
	}

	void set_true_shape_model(ShapeModelTri * shape_model) {
		this -> true_shape_model = shape_model;
	}

	void set_estimated_shape_model(ShapeModel * shape_model) {
		this -> estimated_shape_model = shape_model;
	}

	ShapeModelTri * get_true_shape_model() const {
		return this -> true_shape_model;
	}

	ShapeModel * get_estimated_shape_model() const {
		return this -> estimated_shape_model;
	}

	void set_density(double density) {
		this -> density = density;
	}

	void set_e(double e) {
		this -> eccentricity = e;
	}

	void set_mu(double mu) {
		this -> mu = mu;
	}

	void set_sma(double sma) {
		this -> sma = sma;
	}

	double get_sma() const {
		return this -> sma;
	}

	double get_mu() const {
		return this -> mu;
	}


	void set_Cnm(arma::mat * Cnm){
		this -> Cnm = Cnm;
	}

	void set_Snm(arma::mat * Snm){
		this -> Snm = Snm;
	}

	arma::mat * get_Cnm() const {
		return this -> Cnm;
	}

	arma::mat * get_Snm()const {
		return this -> Snm;
	}

	void set_ref_radius(const double ref_radius){
		this -> ref_radius = ref_radius;
	}

	double get_ref_radius() const{
		return this -> ref_radius;
	}




	double get_density() const {
		return this -> density;
	}

	double get_e() const {
		return this -> eccentricity;
	}

	FrameGraph * get_frame_graph() const{
		return this -> frame_graph;
	}

	

	void set_interpolator(Interpolator * interpolator) {
		this -> interpolator = interpolator;
	}

	void set_stopping_bool(bool stop) {
		this -> stopping_bool = stop;
	}

	void set_mass(double mass) {
		this -> mass = mass;
	}

	double get_mass() const {
		return this -> mass;
	}

	void set_estimated_mass(double estimated_mass) {
		this -> estimated_mass = estimated_mass;
	}

	double get_estimated_mass() const {
		return this -> estimated_mass;
	}

	int get_harmonics_degree() const{
		return this -> harmonics_degree;
	}
	void set_harmonics_degree(int deg){
		this -> harmonics_degree = deg;
	}





	// unsigned int get_degree() const{
	// 	return this -> degree;
	// }

	// void set_degree (unsigned int degree ){
	// 	this -> degree = degree;
	// }

	bool get_stopping_bool() const {
		return this -> stopping_bool;
	}

	Interpolator * get_interpolator() const{
		return this -> interpolator;
	}

	void set_sd_noise(double sd_noise){
		this -> sd_noise = sd_noise;
	}

	double get_sd_noise() const {
		return this -> sd_noise ;
	}

	void set_sd_noise_prop(double sd_noise_prop){
		this -> sd_noise_prop = sd_noise_prop;
	}

	double get_sd_noise_prop() const {
		return this -> sd_noise_prop ;
	}



	void set_time(double time) {
		this -> time = time;
	}

	double get_time() const {
		return this -> time;
	}

	void set_dyn_analyses(DynamicAnalyses * dyn_analyses){
		this -> dyn_analyses = dyn_analyses;
	}

	DynamicAnalyses * get_dyn_analyses() const{
		return this -> dyn_analyses;
	}
	

	void set_minimum_elevation(double el) {
		this -> minimum_elevation = el;
	}

	double get_minimum_elevation() const {
		return this -> minimum_elevation;
	}


	void set_constant_omega(arma::vec omega) {
		this -> constant_omega = omega;
	}

	arma::vec get_constant_omega() const {
		return this -> constant_omega;
	}

	arma::vec get_coords_station() const{
		return this -> coords_station;
	}

	void set_coords_station(arma::vec coords_station){
		this -> coords_station = coords_station;
	}

	void set_lidar(Lidar * lidar){
		this -> lidar = lidar;
	}

	Lidar * get_lidar() const{
		return this -> lidar;
	}

	std::shared_ptr<arma::mat> get_batch_output_covariance_ptr() const{
		return this -> batch_output_covariance_ptr;
	}

	arma::mat get_true_inertia() const{
		return this -> true_inertia;
	}

	void set_true_inertia(arma::mat inertia){
		this -> true_inertia = inertia;
	}


	arma::mat get_estimated_inertia() const{
		return this -> estimated_inertia;
	}

	void set_estimated_inertia(arma::mat inertia){
		this -> estimated_inertia = inertia;
	}

	

	std::vector<double> * get_sigma_consider_vector_ptr() const {
		return this -> sigma_consider_vector_ptr;
	}


	std::vector<double> * get_biases_consider_vector_ptr() const {
		return this -> biases_consider_vector_ptr;
	}

	void set_sigma_consider_vector_ptr(std::vector<double> * ptr)  {
		this -> sigma_consider_vector_ptr = ptr;
	}

	void set_biases_consider_vector_ptr(std::vector<double> * ptr)  {
		this -> biases_consider_vector_ptr = ptr;
	}


	std::vector<double> * get_sigmas_range_vector_ptr() const {
		return this -> sigmas_range_vector_ptr;
	}

	void set_sigmas_range_vector_ptr(std::vector<double> * ptr)  {
		this -> sigmas_range_vector_ptr = ptr;
	}





	arma::vec get_true_pos() const{
		return this -> true_pos;
	}

	arma::vec get_true_vel() const{
		return this -> true_vel;
	}

	arma::vec get_estimated_pos() const{
		return this -> estimated_pos;
	}

	arma::vec get_estimated_vel() const{
		return this -> estimated_vel;
	}




	void set_true_pos(arma::vec x) {
		this -> true_pos = x;
	}

	void set_true_vel(arma::vec x) {
		this -> true_vel = x;
	}

	void set_estimated_pos (arma::vec x){
		this -> estimated_pos = x;
	}

	void set_estimated_vel (arma::vec x){
		this -> estimated_vel = x;
	}


	void set_true_mrp_BN(arma::vec x){
		this -> true_mrp_BN = x;
	}
	void set_estimated_mrp_BN(arma::vec x){
		this -> estimated_mrp_BN = x;
	}

	arma::vec get_true_mrp_BN() const{
		return this -> true_mrp_BN;
	}

	arma::vec get_estimated_mrp_BN() const{
		return this -> estimated_mrp_BN;
	}


protected:

	double density;
	double eccentricity;
	double mu;
	double sma;
	double time;
	double minimum_elevation;
	double mass;
	double estimated_mass;


	double sd_noise;
	double sd_noise_prop = 0;

	// unsigned int degree;
	int harmonics_degree;

	double ref_radius ;

	bool stopping_bool = false;

	FrameGraph * frame_graph;
	ShapeModel * estimated_shape_model;
	ShapeModelTri * true_shape_model;

	DynamicAnalyses * dyn_analyses;
	Interpolator * interpolator;
	Lidar * lidar;

	arma::vec constant_omega;
	arma::vec coords_station;


	arma::mat * Cnm;
	arma::mat * Snm;

	arma::mat true_inertia;
	arma::mat estimated_inertia;


	std::shared_ptr<arma::mat> batch_output_covariance_ptr = std::make_shared<arma::mat>(arma::eye<arma::mat>(3,3));


	arma::vec true_pos;
	arma::vec true_vel;
	arma::vec estimated_pos;
	arma::vec estimated_vel;


	arma::vec true_mrp_BN;
	arma::vec estimated_mrp_BN;



	
	// std::vector<arma::vec> * true_small_body_attitude;
	// std::vector<arma::vec> * estimated_small_body_attitude;


	std::vector<double> * sigma_consider_vector_ptr;
	std::vector<double> * biases_consider_vector_ptr;
	std::vector<double> * sigmas_range_vector_ptr;



};

#endif