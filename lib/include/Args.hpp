#ifndef HEADER_ARGS
#define HEADER_ARGS

#include "FrameGraph.hpp"
#include "Lidar.hpp"
#include <SBGATSphericalHarmo.hpp>
#include <ShapeModelBezier.hpp>


template <class PointType> 
class ShapeModel;

template <class PointType> 
class ShapeModelTri;


template <class PointType> 
class ShapeModelBezier;


class Args {

public:

	void set_frame_graph(FrameGraph * frame_graph) {
		this -> frame_graph = frame_graph;
	}



	void set_estimated_shape_model(ShapeModelBezier<ControlPoint> * shape_model) {
		this -> estimated_shape_model = shape_model;
	}

	void set_true_shape_model(ShapeModelTri<ControlPoint> * shape_model) {
		this -> true_shape_model = shape_model;
	}

	void set_state_covariance(arma::mat P_hat){
		this -> P_hat = P_hat;
	}

	arma::mat get_state_covariance() const{
		return this -> P_hat;
	}


	ShapeModelTri<ControlPoint> * get_true_shape_model() const {
		return this -> true_shape_model;
	}

	ShapeModelBezier<ControlPoint> * get_estimated_shape_model() const {
		return this -> estimated_shape_model;
	}



	void set_density_truth(double density) {
		this -> density_truth = density;
	}

	void set_density_estimate(double density) {
		this -> density_estimate = density;
	}

	void set_e(double e) {
		this -> eccentricity = e;
	}

	
	void set_sma(double sma) {
		this -> sma = sma;
	}

	double get_sma() const {
		return this -> sma;
	}

	double get_mu_estimate() const {
		return this -> mu_estimate;
	}

	double get_mu_truth() const {
		return this -> mu_truth;
	}


	void set_mu_estimate(double mu)  {
		this -> mu_estimate = mu;
	}

	void set_mu_truth(double mu)  {
		this -> mu_truth = mu;
	}


	void set_ref_radius(const double ref_radius){
		this -> ref_radius = ref_radius;
	}

	double get_ref_radius() const{
		return this -> ref_radius;
	}




	double get_density_truth() const {
		return this -> density_truth;
	}

	double get_density_estimate() const {
		return this -> density_estimate;
	}

	double get_e() const {
		return this -> eccentricity;
	}

	FrameGraph * get_frame_graph() const{
		return this -> frame_graph;
	}



	void set_stopping_bool(bool stop) {
		this -> stopping_bool = stop;
	}

	void set_mass_truth(double mass) {
		this -> mass_truth = mass;
	}

	double get_mass_truth() const {
		return this -> mass_truth;
	}

	void set_mass_estimate(double estimated_mass) {
		this -> mass_estimate = estimated_mass;
	}

	double get_mass_estimate() const {
		return this -> mass_estimate;
	}

	int get_harmonics_degree_truth() const{
		return this -> harmonics_degree_truth;
	}
	void set_harmonics_degree_truth(int deg){
		this -> harmonics_degree_truth = deg;
	}

	int get_harmonics_degree_estimate() const{
		return this -> harmonics_degree_estimate;
	}
	void set_harmonics_degree_estimate(int deg){
		this -> harmonics_degree_estimate = deg;
	}



	bool get_stopping_bool() const {
		return this -> stopping_bool;
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

	arma::mat get_inertia_truth() const{
		return this -> inertia_truth;
	}

	void set_inertia_truth(arma::mat inertia){
		this -> inertia_truth = inertia;
	}


	arma::mat get_inertia_estimate() const{
		return this -> inertia_estimate;
	}

	void set_inertia_estimate(arma::mat inertia){
		this -> inertia_estimate = inertia;
	}

	

	

	std::vector<double> * get_sigma_consider_vector_ptr() const {
		return this -> sigma_consider_vector_ptr;
	}

	void set_sigma_consider_vector_ptr(std::vector<double> * ptr)  {
		this -> sigma_consider_vector_ptr = ptr;
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

	void set_use_P_hat_in_batch(bool use_Phat){
		this -> use_P_hat_in_batch = use_Phat;
	}
	bool get_use_P_hat_in_batch() const{
		return this -> use_P_hat_in_batch;
	}

	int get_N_iter_mes_update() const{
		return this -> N_iter_mes_update;
	}

	void set_N_iter_mes_update(int N_iter_mes_update){
		this -> N_iter_mes_update = N_iter_mes_update;
	}

	bool get_use_consistency_test() const{
		return this -> use_consistency_test;
	}

	void set_use_consistency_test(bool use_consistency_test){
		this -> use_consistency_test = use_consistency_test;
	}

	void set_skip_factor(double sf){
		this -> skip_factor = sf;
	}

	double get_skip_factor() const{
		return this -> skip_factor;
	}



	vtkSmartPointer<SBGATSphericalHarmo> get_sbgat_harmonics_truth() const{
		return this -> sbgat_harmonics_truth;
	}


	void set_sbgat_harmonics_truth(vtkSmartPointer<SBGATSphericalHarmo> spherical_harmonics) {
		this -> sbgat_harmonics_truth = spherical_harmonics;
	}


	vtkSmartPointer<SBGATSphericalHarmo> get_sbgat_harmonics_estimate() const{
		return this -> sbgat_harmonics_estimate;
	}


	void set_sbgat_harmonics_estimate(vtkSmartPointer<SBGATSphericalHarmo> spherical_harmonics) {
		this -> sbgat_harmonics_estimate = spherical_harmonics;
	}


	void set_output_dir(std::string dir){
		this -> output_dir = dir;
	}

	std::string get_output_dir() const{
		return this -> output_dir;
	}


	


protected:

	double density_truth;
	double density_estimate;

	double eccentricity;
	double mu_truth;
	double mu_estimate;

	double sma;
	double time;
	double minimum_elevation;
	double mass_truth;
	double mass_estimate;
	double skip_factor;
	double sd_noise;
	double sd_noise_prop = 0;
	double ref_radius ;

	int harmonics_degree_truth;
	int harmonics_degree_estimate;

	int N_iter_mes_update;


	bool stopping_bool = false;
	bool use_P_hat_in_batch = false;
	bool use_consistency_test = false;

	FrameGraph * frame_graph;
	ShapeModelBezier<ControlPoint> * estimated_shape_model;
	ShapeModelTri<ControlPoint> * true_shape_model;
	Lidar * lidar;

	arma::vec constant_omega;
	arma::vec coords_station;

	arma::mat inertia_truth;
	arma::mat inertia_estimate;

	arma::mat P_hat;

	std::shared_ptr<arma::mat> batch_output_covariance_ptr = std::make_shared<arma::mat>(arma::eye<arma::mat>(3,3));


	arma::vec true_pos;
	arma::vec true_vel;
	arma::vec estimated_pos;
	arma::vec estimated_vel;

	arma::vec true_mrp_BN;
	arma::vec estimated_mrp_BN;

	std::vector<double> * sigma_consider_vector_ptr;

	vtkSmartPointer<SBGATSphericalHarmo> sbgat_harmonics_truth;
	vtkSmartPointer<SBGATSphericalHarmo> sbgat_harmonics_estimate;

	std::string output_dir;


};	

#endif