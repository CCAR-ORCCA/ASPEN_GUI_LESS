#ifndef HEADER_ARGS
#define HEADER_ARGS

#include "FrameGraph.hpp"
#include "ShapeModelTri.hpp"
#include "Interpolator.hpp"
#include "DynamicAnalyses.hpp"

class Args {

public:

	void set_frame_graph(FrameGraph * frame_graph) {
		this -> frame_graph = frame_graph;
	}

	void set_shape_model(ShapeModelTri * shape_model) {
		this -> shape_model = shape_model;
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

	arma::mat * get_Cnm(){
		return this -> Cnm;
	}

	arma::mat * get_Snm(){
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

	FrameGraph * get_frame_graph() {
		return this -> frame_graph;
	}

	ShapeModelTri * get_shape_model() {
		return this -> shape_model;
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

	unsigned int get_degree() const{
		return this -> degree;
	}

	void set_degree (unsigned int degree ){
		this -> degree = degree;
	}

	bool get_stopping_bool() {
		return this -> stopping_bool;
	}

	bool get_is_attitude_bool() const {
		return this -> is_attitude_bool;
	}

	void set_is_attitude_bool(bool is_attitude) {
		this -> is_attitude_bool = is_attitude;
	}

	Interpolator * get_interpolator() {
		return this -> interpolator;
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



protected:

	double density;
	double eccentricity;
	double mu;
	double sma;
	double time;
	double minimum_elevation;
	double mass;
	bool stopping_bool = false;
	FrameGraph * frame_graph;
	ShapeModelTri * shape_model;
	DynamicAnalyses * dyn_analyses;
	Interpolator * interpolator;

	arma::vec constant_omega;

	unsigned int degree;
	double ref_radius ;

	arma::mat * Cnm;
	arma::mat * Snm;




	bool is_attitude_bool = false;


};

#endif