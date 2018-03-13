#include "ShapeModelBezier.hpp"
#include "ShapeModelTri.hpp"
#include "ShapeModelImporter.hpp"
#include "Lidar.hpp"
#include "Args.hpp"
// #include "DynamicAnalyses.hpp"

#include <chrono>


// Lidar settings
#define ROW_RESOLUTION 256
#define COL_RESOLUTION 256
#define ROW_FOV 10
#define COL_FOV 10

// Instrument operating frequency
#define INSTRUMENT_FREQUENCY 0.000145 // one flash every 2 hours

// Noise
#define FOCAL_LENGTH 1e-2
#define LOS_NOISE_3SD_BASELINE 0e-2
#define LOS_NOISE_FRACTION_MES_TRUTH 0e-5

// Times (s)
#define T0 0
#define TF 864000// 10 days

int main(){

	// Ref frame graph
	FrameGraph frame_graph;
	frame_graph.add_frame("B");
	frame_graph.add_frame("L");
	frame_graph.add_frame("N");
	frame_graph.add_frame("E");

	frame_graph.add_transform("B", "L");
	frame_graph.add_transform("N", "B");
	frame_graph.add_transform("N", "E");

	// Shape model formed with triangles
	ShapeModelTri true_shape_model("B", &frame_graph);

	ShapeModelImporter shape_io(
		"../itokawa_64.obj", 1, true);

	shape_io.load_obj_shape_model(&true_shape_model);
	true_shape_model.construct_kd_tree_shape();

	ShapeModelBezier estimated_shape_model(&true_shape_model,"E", &frame_graph);
	

	// Integrator extra arguments
	Args args;
	// DynamicAnalyses dyn_analyses(&true_shape_model);
	args.set_frame_graph(&frame_graph);
	args.set_true_shape_model(&true_shape_model);


	// args.set_estimated_shape_model(&estimated_shape_model);

	// args.set_dyn_analyses(&dyn_analyses);
	// args.set_Cnm(&Cnm);
	// args.set_Snm(&Snm);
	// args.set_degree(5);
	// args.set_ref_radius(175);
	// args.set_mu(arma::datum::G * true_shape_model . get_volume() * 1900);
	// args.set_mass(true_shape_model . get_volume() * 1900);
	args.set_sd_noise(LOS_NOISE_3SD_BASELINE / 3);

	double omega = 2 * arma::datum::pi / (12 * 3600);

	// Spacecraft initial state
	// Initial spacecraft state
	arma::vec X0_true_spacecraft = arma::zeros<arma::vec>(6);

	arma::vec pos_0 = {1.5,0,0};
	X0_true_spacecraft.rows(0,2) = pos_0; // r_LN(0) in body frame

	// Velocity determined from sma
	double a = arma::norm(pos_0);
	double v = sqrt(args.get_mu() * (2 / arma::norm(pos_0) - 1./ a));
	arma::vec omega_0 = {0,0,omega};
	arma::vec vel_0_inertial = {0,0,v};
	arma::vec vel_0_body = vel_0_inertial - arma::cross(omega_0,pos_0);
	X0_true_spacecraft.rows(3,5) = vel_0_body; // r'_LN(0) in body frame

	// DEBUG: Asteroid estimated state == spacecraft initial state
	arma::vec X0_true_small_body = {0,0,0,0,0,omega};
	arma::vec X0_estimated_small_body = {0,0,0,0,0,omega};

	// Initial spacecraft position estimate
	arma::vec P0_spacecraft_vec = {0,0,0,0,0,0};
	arma::mat P0_spacecraft_mat = arma::diagmat(P0_spacecraft_vec);

	arma::vec X0_estimated_spacecraft = X0_true_spacecraft + arma::sqrt(P0_spacecraft_mat) * arma::randn<arma::vec>(6);


	// Lidar
	Lidar lidar(&frame_graph,"L",ROW_FOV,COL_FOV ,ROW_RESOLUTION,COL_RESOLUTION,FOCAL_LENGTH,
		INSTRUMENT_FREQUENCY,LOS_NOISE_3SD_BASELINE,LOS_NOISE_FRACTION_MES_TRUTH);


	arma::vec X0_true_augmented = arma::zeros<arma::vec>(12);
	X0_true_augmented.subvec(0,5) = X0_true_spacecraft;
	X0_true_augmented.subvec(6,11) = X0_true_small_body;

	arma::vec X0_estimated_augmented = arma::zeros<arma::vec>(12);
	X0_estimated_augmented.subvec(0,5) = X0_estimated_spacecraft;
	X0_estimated_augmented.subvec(6,11) = X0_estimated_small_body;


	// Instrument orientation
	arma::mat dcm_LB = arma::zeros<arma::mat>(3,3);


	dcm_LB.col(0) = - arma::normalise(X0_true_augmented.rows(0,2));
	dcm_LB.col(2) =  arma::normalise(arma::cross(dcm_LB.col(0),
		X0_true_augmented.rows(3,5)));
	dcm_LB.col(1) = arma::normalise(arma::cross(dcm_LB.col(2),dcm_LB.col(0)));
	arma::inplace_trans(dcm_LB);

	(*args.get_mrp_BN_true()) = X0_true_augmented.rows(6,8);
	(*args.get_mrp_BN_estimated()) = X0_estimated_augmented.subvec(6,8);
	(*args.get_mrp_LN_true()) = RBK::dcm_to_mrp(dcm_LB * RBK::mrp_to_dcm(*args.get_mrp_BN_true()));
	(*args.get_true_pos()) = X0_true_augmented.rows(0,2);

	arma::vec pos = X0_true_augmented.rows(0,2);
	arma::vec mrp_LB = RBK::dcm_to_mrp(dcm_LB);


	frame_graph. get_frame(lidar. get_ref_frame_name()) -> set_origin_from_parent(pos);
	frame_graph. get_frame(lidar. get_ref_frame_name()) -> set_mrp_from_parent(mrp_LB);
	frame_graph. get_frame(args.get_true_shape_model() -> get_ref_frame_name()) -> set_mrp_from_parent(*args.get_mrp_BN_true());
	frame_graph. get_frame("E") -> set_mrp_from_parent(*args.get_mrp_BN_estimated());


	

	return 0;
}