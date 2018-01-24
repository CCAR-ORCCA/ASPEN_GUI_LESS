#include <Lidar.hpp>
#include <ShapeModelTri.hpp>
#include <ShapeModelImporter.hpp>
#include <ShapeBuilder.hpp>
#include <DynamicAnalyses.hpp>
#include <Dynamics.hpp>
#include <Observer.hpp>
#include <boost/numeric/odeint.hpp>
#include "System.hpp"
#include "Observer.hpp"
#include "PC.hpp"

#include <ShapeFitterTri.hpp>
#include <ShapeFitterBezier.hpp>

#include <limits>
#include <chrono>
#include <boost/progress.hpp>

// Various constants that set up the visibility emulator scenario

// Lidar settings
#define ROW_RESOLUTION 128 // Goldeneye
#define COL_RESOLUTION 128 // Goldeneye
#define ROW_FOV 20 // ?
#define COL_FOV 20 // ?

// Instrument operating frequency
#define INSTRUMENT_FREQUENCY 0.0016

// Noise
#define FOCAL_LENGTH 1e1
#define LOS_NOISE_SD_BASELINE 5e-2 // Goldeneye 1sigma
#define LOS_NOISE_FRACTION_MES_TRUTH 0.

// Times (s)
#define T0 0
#define TF 200000 // 10 days

// Indices
#define INDEX_INIT 200
#define INDEX_END 200

// Downsampling factor (between 0 and 1)
#define DOWNSAMPLING_FACTOR 0.1

// Filter iterations
#define ITER_FILTER 10

// Number of edges in a-priori
#define N_EDGES 1000

// Shape order
#define SHAPE_DEGREE 3


// Target shape
#define TARGET_SHAPE "itokawa_64_scaled_aligned"







///////////////////////////////////////////






int main() {

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

	// Spherical harmonics coefficients
	arma::mat Cnm;
	arma::mat Snm;

#ifdef __APPLE__
	ShapeModelImporter shape_io_truth(
		"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/"+ std::string(TARGET_SHAPE) + ".obj", 1, true);
	Cnm.load("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/gravity/itokawa_150_Cnm_n10_r175.txt", arma::raw_ascii);
	Snm.load("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/gravity/itokawa_150_Snm_n10_r175.txt", arma::raw_ascii);
#elif __linux__
	ShapeModelImporter shape_io_truth(
		"../../../resources/shape_models/" +std::string(TARGET_SHAPE) +".obj", 1 , false);
#else

	throw (std::runtime_error("Neither running on linux or mac os"));
#endif

	shape_io_truth.load_obj_shape_model(&true_shape_model);
	
	true_shape_model.construct_kd_tree_shape();
	DynamicAnalyses dyn_analyses(&true_shape_model);

	// Integrator extra arguments
	Args args;
	args.set_frame_graph(&frame_graph);
	args.set_true_shape_model(&true_shape_model);
	// args.set_dyn_analyses(&dyn_analyses);
	// args.set_Cnm(&Cnm);
	// args.set_Snm(&Snm);
	// args.set_degree(5);
	// args.set_ref_radius(175);
	args.set_mu(arma::datum::G * true_shape_model . get_volume() * 1900);
	args.set_mass(true_shape_model . get_volume() * 1900);


	// Initial state
	arma::vec X0_augmented = arma::zeros<arma::vec>(12);

	double omega = 2 * arma::datum::pi / (12 * 3600);

	arma::vec omega_0 = {0,0,omega};
	X0_augmented.rows(9,11) = omega_0; // Omega_BN(0)

	arma::vec pos_0 = {1000,0,0};
	X0_augmented.rows(0,2) = pos_0; // r_LN(0) in body frame

	// Velocity determined from sma
	double a = arma::norm(pos_0);
	double v = sqrt(args.get_mu() * (2 / arma::norm(pos_0) - 1./ a));


	arma::vec vel_0_inertial = {0,0,v};
	arma::vec vel_0_body = vel_0_inertial - arma::cross(omega_0,pos_0);

	X0_augmented.rows(3,5) = vel_0_body; // r'_LN(0) in body frame

	arma::vec times = arma::regspace<arma::vec>(T0,  1./INSTRUMENT_FREQUENCY,  TF); 
	arma::vec times_dense = arma::regspace<arma::vec>(T0,  10,  TF); 

	std::vector<double> T_obs;
	std::vector<double> T_obs_dense;

	for (unsigned int i = 0; i < times.n_rows; ++i){
		T_obs.push_back(times(i));
	}
	for (unsigned int i = 0; i < times_dense.n_rows; ++i){
		T_obs_dense.push_back(times_dense(i));
	}

	// Containers
	std::vector<arma::vec> X_augmented;
	std::vector<arma::vec> X_augmented_dense;

	auto N_true = X0_augmented.n_rows;

	// Set active inertia here
	args.set_active_inertia(true_shape_model.get_inertia());

	System dynamics(args,
		N_true,
		Dynamics::point_mass_attitude_dxdt_body_frame );

	typedef boost::numeric::odeint::runge_kutta_cash_karp54< arma::vec > error_stepper_type;
	auto stepper = boost::numeric::odeint::make_controlled<error_stepper_type>( 1.0e-10 , 1.0e-16 );

	auto tbegin = T_obs.begin();
	auto tend = T_obs.end();

	boost::numeric::odeint::integrate_times(stepper, dynamics, X0_augmented, tbegin, tend,1e-3,
		Observer::push_back_augmented_state(X_augmented));


	// The orbit is propagated with a finer timestep for visualization purposes

	boost::numeric::odeint::integrate_times(stepper, dynamics, X0_augmented, T_obs_dense.begin(), 
		T_obs_dense.end(),1e-3,
		Observer::push_back_augmented_state(X_augmented_dense));

	arma::mat X_dense(12,X_augmented_dense.size());
	for (unsigned int i = 0; i < X_augmented_dense.size(); ++i){
		X_dense.col(i) = X_augmented_dense[i];
	}
	X_dense.save("../output/trajectory.txt",arma::raw_ascii);
	

// Lidar
	Lidar lidar(&frame_graph,
		"L",
		ROW_FOV,
		COL_FOV ,
		ROW_RESOLUTION,
		COL_RESOLUTION,
		FOCAL_LENGTH,
		INSTRUMENT_FREQUENCY,
		LOS_NOISE_SD_BASELINE,
		LOS_NOISE_FRACTION_MES_TRUTH);


	ShapeBuilderArguments shape_filter_args;
	shape_filter_args.set_los_noise_sd_baseline(LOS_NOISE_SD_BASELINE);
	shape_filter_args.set_index_init(INDEX_INIT);
	shape_filter_args.set_index_end(INDEX_END);
	shape_filter_args.set_downsampling_factor(DOWNSAMPLING_FACTOR);
	shape_filter_args.set_iter_filter(ITER_FILTER);
	shape_filter_args.set_N_edges(N_EDGES);
	shape_filter_args.set_shape_degree(SHAPE_DEGREE);


	ShapeBuilder shape_filter(&frame_graph,
		&lidar,
		&true_shape_model,
		&shape_filter_args);

	shape_filter.run_shape_reconstruction(times,X_augmented,true);


	// The estimated shape model has its barycenter and principal axes lined up with the
	// true shape model
	ShapeModelImporter shape_io_fit_obj(
		"../output/shape_model/fit_source_" +std::to_string(INDEX_END) + ".obj", 1, true);

	
	ShapeModelTri fit_shape("EF", &frame_graph);
	
	shape_io_fit_obj.load_obj_shape_model(&fit_shape);


	// At this stage, the bezier shape model is NOT aligned with the true shape model
	std::shared_ptr<ShapeModelBezier> estimated_shape_model = shape_filter.get_estimated_shape_model();
	
	// This shape model should undergo the same transform as the one imparted to 
	// fit_source_300.obj when it is loaded and aligned with its barycenter/principal axes

	estimated_shape_model -> translate(-fit_shape.get_center_of_mass());
	fit_shape.translate(-fit_shape.get_center_of_mass());
	fit_shape.update_mass_properties();
	arma::mat axes;
	arma::vec moments ;
	fit_shape.get_principal_inertias(axes,moments);
	estimated_shape_model -> rotate(axes.t());
	fit_shape.rotate(axes.t());;

	fit_shape.save("../output/shape_model/fit_shape_aligned.obj");
	

	auto fit_elements = estimated_shape_model -> get_elements();

	std::vector<arma::rowvec> distance_error_vec;

	unsigned int N_samples_per_el = 30;

	for (unsigned int i = 0; i < fit_elements -> size(); ++i){
		
		Bezier * patch = dynamic_cast<Bezier *>(fit_elements -> at(i).get());

		for (unsigned int k = 0; k < N_samples_per_el; ++k){


			arma::vec random = arma::randu(2);

			double u = random(0);
			double v = 1 - random(1) * u;


		// For each Bezier element, the distance to the true shape is
		// found by ray tracing along the element normal evaluated at its center

			auto n = patch -> get_normal(u,v);
			// std::cout << "Normal : " << n.t() << std::endl;

			arma::vec P = patch -> evaluate(u,v);
			// std::cout << "Center : " << P.t() << std::endl;

			Ray ray(P,n);
			Ray ray_rev(P,-n);


			bool hit = true_shape_model.ray_trace(&ray);
			bool hit_rev = true_shape_model.ray_trace(&ray_rev);

			double distance = ray.get_true_range();
			double distance_rev = -ray_rev.get_true_range();

			double distance_final;

			if (hit || hit_rev){
				if (hit && hit_rev){


					if (std::abs(distance) > std::abs(distance_rev)){
						distance_final = distance_rev;
					}
					else{
						distance_final = distance;
					}
				}
				else if (hit){
					distance_final = distance;
				}
				else{
					distance_final = distance_rev;
				}
			}

			else {
				distance_final = std::numeric_limits<double>::infinity();
				std::cout << "Spurious normal cast at:" << P.t();
			}

			
			arma::mat P_CC;
			if (patch -> get_info_mat_ptr() == nullptr){
				P_CC = 1e10 * arma::eye<arma::mat>(3 * patch -> get_control_points() -> size(),
					3 * patch -> get_control_points() -> size());
			}
			else{
				P_CC = arma::inv(*patch -> get_info_mat_ptr());
			}



			arma::mat Pp = patch -> covariance_surface_point(
				u,
				v,
				n,
				P_CC);

			// std::cout << "Patch/range covariance eigenvalue: " << arma::eig_sym(Pp) << std::endl;

			arma::rowvec result = {distance_final,3 * std::sqrt(arma::dot(n,Pp * n)),arma::eig_sym(P_CC)(0),arma::eig_sym(Pp)(0)};

			distance_error_vec.push_back(result);

		}

	}


	arma::mat distance_error = arma::mat(distance_error_vec.size(),4);
	for (unsigned int i =0; i < distance_error_vec.size(); ++i){
		distance_error.row(i) = distance_error_vec[i];
	}

	distance_error.save("../output/results.txt",arma::raw_ascii);


	return 0;
}












