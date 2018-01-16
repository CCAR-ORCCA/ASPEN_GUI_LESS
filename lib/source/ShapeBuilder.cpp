#include "ShapeBuilder.hpp"
#include <chrono>


ShapeBuilder::ShapeBuilder(FrameGraph * frame_graph,
	Lidar * lidar,
	ShapeModelTri * true_shape_model,
	ShapeBuilderArguments * filter_arguments) {

	this -> frame_graph = frame_graph;
	this -> lidar = lidar;
	this -> true_shape_model = true_shape_model;
	this -> filter_arguments = filter_arguments;

}

ShapeBuilder::ShapeBuilder(FrameGraph * frame_graph,
	Lidar * lidar,
	ShapeModelTri * true_shape_model) {

	this -> frame_graph = frame_graph;
	this -> lidar = lidar;
	this -> true_shape_model = true_shape_model;

}



void ShapeBuilder::run_shape_reconstruction(arma::vec &times ,
	const std::vector<arma::vec> & X,
	bool save_shape_model) {


	std::cout << "Running the filter" << std::endl;

	arma::vec X_S = arma::zeros<arma::vec>(X[0].n_rows);

	arma::vec lidar_pos = X_S.rows(0,2);
	arma::vec lidar_vel = X_S.rows(3,5);

	arma::vec e_r;
	arma::vec e_t;
	arma::vec e_h;

	arma::mat dcm_LB = arma::eye<arma::mat>(3, 3);
	arma::mat dcm_LB_t_D = arma::eye<arma::mat>(3, 3);

	arma::vec mrp_LB = {0,0,0};
	arma::vec mrp_BN = X_S.rows(0,2);
	arma::vec mrp_BN_t_D = X_S.rows(0,2);

	arma::mat LN_t_S = arma::eye<arma::mat>(3, 3);
	arma::mat LN_t_D = arma::eye<arma::mat>(3, 3);

	arma::vec volume_dif = arma::vec(times.size());
	arma::vec surface_dif = arma::vec(times.size());
	arma::mat offset_DCM;
	arma::vec OL_t0;
	arma::mat LN_t0;

	arma::mat M_pc = arma::eye<arma::mat>(3,3);
	arma::vec X_pc = arma::zeros<arma::vec>(3);

	for (unsigned int time_index = 0; time_index < times.size(); ++time_index) {

		std::stringstream ss;
		ss << std::setw(6) << std::setfill('0') << time_index + 1;
		std::string time_index_formatted = ss.str();

		std::cout << "\n################### Index : " << time_index << " / " << times.n_rows - 1  << ", Time : " << times(time_index) << " / " <<  times(times.n_rows - 1) << " ########################" << std::endl;

		X_S = X[time_index];

		this -> get_new_relative_states(X_S,dcm_LB,dcm_LB_t_D,LN_t_S, LN_t_D,mrp_BN,mrp_BN_t_D,mrp_LB,lidar_pos,lidar_vel );

		// Setting the Lidar frame to its new state
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_origin_from_parent(lidar_pos);
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_LB);

		// Setting the small body to its inertial attitude. This should not affect the 
		// measurements at all
		this -> frame_graph -> get_frame(this -> true_shape_model -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_BN);

		// Getting the true observations (noise is added)



		this -> lidar -> send_flash(this -> true_shape_model,true);

		// The rigid transform best aligning the two point clouds is found
		// The solution to this first registration will be used to prealign the 
		// shape model and the source point cloud

		this -> store_point_clouds(time_index);


		if (this -> destination_pc != nullptr && this -> source_pc != nullptr) {

			// The point-cloud to point-cloud ICP is used for point cloud registration
			ICP icp_pc(
				this -> destination_pc, 
				this -> source_pc, 
				M_pc, 
				X_pc);

			// These two align the consecutive point clouds 
			// in the instrument frame at t_D
			M_pc = icp_pc.get_M();
			X_pc = icp_pc.get_X();

			// this -> source_pc -> save("../output/pc/source_pc_" + std::to_string(time_index)+ ".obj");
			this -> source_pc -> transform(M_pc,X_pc);
			// this -> source_pc -> save("../output/pc/source_pc_registered_" + std::to_string(time_index)+ ".obj");

			if (time_index <= 100){
				this -> concatenate_point_clouds(time_index);
			}

			if (time_index == 100){
				std::cout << "- Initializing shape model" << std::endl;
				this -> initialize_shape(time_index);
			}

			

			if (this -> estimated_shape_model != nullptr  && this -> destination_pc_concatenated != nullptr){

				ShapeFitterBezier shape_fitter(this -> estimated_shape_model.get(),this -> source_pc.get());

			// shape_fitter.fit_shape_batch(15,1e-5,arma::eye<arma::mat>(3,3), arma::zeros<arma::vec>(3));
				shape_fitter.fit_shape_KF(time_index,
					15,1e-5,arma::eye<arma::mat>(3,3), arma::zeros<arma::vec>(3),
					this -> filter_arguments -> get_los_noise_sd_baseline());

				this -> estimated_shape_model -> save("../output/shape_model/fit_source_" + std::to_string(time_index)+ ".b");

				ShapeModelImporter shape_io_fit("../output/shape_model/fit_source_" + std::to_string(time_index)+ ".b", 1, true);
				ShapeModelBezier fit_source("", this -> frame_graph);
				shape_io_fit.load_bezier_shape_model(&fit_source);
				fit_source.elevate_degree();
				fit_source.elevate_degree();
				fit_source.elevate_degree();
				fit_source.elevate_degree();
				fit_source.elevate_degree();
				fit_source.save_to_obj("../output/shape_model/fit_source_" + std::to_string(time_index)+ ".obj");

			}

		}


	}

}


std::shared_ptr<ShapeModelBezier> ShapeBuilder::get_estimated_shape_model() const{
	return this -> estimated_shape_model;
}


void ShapeBuilder::concatenate_point_clouds(unsigned int index){

	int N_max = 500;


	// The destination point cloud is augmented with the source point cloud
	std::vector< std::shared_ptr<PointNormal> > source_points;
	std::vector< std::shared_ptr<PointNormal> > source_points_downsampled;

	std::vector< std::shared_ptr<PointNormal> > destination_points;

	unsigned int N_destination_pc_points;
	unsigned int N_source_pc_points;


	// These points have their coordinate in the stitching frame
	// Their normals have been computed and also transformed
	source_points = this -> source_pc -> get_points();

	if (source_points.size() < N_max){
		source_points_downsampled = source_points;
	}
	else{

		auto random_order = arma::randi( std::min(N_max,int(source_points.size())),arma::distr_param(0,  source_points.size() - 1));

		for (unsigned int i = 0; i < random_order.n_rows; ++i){
			source_points_downsampled.push_back(source_points[random_order(i)]);
		}


	}


	if(this -> destination_pc_concatenated != nullptr){
		destination_points = this -> destination_pc_concatenated -> get_points();
	}

	N_destination_pc_points = destination_points.size();
	N_source_pc_points = source_points_downsampled.size();

	arma::mat point_coords_all(3,N_destination_pc_points + N_source_pc_points);

	std::vector<std::shared_ptr<PointNormal> > point_normals_all;

	for (unsigned int i = 0; i < N_destination_pc_points; ++ i){
		point_normals_all.push_back(destination_points[i]);
	}

	for (unsigned int i = 0; i < N_source_pc_points; ++ i){
		point_normals_all.push_back(source_points_downsampled[i]);

	}

	this -> destination_pc_concatenated = std::make_shared<PC>(PC(point_normals_all));



}


void ShapeBuilder::measure_spin_axis(const arma::mat & dcm) {

	std::pair<double, arma::vec > prv = RBK::dcm_to_prv(dcm);
	this -> filter_arguments -> append_spin_axis_mes(prv.second);

}





void ShapeBuilder::measure_omega(const arma::mat & dcm) {
	std::pair<double, arma::vec > prv = RBK::dcm_to_prv(dcm);
	this -> filter_arguments -> append_omega_mes(this -> lidar -> get_frequency() * prv.first * this -> filter_arguments -> get_latest_spin_axis_mes());
}


void ShapeBuilder::store_point_clouds(int index,const arma::mat & M_pc,const arma::mat & X_pc) {


	// No point cloud has been collected yet
	if (this -> destination_pc == nullptr) {

		this -> destination_pc = std::make_shared<PC>(PC(this -> lidar -> get_focal_plane()));
		

		
		

	}

	else {

		// Only one source point cloud has been collected
		if (this -> source_pc == nullptr) {


			this -> source_pc = std::make_shared<PC>(PC(this -> lidar -> get_focal_plane()));

			// this -> source_pc -> save(
			// 	"../output/pc/source_pc_" + std::to_string(index) + ".obj");

		}

		// Two point clouds have been collected : "nominal case")
		else {

			// The source and destination point clouds are combined into the new source point cloud
			this -> destination_pc = this -> source_pc;


			// The registered source should be concatenated to the previous destination point
			// cloud here



			this -> source_pc = std::make_shared<PC>(PC(this -> lidar -> get_focal_plane()));

		}
	}
}


void ShapeBuilder::perform_measurements_pc(
	const arma::vec & X_S, 
	double time, 
	const arma::mat & NE_tD_EN_tS_pc,
	const arma::vec & X_relative_from_pc,
	const arma::mat & LN_t_S, 
	const arma::mat & LN_t_D, 
	const arma::vec & mrp_BN,
	const arma::vec & X_relative_true,
	const arma::mat & offset_DCM,
	const arma::vec & OL_t0,
	const arma::mat & LN_t0){

	// Measurements are stored
	arma::vec mrp_mes_pc;

	if (this -> filter_arguments -> get_number_of_measurements() == 0){

		this -> measure_spin_axis(arma::eye<arma::mat>(3,3));
		this -> measure_omega(arma::eye<arma::mat>(3,3));

		this -> filter_arguments -> append_time(time);
		this -> filter_arguments -> append_omega_true(X_S.rows(9, 11));

		this -> filter_arguments -> append_relative_pos_mes(arma::zeros<arma::vec>(3));
		this -> filter_arguments -> append_relative_pos_true(arma::zeros<arma::vec>(3));

		// No need to remove the initial offset. It is added to the mrp measurement 
		this -> filter_arguments -> append_mrp_true(mrp_BN);
		this -> filter_arguments -> append_mrp_mes(mrp_BN);

	}

	else {

		mrp_mes_pc = RBK::dcm_to_mrp(RBK::mrp_to_dcm(this -> filter_arguments -> get_latest_mrp_mes())  * NE_tD_EN_tS_pc );

		this -> filter_arguments -> append_mrp_mes(mrp_mes_pc);

		this -> measure_spin_axis(NE_tD_EN_tS_pc);
		this -> measure_omega(NE_tD_EN_tS_pc);

		this -> filter_arguments -> append_time(time);
		this -> filter_arguments -> append_omega_true(X_S.rows(9, 11));
		this -> filter_arguments -> append_mrp_true(mrp_BN);

		this -> filter_arguments -> append_relative_pos_mes(X_relative_from_pc);
		this -> filter_arguments -> append_relative_pos_true(X_relative_true);

	}


}


void ShapeBuilder::perform_measurements_shape(const arma::vec & X_S, 
	double time, 
	const arma::mat & M,
	const arma::mat & NE_tD_EN_tS_pc,
	const arma::vec & X_pc,
	const arma::mat & LN_t_S, 
	const arma::mat & LN_t_D, 
	const arma::vec & mrp_BN,
	const arma::vec & X_relative_true,
	const arma::mat & offset_DCM,
	const arma::vec & OL_t0,
	const arma::mat & LN_t0){


	this -> filter_arguments -> append_mrp_mes(RBK::dcm_to_mrp( M ));

	this -> measure_spin_axis(NE_tD_EN_tS_pc);
	this -> measure_omega(NE_tD_EN_tS_pc);

	this -> filter_arguments -> append_time(time);
	this -> filter_arguments -> append_omega_true(X_S.rows(9, 11));
	this -> filter_arguments -> append_mrp_true(mrp_BN);

	this -> filter_arguments -> append_relative_pos_mes(arma::zeros<arma::vec>(3));
	this -> filter_arguments -> append_relative_pos_true(arma::zeros<arma::vec>(3));

}



void ShapeBuilder::get_new_relative_states(
	const arma::vec & X_S, 
	arma::mat & dcm_LB, 
	arma::mat & dcm_LB_t_D, 
	arma::mat & LN_t_S, 
	arma::mat & LN_t_D, 
	arma::vec & mrp_BN, 
	arma::vec & mrp_BN_t_D,
	arma::vec & mrp_LB, 
	arma::vec & lidar_pos,
	arma::vec & lidar_vel ){

	// Swapping new and old attitude
	dcm_LB_t_D = dcm_LB;
	mrp_BN_t_D = mrp_BN;

	// Getting the new small body inertial attitude
	// and spacecraft relative position
	mrp_BN = X_S.rows(6,8);
	lidar_pos = X_S.rows(0, 2);
	lidar_vel = X_S.rows(3, 5);

	// The [LB] DCM is assembled. Note that e_r does not exactly have to point towards the target
	// barycenter
	arma::vec e_r = - arma::normalise(lidar_pos);
	arma::vec e_h = arma::normalise(arma::cross(e_r,-lidar_vel));
	arma::vec e_t = arma::cross(e_h,e_r);

	dcm_LB.row(0) = e_r.t();
	dcm_LB.row(1) = e_t.t();
	dcm_LB.row(2) = e_h.t();

	mrp_LB = RBK::dcm_to_mrp(dcm_LB);

	// The [LN] DCM at the present time (t_S) and at the past observation time (t_D) is built
	LN_t_S = dcm_LB * RBK::mrp_to_dcm(mrp_BN);
	LN_t_D = dcm_LB_t_D * RBK::mrp_to_dcm(mrp_BN_t_D);
}



void ShapeBuilder::initialize_shape(unsigned int time_index){

	
	// This is where CGAL should be called
	std::string pc_path = "../output/pc/source_transformed_poisson.cgal";
	std::string pc_path_obj = "../output/pc/source_transformed_poisson.obj";
	std::string a_priori_path = "../output/shape_model/apriori.obj";

	this -> destination_pc_concatenated -> save(
		pc_path, 
		arma::eye<arma::mat>(3,3), 
		arma::zeros<arma::vec>(3), 
		true,
		false);

	this -> destination_pc_concatenated -> save(
		pc_path_obj, 
		arma::eye<arma::mat>(3,3), 
		arma::zeros<arma::vec>(3), 
		true,
		false);

	CGALINTERFACE::CGAL_interface(
		pc_path,
		a_priori_path);

	ShapeModelImporter shape_io_guess(a_priori_path, 1, true);

	ShapeModelTri a_priori_obj("", nullptr);

	shape_io_guess.load_obj_shape_model(&a_priori_obj);
	
	std::shared_ptr<ShapeModelBezier> a_priori_bezier = std::make_shared<ShapeModelBezier>(ShapeModelBezier(&a_priori_obj,"E", this -> frame_graph));

	a_priori_bezier -> elevate_degree();

	a_priori_bezier -> save_to_obj("../output/shape_model/a_priori_bezier.obj");

	ShapeFitterBezier shape_fitter(a_priori_bezier.get(),this -> destination_pc_concatenated.get());

	shape_fitter.fit_shape_KF(time_index,
		15,1e-5,arma::eye<arma::mat>(3,3), arma::zeros<arma::vec>(3),
		this -> filter_arguments -> get_los_noise_sd_baseline());

	a_priori_bezier -> save("../output/shape_model/fit_a_priori.b");

	ShapeModelImporter shape_io_fit("../output/shape_model/fit_a_priori.b", 1, true);
	ShapeModelBezier fit_a_priori("", this -> frame_graph);
	shape_io_fit.load_bezier_shape_model(&fit_a_priori);
	fit_a_priori.elevate_degree();
	fit_a_priori.elevate_degree();
	fit_a_priori.elevate_degree();
	fit_a_priori.elevate_degree();
	fit_a_priori.elevate_degree();
	fit_a_priori.save_to_obj("../output/shape_model/fit_a_priori.obj");

	this -> estimated_shape_model = a_priori_bezier;


}





