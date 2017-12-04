#include "Filter.hpp"
#include <chrono>


Filter::Filter(FrameGraph * frame_graph,
	Lidar * lidar,
	ShapeModelTri * true_shape_model,
	ShapeModelTri * estimated_shape_model,
	FilterArguments * filter_arguments) {

	this -> frame_graph = frame_graph;
	this -> lidar = lidar;
	this -> true_shape_model = true_shape_model;
	this -> estimated_shape_model = estimated_shape_model;
	this -> filter_arguments = filter_arguments;
}

Filter::Filter(FrameGraph * frame_graph,
	Lidar * lidar,
	ShapeModelTri * true_shape_model,
	FilterArguments * filter_arguments) {

	this -> frame_graph = frame_graph;
	this -> lidar = lidar;
	this -> true_shape_model = true_shape_model;
	this -> filter_arguments = filter_arguments;

}

Filter::Filter(FrameGraph * frame_graph,
	Lidar * lidar,
	ShapeModelTri * true_shape_model) {

	this -> frame_graph = frame_graph;
	this -> lidar = lidar;
	this -> true_shape_model = true_shape_model;

}



void Filter::run_shape_reconstruction(arma::vec &times ,
	Interpolator * interpolator,
	bool save_shape_model) {


	std::cout << "Running the filter" << std::endl;

	arma::vec X_S = interpolator -> interpolate(times(0), true);

	arma::vec lidar_pos = X_S.rows(6,8);
	arma::vec lidar_vel = X_S.rows(9,11);

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



	for (unsigned int time_index = 0; time_index < times.size(); ++time_index) {

		std::stringstream ss;
		ss << std::setw(6) << std::setfill('0') << time_index + 1;
		std::string time_index_formatted = ss.str();

		std::cout << "\n################### Index : " << time_index << " / " << times.n_rows - 1  << ", Time : " << times(time_index) << " / " <<  times(times.n_rows - 1) << " ########################" << std::endl;

		X_S = interpolator -> interpolate(times(time_index), true);

		this -> get_new_relative_states(X_S,
			dcm_LB,
			dcm_LB_t_D,
			LN_t_S, 
			LN_t_D,
			mrp_BN,
			mrp_BN_t_D,
			mrp_LB,
			lidar_pos,
			lidar_vel );

		std::cout << times(time_index) << std::endl;

		
		// Setting the Lidar frame to its new state
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_origin_from_parent(lidar_pos);
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_LB);

		// Setting the small body to its inertial attitude. This should not affect the 
		// measurements at all
		this -> frame_graph -> get_frame(this -> true_shape_model -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_BN);

		// Getting the true observations (noise is added)
		this -> lidar -> send_flash(this -> true_shape_model);

		
		// The rigid transform best aligning the two point clouds is found
		// The solution to this first registration will be used to prealign the 
		// shape model and the source point cloud

		this -> store_point_clouds(time_index);


		if (this -> destination_pc == nullptr){

			
			// if(this -> source_pc == nullptr && this -> destination_pc != nullptr){
			arma::mat I = arma::eye<arma::mat>(3,3);	

			// This dcm is [LN](t_0)[NB](t_0)
			offset_DCM = LN_t_S * RBK::mrp_to_dcm(mrp_BN).t();

			// This is the position of the spacecraft in the body frame when measurements start to be accumulated
			OL_t0 = X_S.rows(6,8);

			LN_t0 = LN_t_S;

			this -> perform_measurements_pc(X_S, 
				times(time_index),
				I,
				arma::zeros<arma::vec>(3),
				I,
				I, 
				mrp_BN,
				arma::zeros<arma::vec>(3),
				offset_DCM,
				OL_t0,
				LN_t0);

		}
		
		
		if (this -> destination_pc != nullptr && this -> source_pc != nullptr) {

			this -> destination_pc -> save("../output/pc/destination_before.obj");
			this -> source_pc -> save("../output/pc/source_before.obj");


			// The point-cloud to point-cloud ICP is used for point cloud registration
			ICP icp_pc(this -> destination_pc, this -> source_pc, arma::eye<arma::mat>(3,3), arma::zeros<arma::vec>(3));

			// These two align the consecutive point clouds 
			// in the instrument frame at t_D
			arma::mat M_pc = icp_pc.get_M();
			arma::vec X_pc = icp_pc.get_X();


			// Point cloud registration and attitude estimation occurs first
			// this -> store_point_clouds(time_index,M_pc,X_pc);

			this -> source_pc -> save("../output/pc/source_registered_" + std::to_string(time_index) + ".obj",M_pc,X_pc);

			// Attitude is measured. The DCM extracted from the ICP 
			// corresponds to M_pc = [LN](t_D)[NB](t_D)[BN](t_S)[NL](t_S)
			// We want [NB](t_D)[BN](t_S)
			// So we need to get
			// M = [NL](t_D)M_pc[LN](t_S)
			// Now M_pc really measures an incremental rotation of the body frame
			// M = [NB](t_D)[BN](t_S)

			arma::mat NE_tD_EN_tS_pc = LN_t_D.t() * M_pc * LN_t_S;
			arma::mat EN_pc = RBK::mrp_to_dcm(this -> filter_arguments -> get_latest_mrp_mes()) * NE_tD_EN_tS_pc;

			// L_t0_L_tS_pc and rel_pos_L_t0_pc measure source-point-to-shape rotation and translation
			// L_t0_L_tS_pc would be exact if the ICP was error-less

			// As a reminder, 
			// offset_DCM = [LN](t_0)[NB](t_0)

			arma::mat L_t0_L_tS_pc = offset_DCM * EN_pc * LN_t_S .t();

			// The relative position is incremented from the previous one, and the previous measured attitude
			// Note that the current measurement of the attitude (EN_pc and L_t0_L_tS_pc) is not used
			arma::vec X_relative_from_pc = (this -> filter_arguments -> get_latest_relative_pos_mes() 
				+ offset_DCM * RBK::mrp_to_dcm(this -> filter_arguments -> get_latest_mrp_mes()) * LN_t_D .t() * X_pc);

			arma::vec X_relative_true =  offset_DCM * X_S.rows(6,8) - offset_DCM * OL_t0 ;



			this -> source_pc -> save(
				"../output/pc/source_pc_prealigned_" + std::to_string(time_index) + ".obj",
				L_t0_L_tS_pc,X_relative_from_pc);

		// 	// The source point cloud is now registered to the shape
		// 	std::shared_ptr<PC> shape_pc = std::make_shared<PC>(PC(this -> estimated_shape_model));
		// 	ICP icp_shape(shape_pc, this -> source_pc,L_t0_L_tS_pc,X_relative_from_pc);

		// 	arma::mat M = icp_shape.get_M();
		// 	arma::vec X = icp_shape.get_X();

		// 	this -> estimated_shape_model -> save(
		// 		"../output/shape_model/before_fitting_" + std::to_string(time_index) + ".obj");

		// 	// Depending upong which ICP (between the shape-based one and the point-cloud to point-cloud one
		// 	// has the least residuals, the attitude is measured using the best method

		// 	if (icp_pc.get_J_res() < icp_shape.get_J_res()){
		// 		std::cout << "Using point clouds\n";



				// Using the point clouds, measurements of the relative attitude, position are computed
			this -> perform_measurements_pc(X_S, 
				times(time_index),
				NE_tD_EN_tS_pc,
				X_relative_from_pc, 
				LN_t_S, 
				LN_t_D, 
				mrp_BN,
				X_relative_true,
				offset_DCM,
				OL_t0,
				LN_t0);


		// 		this -> source_pc -> save(
		// 			"../output/pc/source_shape_aligned_" + std::to_string(time_index) + ".obj",
		// 			L_t0_L_tS_pc,
		// 			X_relative_from_pc);



		// 		this -> source_pc -> save(
		// 			"../output/pc/source_shape_aligned_true_" + std::to_string(time_index) + ".obj",
		// 			offset_DCM * RBK::mrp_to_dcm(mrp_BN) * LN_t_S.t(),
		// 			X_relative_true);




		}
		// 	else{
		// 		// Using the shape
		// 		std::cout << "Using shape\n";


		// 		arma::mat EN_dcm_shape =  offset_DCM.t() * M * LN_t_S;




		// 		this -> perform_measurements_shape(X_S, 
		// 			times(time_index),
		// 			EN_pc,
		// 			NE_tD_EN_tS_pc, 
		// 			arma::zeros<arma::vec>(3),
		// 			LN_t_S, 
		// 			LN_t_D,
		// 			mrp_BN,
		// 			X_relative_true,
		// 			offset_DCM,
		// 			OL_t0,
		// 			LN_t0);


		// 	// The shape is fitted
		// 		// this -> fit_shape(this -> source_pc.get(),10,0.1,M,X);
		// 		this -> estimated_shape_model -> save("../output/shape_model/fitted_" + std::to_string(time_index) + ".obj");

		// 	}






		// }

		// if (this -> filter_arguments -> get_number_of_measurements() > 0){
		// 	// The attitude of the estimated shape model
		// 	// is set using the latest mrp measurement
		// 	arma::vec mrp_EN = this -> filter_arguments -> get_latest_mrp_mes();

		// 	this -> frame_graph -> get_frame(
		// 		this -> estimated_shape_model -> get_ref_frame_name()) -> set_mrp_from_parent(
		// 		mrp_EN);
		// 	}

	}

}


void Filter::concatenate_point_clouds(const arma::mat & M_pc,const arma::mat & X_pc){


	// The destination point cloud is augmented with the source point cloud

	std::vector< std::shared_ptr<PointNormal> > destination_points = this -> destination_pc -> get_points();
	std::vector< std::shared_ptr<PointNormal> > source_points = this -> source_pc -> get_points();



	arma::mat point_coords(3,destination_points.size() + source_points.size());

	for (unsigned int i = 0; i < destination_points.size(); ++ i){
		point_coords.col(i) = destination_points[i] -> get_point();
	}

	for (unsigned int i = 0; i < source_points.size(); ++ i){
		point_coords.col(i + destination_points.size()) = M_pc * destination_points[i] -> get_point() + X_pc;
	}


	arma::vec u = {1,0,0};
	this -> destination_pc = this -> destination_pc = std::make_shared<PC>(PC(
		u,
		point_coords));


}


void Filter::measure_spin_axis(const arma::mat & dcm) {

	std::pair<double, arma::vec > prv = RBK::dcm_to_prv(dcm);
	this -> filter_arguments -> append_spin_axis_mes(prv.second);

}





void Filter::measure_omega(const arma::mat & dcm) {
	std::pair<double, arma::vec > prv = RBK::dcm_to_prv(dcm);
	this -> filter_arguments -> append_omega_mes(this -> lidar -> get_frequency() * prv.first * this -> filter_arguments -> get_latest_spin_axis_mes());
}


void Filter::store_point_clouds(int index,const arma::mat & M_pc,const arma::mat & X_pc) {


	// No point cloud has been collected yet
	if (this -> destination_pc == nullptr) {

		this -> destination_pc = std::make_shared<PC>(PC(
			this -> lidar -> get_focal_plane(),
			this -> frame_graph));
		std::cout << "Storing first destination point cloud" << std::endl;

		this -> destination_pc -> save(
			"../output/pc/destination_pc_" + std::to_string(index) + ".obj");
	}

	else {

		// Only one source point cloud has been collected
		if (this -> source_pc == nullptr) {


			this -> source_pc = std::make_shared<PC>(PC(
				this -> lidar -> get_focal_plane(),
				this -> frame_graph));
			
			std::cout << "Storing first source point cloud" << std::endl;

			this -> source_pc -> save(
				"../output/pc/source_pc_" + std::to_string(index) + ".obj");

		}

		// Two point clouds have been collected : "nominal case")
		else {

			// The source and destination point clouds are combined into the new source point cloud
			this -> destination_pc = this -> source_pc;


			// The registered source should be concatenated to the previous destination point
			// cloud here


			this -> concatenate_point_clouds(M_pc,X_pc);



			this -> source_pc = std::make_shared<PC>(PC(this -> lidar -> get_focal_plane(),this -> frame_graph));







			this -> destination_pc -> save(
				"../output/pc/destination_pc_" + std::to_string(index) + ".obj");

			this -> source_pc -> save(
				"../output/pc/source_pc_" + std::to_string(index) + ".obj");

		}
	}
}


void Filter::perform_measurements_pc(
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
		this -> filter_arguments -> append_omega_true(X_S.rows(3, 5));

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
		this -> filter_arguments -> append_omega_true(X_S.rows(3, 5));
		this -> filter_arguments -> append_mrp_true(mrp_BN);

		this -> filter_arguments -> append_relative_pos_mes(X_relative_from_pc);
		this -> filter_arguments -> append_relative_pos_true(X_relative_true);

	}


}


void Filter::perform_measurements_shape(const arma::vec & X_S, 
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
	this -> filter_arguments -> append_omega_true(X_S.rows(3, 5));
	this -> filter_arguments -> append_mrp_true(mrp_BN);

	this -> filter_arguments -> append_relative_pos_mes(arma::zeros<arma::vec>(3));
	this -> filter_arguments -> append_relative_pos_true(arma::zeros<arma::vec>(3));

}



void Filter::get_new_relative_states(
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
	mrp_BN = X_S.rows(0,2);
	lidar_pos = X_S.rows(6, 8);
	lidar_vel = X_S.rows(9, 11);

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

