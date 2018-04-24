#include "ShapeBuilder.hpp"

#include "ShapeModelTri.hpp"
#include "ShapeModelBezier.hpp"

#include "Lidar.hpp"
#include "FrameGraph.hpp"
#include "ShapeBuilderArguments.hpp"
#include "PC.hpp"
#include "ICP.hpp"
#include "BundleAdjuster.hpp"
#include "Ray.hpp"
#include "CustomException.hpp"
#include "ControlPoint.hpp"
#include "Facet.hpp"
#include "Element.hpp"
#include "ShapeModelImporter.hpp"
#include "ShapeFitterBezier.hpp"

#include <CGAL_interface.hpp>
#include <RigidBodyKinematics.hpp>

#include <boost/progress.hpp>
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

void ShapeBuilder::run_shape_reconstruction(const arma::vec &times ,
	const std::vector<arma::vec> & X,
	bool save_shape_model) {


	std::cout << "Running the filter" << std::endl;

	arma::vec X_S = arma::zeros<arma::vec>(X[0].n_rows);

	arma::vec lidar_pos = X_S.rows(0,2);
	arma::vec lidar_vel = X_S.rows(3,5);

	arma::mat dcm_LB = arma::eye<arma::mat>(3, 3);
	arma::vec mrp_LN(3);


	arma::mat M_pc = arma::eye<arma::mat>(3,3);
	arma::vec X_pc = arma::zeros<arma::vec>(3);

	for (unsigned int time_index = 0; time_index <= this -> filter_arguments -> get_index_end(); ++time_index) {

		std::stringstream ss;
		ss << std::setw(6) << std::setfill('0') << time_index + 1;
		std::string time_index_formatted = ss.str();

		std::cout << "\n################### Index : " << time_index << " / " <<  this -> filter_arguments -> get_index_end()  << ", Time : " << times(time_index) << " / " <<  times( this -> filter_arguments -> get_index_end()) << " ########################" << std::endl;

		X_S = X[time_index];

		this -> get_new_states(X_S,dcm_LB,mrp_LN,lidar_pos,lidar_vel );

		// Setting the Lidar frame to its new state
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_origin_from_parent(X_S.subvec(0,2));
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_LN);

		// Setting the small body to its new attitude
		this -> frame_graph -> get_frame(this -> true_shape_model -> get_ref_frame_name()) -> set_mrp_from_parent(X_S.subvec(6,8));

		// Getting the true observations (noise is added)
		this -> lidar -> send_flash(this -> true_shape_model,true);

		// The rigid transform best aligning the two point clouds is found
		// The solution to this first registration will be used to prealign the 
		// shape model and the source point cloud

		this -> store_point_clouds(time_index);

		if (this -> destination_pc != nullptr && this -> source_pc == nullptr){
			this -> all_registered_pc.push_back(this -> destination_pc);
		}

		if (this -> destination_pc != nullptr && this -> source_pc != nullptr) {


			// The point-cloud to point-cloud ICP is used for point cloud registration
			ICP icp_pc(this -> destination_pc, this -> source_pc, M_pc, X_pc);

			// These two align the consecutive point clouds 
			// in the instrument frame at t_D == t_0
			M_pc = icp_pc.get_M();
			X_pc = icp_pc.get_X();

			this -> source_pc -> transform(M_pc,X_pc);
			this -> all_registered_pc.push_back(this -> source_pc);

			this -> source_pc -> save("../output/pc/source_" + std::to_string(time_index) + ".obj");

			if (time_index <= this -> filter_arguments -> get_index_init()){
				this -> concatenate_point_clouds(time_index);
			}

			if (time_index == this -> filter_arguments -> get_index_init()|| !this -> filter_arguments -> get_use_icp()){
				std::cout << "- Initializing shape model" << std::endl;
				this -> initialize_shape(time_index);

				if (!this -> filter_arguments -> get_use_icp()){
					return;
				}

			}

			
			if (this -> estimated_shape_model != nullptr ){

				if (time_index != this -> filter_arguments -> get_index_init()){
					ShapeFitterBezier shape_fitter(this -> estimated_shape_model.get(),this -> source_pc.get());

					shape_fitter.fit_shape_batch(this -> filter_arguments -> get_N_iter_shape_filter(),this -> filter_arguments -> get_ridge_coef());
				}

				this -> estimated_shape_model -> save("../output/shape_model/fit_source_" + std::to_string(time_index)+ ".b");


			}

		}

	}









}


std::shared_ptr<ShapeModelBezier> ShapeBuilder::get_estimated_shape_model() const{
	return this -> estimated_shape_model;
}


void ShapeBuilder::concatenate_point_clouds(unsigned int index){

	// The destination point cloud is augmented with the source point cloud
	std::vector< std::shared_ptr<PointNormal> > source_points = this -> source_pc -> get_points();

	int N_source_pc_points = int(this -> filter_arguments -> get_downsampling_factor() * source_points.size());

	// These points have their coordinate in the stitching frame
	// Their normals have been computed and also transformed

	arma::uvec random_order =  arma::regspace< arma::uvec>(0,  source_points.size() - 1);		
	random_order = arma::shuffle(random_order);		

	for (int i = 0; i < N_source_pc_points; ++i){

		this -> concatenated_pc_vector.push_back(source_points[random_order(i)]);

	}

	


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

			
		}

		// Two point clouds have been collected : "nominal case")
		else {

			// The source and destination point clouds are combined into the new source point cloud
			this -> destination_pc = this -> source_pc;

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


	this -> filter_arguments -> append_time(time);
	this -> filter_arguments -> append_omega_true(X_S.rows(9, 11));
	this -> filter_arguments -> append_mrp_true(mrp_BN);

	this -> filter_arguments -> append_relative_pos_mes(arma::zeros<arma::vec>(3));
	this -> filter_arguments -> append_relative_pos_true(arma::zeros<arma::vec>(3));

}



void ShapeBuilder::get_new_states(
	const arma::vec & X_S, 
	arma::mat & dcm_LB, 
	arma::vec & mrp_LN, 
	arma::vec & lidar_pos,
	arma::vec & lidar_vel){

	// Getting the new small body inertial attitude
	// and spacecraft relative position expressed in the small body centered inertia frame
	lidar_pos = X_S.rows(0, 2);
	lidar_vel = X_S.rows(3, 5);

	// The [LB] DCM is assembled. Note that e_r does not exactly have to point towards the target
	// barycenter
	arma::vec e_r = - arma::normalise(lidar_pos);
	arma::vec e_h = arma::normalise(arma::cross(e_r,-lidar_vel));
	arma::vec e_t = arma::cross(e_h,e_r);

	arma::mat dcm_LN(3,3);
	dcm_LN.row(0) = e_r.t();
	dcm_LN.row(1) = e_t.t();
	dcm_LN.row(2) = e_h.t();

	mrp_LN = RBK::dcm_to_mrp(dcm_LN);
	dcm_LB = dcm_LN * RBK::mrp_to_dcm(X_S.rows(3, 8)).t();

}



void ShapeBuilder::initialize_shape(unsigned int time_index){

	// The point clouds are bundle-adjusted
	BundleAdjuster bundle_adjuster(&this -> all_registered_pc);


	throw;	




	std::string pc_path = "../output/pc/source_transformed_poisson.cgal";
	std::string pc_path_obj = "../output/pc/source_transformed_poisson.obj";
	std::string a_priori_path = "../output/shape_model/apriori.obj";
	std::shared_ptr<PC> destination_pc_concatenated;


	std::cout << "-- Constructing point cloud...\n";

	if (this -> filter_arguments -> get_use_icp()){
		
		destination_pc_concatenated = std::make_shared<PC>(PC(this -> concatenated_pc_vector));


		destination_pc_concatenated -> save(
			pc_path, 
			arma::eye<arma::mat>(3,3), 
			arma::zeros<arma::vec>(3), 
			true,
			false);



		destination_pc_concatenated -> save(
			pc_path_obj, 
			arma::eye<arma::mat>(3,3), 
			arma::zeros<arma::vec>(3), 
			false,
			true);
	}

	else{

		arma::mat points,normals;

		this -> true_shape_model -> random_sampling(this -> filter_arguments -> get_surface_samples(),points,normals);

		destination_pc_concatenated = std::make_shared<PC>(PC(points,normals));

		destination_pc_concatenated -> save(
			pc_path, 
			arma::eye<arma::mat>(3,3), 
			arma::zeros<arma::vec>(3), 
			true,
			false);

		destination_pc_concatenated -> save(
			pc_path_obj, 
			arma::eye<arma::mat>(3,3), 
			arma::zeros<arma::vec>(3), 
			false,
			true);


	}


	std::cout << "-- Running PSR...\n";
	CGALINTERFACE::CGAL_interface(
		pc_path,
		a_priori_path,
		this -> filter_arguments -> get_N_edges());


	ShapeModelImporter shape_io_guess(a_priori_path, 1, true);

	ShapeModelTri a_priori_obj("", nullptr);

	shape_io_guess.load_obj_shape_model(&a_priori_obj);

	std::shared_ptr<ShapeModelBezier> a_priori_bezier = std::make_shared<ShapeModelBezier>(ShapeModelBezier(&a_priori_obj,"E", this -> frame_graph));

	// the shape is elevated to the prescribed degree
	unsigned int starting_degree = a_priori_bezier -> get_degree();
	for (unsigned int i = starting_degree; i < this -> filter_arguments -> get_shape_degree(); ++i){
		a_priori_bezier -> elevate_degree();
	}

	a_priori_bezier -> initialize_index_table();
	a_priori_bezier -> save_both("../output/shape_model/a_priori_bezier");

	ShapeFitterBezier shape_fitter(a_priori_bezier.get(),destination_pc_concatenated.get());

	shape_fitter.fit_shape_batch(this -> filter_arguments -> get_N_iter_shape_filter(),this -> filter_arguments -> get_ridge_coef());

	a_priori_bezier -> save_both("../output/shape_model/fit_a_priori");



	// The estimated shape model is finally initialized
	this -> estimated_shape_model = a_priori_bezier;


}





