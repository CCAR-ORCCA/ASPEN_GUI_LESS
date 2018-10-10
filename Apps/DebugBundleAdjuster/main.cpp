#include <ShapeModelBezier.hpp>
#include <ShapeModelTri.hpp>
#include <ShapeModelImporter.hpp>
#include <Lidar.hpp>
#include <PointCloud.hpp>
#include <PointNormal.hpp>
#include <RigidBodyKinematics.hpp>
#include <BundleAdjuster.hpp>
#include <EstimationNormals.hpp>
#include <IterativeClosestPointToPlane.hpp>

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
	frame_graph.add_transform("N", "L");
	frame_graph.add_transform("N", "B");

	// Shape model formed with triangles
	ShapeModelTri<ControlPoint> tri_shape("B", &frame_graph);
	ShapeModelImporter::load_obj_shape_model("../../../resources/shape_models/itokawa_64_scaled_aligned.obj", 1, false,tri_shape);

	tri_shape.construct_kd_tree_shape();


	// Lidar
	Lidar lidar(&frame_graph,
		"L",
		ROW_FOV,
		COL_FOV ,
		ROW_RESOLUTION,
		COL_RESOLUTION,
		FOCAL_LENGTH,
		INSTRUMENT_FREQUENCY,
		LOS_NOISE_3SD_BASELINE,
		LOS_NOISE_FRACTION_MES_TRUTH);


	arma::vec::fixed<3> pos = {3000,0,0};
	arma::vec::fixed<3> mrp_LN = {0,0,-1};
	arma::vec::fixed<3> mrp_BN = {-0.2,0.3,-0.3};


	frame_graph. get_frame(lidar. get_ref_frame_name()) -> set_origin_from_parent(pos);
	frame_graph. get_frame(lidar. get_ref_frame_name()) -> set_mrp_from_parent(mrp_LN);
	frame_graph. get_frame("B") -> set_mrp_from_parent(mrp_BN);

	lidar.send_flash(&tri_shape,false);


	PointCloud<PointNormal> p0(lidar.get_focal_plane());
	p0.build_kdtree();


	EstimationNormals<PointNormal,PointNormal> en(p0,p0);
	arma::vec los_dir = {1,0,0};
	en.set_los_dir(los_dir);
	en.estimate(5);


	PointCloud<PointNormal> p1;
	for (int i = 0; i < p0.size(); ++i){
		p1.push_back(p0[i]);
	}


	arma::vec mrp = {0.0,0.001,0.001};
	arma::vec x = {0,0,10};

	p1.transform(RBK::mrp_to_dcm(mrp),x);
	p1.build_kdtree();

	std::vector<std::shared_ptr<PointCloud<PointNormal> >  > all_registered_pcs;

	all_registered_pcs.push_back(std::make_shared<PointCloud<PointNormal>>(p0));
	all_registered_pcs.push_back(std::make_shared<PointCloud<PointNormal>>(p1));


	std::map<int,arma::mat> M_pcs;
	std::map<int,arma::vec> X_pcs;
	std::vector<arma::mat> BN_measured;
	std::vector<arma::vec> mrps_LN;

	for (int i =0; i < all_registered_pcs.size(); ++i){

		M_pcs[i] = arma::eye<arma::mat>(3,3);
		BN_measured.push_back(arma::eye<arma::mat>(3,3));
		X_pcs[i] = arma::zeros<arma::vec>(3);
		mrps_LN.push_back(arma::zeros<arma::vec>(3));

	}
	auto LN_t0 = arma::eye<arma::mat>(3,3);
	auto x_t0 = arma::zeros<arma::vec>(3);

	int previous_closure_index = 0;


	IterativeClosestPointToPlane icp2p;
	icp2p.set_pc_source(all_registered_pcs.front());
	icp2p.set_pc_destination(all_registered_pcs.back());
	icp2p.set_use_true_pairs(true);
	icp2p.set_maximum_h(0);
	icp2p.set_minimum_h(0);

	icp2p.register_pc();





	BundleAdjuster ba(
		0, 
		1,
		&all_registered_pcs, 
		10,
		5,
		LN_t0,
		x_t0,
		""); 
	
	ba.set_use_true_pairs(true);


	ba.run(
		M_pcs,
		X_pcs,
		BN_measured,
		mrps_LN,
		false,
		previous_closure_index);








	return 0;
}