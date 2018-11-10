#include "ShapeModelBezier.hpp"
#include "ShapeModelTri.hpp"
#include "ShapeModelImporter.hpp"
#include "Lidar.hpp"

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
	ShapeModelBezier<ControlPoint> bezier_shape("B", &frame_graph);
	ShapeModelImporter::load_bezier_shape_model("../fit_shape_N_frame.b", 1, false,bezier_shape);

	bezier_shape.construct_kd_tree_shape();

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


	frame_graph . get_frame(lidar. get_ref_frame_name()) -> set_origin_from_parent(pos);
	frame_graph . get_frame(lidar. get_ref_frame_name()) -> set_mrp_from_parent(mrp_LN);
	frame_graph . get_frame("B") -> set_mrp_from_parent(mrp_BN);

	lidar.send_flash(&bezier_shape,false);
	lidar.save("lidar",true);


	return 0;
}