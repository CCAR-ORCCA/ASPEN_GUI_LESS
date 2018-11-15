#include "ShapeModelBezier.hpp"
#include "ShapeModelTri.hpp"
#include "ShapeModelImporter.hpp"
#include "Lidar.hpp"
#include <RigidBodyKinematics.hpp>
#include <BatchAttitude.hpp>


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
	ShapeModelBezier<ControlPoint> estimated_shape_model("B", &frame_graph);
	ShapeModelImporter::load_bezier_shape_model("../fit_shape_N_frame.b", 1, false,estimated_shape_model);

	estimated_shape_model.construct_kd_tree_shape();



	estimated_shape_model.update_mass_properties();
	int N_times = 100;

	double T_L = 1./0.0001;
	double T_B = 12 * 3600;
	double dt = 1000;

	arma::vec times(N_times);
	std::map<int,arma::mat::fixed<3,3> > M_pcs;
	std::vector<arma::vec::fixed<3>> mrps_LN;
	std::vector<arma::mat::fixed<3,3>> BN_measured;
	std::map<int, arma::mat::fixed<6,6> > R_pcs;


	mrps_LN.push_back(arma::zeros<arma::vec>(3));
	times(0) = 0;
	BN_measured.push_back(arma::eye<arma::mat>(3,3));
	M_pcs[0] = arma::eye<arma::mat>(3,3);
	for (int i = 1; i < 100; ++i){
		double t = i * dt;
		times(i) = t;
		M_pcs[i] = RBK::M3(t/T_B * 2 * arma::datum::pi);
		mrps_LN.push_back(RBK::dcm_to_mrp(RBK::M3(t/T_L * 2 * arma::datum::pi)));
		BN_measured.push_back(RBK::M3(t/T_B * 2 * arma::datum::pi));
		R_pcs[i] = arma::eye<arma::mat>(6,6);
	}


	arma::vec::fixed<6> a_priori_state;
	a_priori_state.subvec(0,2) = RBK::dcm_to_mrp(BN_measured.front());
	a_priori_state.subvec(3,5) = 4 * arma::inv(RBK::Bmat(RBK::dcm_to_mrp(BN_measured.front()))) * (RBK::dcm_to_mrp(BN_measured[1]) - RBK::dcm_to_mrp(BN_measured.front()))/(times(1) - times(0));

	std::cout << "A-priori state: " << a_priori_state.t() << std::endl;


	std::cout << "Running the batch\n";
	BatchAttitude batch_attitude(times,M_pcs);
	batch_attitude.set_a_priori_state(a_priori_state);
	batch_attitude.set_inertia_estimate(estimated_shape_model.get_inertia());
	batch_attitude.run(R_pcs,mrps_LN);


	return 0;
}