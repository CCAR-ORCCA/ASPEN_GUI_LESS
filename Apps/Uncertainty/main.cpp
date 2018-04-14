#include <Bezier.hpp>
#include <ControlPoint.hpp>
#include <Ray.hpp>
#include <PC.hpp>
#include <ShapeFitterBezier.hpp>
#include <ShapeModelBezier.hpp>
#include <chrono>

#include "boost/progress.hpp"




int main(){

	arma::arma_rng::set_seed(0);

	std::vector<std::shared_ptr<ControlPoint> > vertices;

	std::shared_ptr<ControlPoint> v0 = std::make_shared<ControlPoint>(ControlPoint());
	arma::vec nominal_coords0 = {0,0,0};
	v0 -> set_coordinates(nominal_coords0);
	vertices.push_back(v0);

	std::shared_ptr<ControlPoint> v1 = std::make_shared<ControlPoint>(ControlPoint());
	arma::vec nominal_coords1 = {1,0,0.5};
	v1 -> set_coordinates(nominal_coords1);
	vertices.push_back(v1);

	std::shared_ptr<ControlPoint> v2 = std::make_shared<ControlPoint>(ControlPoint());
	arma::vec nominal_coords2 = {0.0,1,0.3};
	v2 -> set_coordinates(nominal_coords2);
	vertices.push_back(v2);

	std::shared_ptr<ControlPoint> v3 = std::make_shared<ControlPoint>(ControlPoint());
	arma::vec nominal_coords3 = {2,0,0};
	v3 -> set_coordinates(nominal_coords3);
	vertices.push_back(v3);


	std::shared_ptr<ControlPoint> v4 = std::make_shared<ControlPoint>(ControlPoint());
	arma::vec nominal_coords4 = {1.5,0.5,1};
	v4 -> set_coordinates(nominal_coords4);
	vertices.push_back(v4);

	std::shared_ptr<ControlPoint> v5 = std::make_shared<ControlPoint>(ControlPoint());
	arma::vec nominal_coords5 = {0.5,2,0};
	v5 -> set_coordinates(nominal_coords5);
	vertices.push_back(v5);
	
	Bezier nominal_patch(vertices);
	ShapeModelBezier dummy(nominal_patch);
	
	dummy.save_both("nominal");

	// A 18 by 18 covariance is created
	arma::mat P_X(18,18);

	for (int i = 0; i < 6; ++i){
		arma::vec rand_vec = arma::randu<arma::vec>(1);
		double rand_n = 0.01 * rand_vec(0) + 0.001;
		P_X.submat(3 * i, 3 * i, 3 * i + 2, 3 * i + 2) = rand_n * arma::eye<arma::mat>(3,3);
	}
	
	// Removing correlations
	// arma::mat P_X_no_correl = arma::diagmat(P_X.diag());
	arma::mat P_X_no_correl = P_X;


	P_X.save("P_X.txt",arma::raw_ascii);

	arma::mat Z =  arma::chol( P_X, "lower" ) ;
	arma::vec E(18);

	// The measurement direction is created
	int N_rays = 50000;
	arma::vec ranges(N_rays);
	arma::vec pos = {2,0,5};
	arma::vec dir = nominal_patch.get_center() - pos;

	
	dir = arma::normalise(dir);
	Ray original_ray(pos,dir);

	// The a-priori surface is ray traced
	double u,v;
	original_ray.single_patch_ray_casting(&nominal_patch,u,v,false);

	double apriori_range = original_ray.get_true_range();

	// arma::mat P = nominal_patch.covariance_surface_point(u,v,dir,P_X_no_correl);
	arma::mat P = nominal_patch.covariance_surface_point(u,v,dir,P_X_no_correl);

	auto start = std::chrono::system_clock::now();

	arma::vec data(N_rays);
	arma::vec simulated(N_rays);
	arma::mat training_data(3,N_rays);

	boost::progress_display progress(N_rays);

	for (unsigned int i = 0; i < N_rays; ++ i){

		Ray ray(pos,dir);

		// A random displacement is applied to the control points
		E = Z * arma::randn(18);

		// The random displacement is applied to vertices
		v0 -> set_coordinates(nominal_coords0 + E.rows(0,2));
		v1 -> set_coordinates(nominal_coords1 + E.rows(3,5));
		v2 -> set_coordinates(nominal_coords2 + E.rows(6,8));
		v3 -> set_coordinates(nominal_coords3 + E.rows(9,11));
		v4 -> set_coordinates(nominal_coords4 + E.rows(12,14));
		v5 -> set_coordinates(nominal_coords5 + E.rows(15,17));


		// The patch is ray traced
		bool hit = ray.single_patch_ray_casting(&nominal_patch,u,v,false);
		if (!hit){
			throw(std::runtime_error("Missed the target"));
		}
		data(i) = (ray.get_true_range() - apriori_range)/std::sqrt(arma::dot(dir,P * dir ));

		arma::vec r = arma::randn(1);
		simulated(i) =  r(0);

		// The training is validated by creating a point cloud of impacts

		arma::vec V0 = {0,0,0};
		arma::vec V1 = {1,0,0};
		arma::vec V2 = {0,1,0};

		arma::vec random = arma::randu<arma::vec>(2);
		
		double u = random(0);
		double v = random(1);

		arma::vec rand_coords = (1 - std::sqrt(u)) * V0 + std::sqrt(u) * ( 1 - v) * V1 + std::sqrt(u) * v * V2;
		u = rand_coords(0);
		v = rand_coords(1);

		arma::vec dir_training = arma::normalise(nominal_patch.evaluate(u,v) - pos);
		Ray ray_training(pos,dir_training);

		hit = ray_training.single_patch_ray_casting(&nominal_patch,u,v,false);
		
		if (!hit){
			throw(std::runtime_error("Missed the target"));
		}

		training_data.col(i) = ray_training.get_impact_point();

		++progress;

	}

	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end-start;

	std::cout << " Done ray tracing in " << elapsed_seconds.count() << " s\n";
	std::cout << " Range mean from data: " << arma::mean(data) << std::endl;
	std::cout << " Range mean from apriori: " << apriori_range << std::endl;
	std::cout << " Standard deviation from data: " << std::sqrt(arma::var( data, 1 )) << std::endl;
	std::cout << " Standard deviation from covariance: " << 1 << std::endl;
	std::cout << " Percentage of sd error: " << std::abs(arma::stddev(data) - 1)  * 100  << " % " << std::endl;

	simulated.save("simulated.txt",arma::raw_ascii);
	data.save("data.txt",arma::raw_ascii);

	// The nominal patch is saved
	dummy.save_both("perturbed") ;

	// Creating the training PC
	PC training_pc(dir,training_data);

	// The patch is moved back to its nominal coordinates
	v0 -> set_coordinates(nominal_coords0);
	v1 -> set_coordinates(nominal_coords1);
	v2 -> set_coordinates(nominal_coords2);
	v3 -> set_coordinates(nominal_coords3);
	v4 -> set_coordinates(nominal_coords4);
	v5 -> set_coordinates(nominal_coords5);

	dummy.construct_kd_tree_control_points();
	ShapeFitterBezier shape_fitter(&dummy,&training_pc);

	shape_fitter.fit_shape_batch(0,0);
	Bezier * trained_patch = static_cast<Bezier *>(dummy . get_elements() -> at(0).get());

	std::cout << "Trained covariance: " << std::endl;
	std::cout <<  trained_patch -> get_P_X().diag();

	std::cout << "True covariance: " << std::endl;
	std::cout << P_X.diag();

	std::cout << "Trained point standard deviations: " << std::endl;
	std::cout << arma::sqrt(trained_patch -> get_P_X().diag());

	std::cout << "True point standard deviations: " << std::endl;
	std::cout << arma::sqrt(P_X.diag());

	std::cout << "Error (\%):\n";
	std::cout << (arma::sqrt(trained_patch -> get_P_X().diag()) - arma::sqrt(P_X.diag())) / arma::sqrt(P_X.diag()) * 100;

	training_pc.save("training_pc.obj");

	std::cout << "Trained log-likelihood:" << std::endl;
	std::cout << trained_patch -> evaluate_log_likelihood(trained_patch -> get_P_X()) << std::endl;

	std::cout << "True log-likelihood:" << std::endl;
	std::cout << trained_patch -> evaluate_log_likelihood(P_X) << std::endl;


	return 0;
}