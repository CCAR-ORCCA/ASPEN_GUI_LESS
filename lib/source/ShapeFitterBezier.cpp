#include <ShapeFitterBezier.hpp>
#include <ShapeModelBezier.hpp>
#include <ShapeModelTri.hpp>
#include <PointCloud.hpp>
#include <PointCloudIO.hpp>


#include <PointNormal.hpp>
#include <ControlPoint.hpp>
#include <Footpoint.hpp>
#include <Bezier.hpp>
#include <Ray.hpp>
#include "boost/progress.hpp"

ShapeFitterBezier::ShapeFitterBezier(ShapeModelTri<ControlPoint> * psr_shape,
	ShapeModelBezier<ControlPoint> * shape_model,
	PointCloud<PointNormal> * pc) {

	this -> psr_shape = psr_shape;
	this -> shape_model = shape_model;
	this -> pc = pc;
} 


bool ShapeFitterBezier::fit_shape_batch(unsigned int N_iter, double ridge_coef){

	std::cout << "- Fitting shape ...\n";

	// Because the a-priori is really good, P_tilde from the cloud will not be reassigned 
	// to another element so there is no need to recompute the KD tree

	// The initial matches are found
	std::vector<Footpoint> footpoints = this -> find_footpoints_omp();
	PointCloud<PointNormal> pc_footpoints;

	for (int i = 0; i < footpoints.size(); ++i ){
		pc_footpoints.push_back(footpoints[i].Pbar);
	}

	PointCloudIO<PointNormal>::save_to_obj(pc_footpoints,"footpoints_pc.obj");


	for (unsigned int i = 0; i < N_iter; ++i){

		std::cout << "\nIteration " << i + 1 << " / " << N_iter << std::endl;

		std::cout << "\n\t Updating shape from the " << footpoints.size() <<  " footpoints...\n";

		bool done = this -> update_shape(footpoints,ridge_coef);

		std::cout << "\n\t Refining footpoints" << std::endl;
		boost::progress_display progress_1(footpoints.size());
		#pragma omp parallel for
		for (int e = 0; e < footpoints.size(); ++e){
			if (footpoints[e].element >= 0){
				this -> refine_footpoint_coordinates(this -> shape_model -> get_element(footpoints[e].element),footpoints[e]);
			}
			++progress_1;
		}

		std::vector<Footpoint> footpoints_temp;
		std::cout << "\n\t Pruning footpoints" << std::endl;
		boost::progress_display progress_2(footpoints.size());
		
		for (int e = 0; e < footpoints.size(); ++e){
			if (footpoints[e].element >= 0){
				footpoints_temp.push_back(footpoints[e]);
			}
			++progress_2;
		}

		std::cout << "\n\t Discarded " << (int)(footpoints.size()) - (int)(footpoints_temp.size()) << " from the " << footpoints.size() << " initial ones\n";
		footpoints = footpoints_temp;

		if (done){
			break;
		}

	}

	std::cout << "\n - Done fitting. Training covariances... \n";

	this -> train_shape_covariances(footpoints);


	return false;

}


void ShapeFitterBezier::penalize_tangential_motion(std::vector<T>& coeffs,
	unsigned int N_measurements){

	for (unsigned int index =  0 ; index < this -> shape_model -> get_NControlPoints(); ++index){

		const ControlPoint & point = this -> shape_model -> get_point(index);

		const Bezier & patch =  this -> shape_model -> get_element(*point.get_owning_elements().begin());

		const std::vector<int> & points = patch.get_points();


		auto it_index = std::find(points.begin(), points.end(), index);

		if (it_index == points.end()){
			throw(std::runtime_error("Could not find point in patch"));
		}

		auto indices = patch.get_local_indices(std::distance(points.begin(),it_index));

		unsigned int i = std::get<0>(indices);
		unsigned int j = std::get<1>(indices);
		double u = double(i) / double(patch.get_degree());
		double v = double(j) / double(patch.get_degree());


		arma::vec::fixed<3> n = patch.get_normal_coordinates(u, v);
		arma::mat proj = 10 * double(N_measurements) / double( this -> shape_model -> get_NElements()) * (arma::eye<arma::mat>(3,3) - n * n.t());

		// unsigned int index = this -> shape_model -> get_point_index(*point);
		// std::cout << index + 1 << "/" << control_points -> size() << std::endl;

		unsigned int row = 3 * index;

		coeffs.push_back(T(row + 0,row + 0,proj(0,0)));
		coeffs.push_back(T(row + 1,row + 0,proj(1,0)));
		coeffs.push_back(T(row + 2,row + 0,proj(2,0)));
		coeffs.push_back(T(row + 0,row + 1,proj(0,1)));
		coeffs.push_back(T(row + 1,row + 1,proj(1,1)));
		coeffs.push_back(T(row + 2,row + 1,proj(2,1)));
		coeffs.push_back(T(row + 0,row + 2,proj(0,2)));
		coeffs.push_back(T(row + 1,row + 2,proj(1,2)));
		coeffs.push_back(T(row + 2,row + 2,proj(2,2)));

	}


}










void ShapeFitterBezier::add_to_problem(
	std::vector<T>& coeffs,
	EigVec & N,
	const double y,
	const std::vector<arma::rowvec> & elements_to_add,
	const std::vector<int> & global_indices){


	for (auto outer_index = 0; outer_index != global_indices.size(); ++outer_index){

		int row = 3 * (global_indices[outer_index]);

		auto iter_row = std::next(elements_to_add.begin(),outer_index);

		double H_i_row = iter_row -> at(0);
		double H_i_row_1 = iter_row -> at(1);
		double H_i_row_2 = iter_row -> at(2);


		for (auto inner_index = 0; inner_index != global_indices.size(); ++inner_index){


			int col = 3 * (global_indices[inner_index]);

			auto iter_col = std::next(elements_to_add.begin(),inner_index);

			double H_i_col = iter_col -> at(0);
			double H_i_col_1 = iter_col -> at(1);
			double H_i_col_2 = iter_col -> at(2);


			coeffs.push_back(T(row,col,H_i_row * H_i_col));
			coeffs.push_back(T(row,col + 1,H_i_row * H_i_col_1));  
			coeffs.push_back(T(row,col + 2,H_i_row * H_i_col_2));          
			coeffs.push_back(T(row + 1,col,H_i_row_1 * H_i_col));   
			coeffs.push_back(T(row + 1,col + 1,H_i_row_1 * H_i_col_1));          
			coeffs.push_back(T(row + 1,col + 2,H_i_row_1 * H_i_col_2));  
			coeffs.push_back(T(row + 2,col,H_i_row_2 * H_i_col));          
			coeffs.push_back(T(row + 2,col + 1,H_i_row_2 * H_i_col_1));          			   
			coeffs.push_back(T(row + 2,col + 2,H_i_row_2 * H_i_col_2));          

		}	



		N(row) += y * H_i_row;
		N(row + 1) += y * H_i_row_1;
		N(row + 2) += y * H_i_row_2;

	}

}


bool ShapeFitterBezier::update_shape(std::vector<Footpoint> & footpoints,double ridge_coef){



	// The normal and information matrices are created
	unsigned int N = this -> shape_model -> get_NControlPoints();
	arma::vec residuals = arma::zeros<arma::vec>(footpoints.size());

	std::vector<T> coefficients;            // list of non-zeros coefficients
	EigVec Nmat(3 * N); 
	Nmat.setZero();
	SpMat Lambda(3 * N, 3 * N);
	arma::mat dndCk;
	boost::progress_display progress(footpoints.size());

	// All the measurements are processed	
	for (unsigned int k = 0; k < footpoints.size(); ++k){
		
		const Footpoint & footpoint = footpoints[k];
		const Bezier & patch = this -> shape_model -> get_element(footpoint . element);

		const std::vector<int> & control_points = patch . get_points();
		
		std::vector<int> global_indices;

		std::vector<arma::rowvec> elements_to_add;

		// The different control points for this patch have their contribution added
		// for (auto iter_points = control_points.begin(); iter_points != control_points.end(); ++iter_points){


		for (int l = 0; l < control_points.size(); ++l){

			auto local_indices = patch.get_local_indices(l);

			unsigned int i = std::get<0>(local_indices);
			unsigned int j = std::get<1>(local_indices);

			dndCk = patch.partial_n_partial_Ck(footpoint . u,footpoint . v,i, j,patch.get_degree());

			double B = Bezier::bernstein(footpoint . u,footpoint . v,i,j,patch.get_degree());
			
			elements_to_add.push_back((B * footpoint . n.t() - (footpoint . Ptilde - footpoint . Pbar).t() * dndCk ));

			global_indices.push_back(control_points[l]);
		}
		

		double y = arma::dot(footpoint . n,footpoint . Ptilde - footpoint . Pbar);


		this -> add_to_problem(coefficients,Nmat,y,elements_to_add,global_indices);

		residuals(k) = y;

		++ progress;


	}

	std::cout << "- Penalizing tangential motion\n";
	this -> penalize_tangential_motion(coefficients,footpoints.size());

	std::cout << "- Setting Lambda from the " << coefficients.size() <<  " coefs\n";

	// The information matrix is constructed
	Lambda.setFromTriplets(coefficients.begin(), coefficients.end());

	// The cholesky decomposition of Lambda is computed
	std::cout << "- Computing cholesky decomposition of Lambda\n";

	Eigen::SimplicialCholesky<SpMat> chol(Lambda);  

	// The deviation is computed
	std::cout << "- Solving for deviation\n";

	EigVec deviation = chol.solve(Nmat);    

	arma::vec dC(3*N);

	std::cout << "- Applying deviation\n";
	
	#pragma omp parallel for
	for (unsigned int i = 0; i < 3 * N; ++i){
		dC(i) = deviation(i);
	}

	double update_norm = 0;
	for (unsigned int k = 0; k < N; ++k){
		update_norm += arma::norm(dC.subvec(3 * k, 3 * k + 2))/N;

	}

	std::cout << "\nAverage update norm: " << update_norm << std::endl;
	std::cout << "Residuals: \n";
	std::cout << "- Mean: " << arma::mean(residuals) << std::endl;
	std::cout << "- Standard deviation: " << arma::stddev(residuals) << std::endl;

	// The deviations are added to the coordinates
	auto control_points = this -> shape_model -> get_points();

	for (int g = 0; g < this -> shape_model -> get_NControlPoints(); ++g){

		ControlPoint & p = this -> shape_model -> get_point(g);
		p.set_point_coordinates(p.get_point_coordinates()+ dC.rows(3 * g, 3 * g + 2));
	}

	return  (update_norm < 1e-5);


}

std::vector<Footpoint> ShapeFitterBezier::find_footpoints_omp() const{

	std::vector<Footpoint> tentative_footpoints,footpoints;

	std::cout << "\t\tFinding footpoints ...";

	for (unsigned int i = 0; i < this -> pc -> size(); ++i){
		
		Footpoint footpoint;
		footpoint.Ptilde = this -> pc -> get_point_coordinates(i);
		footpoint.ntilde = this -> pc -> get_normal_coordinates(i);
		footpoint.u = 1./3;
		footpoint.v = 1./3;
		footpoint.element = -1;

		
		tentative_footpoints.push_back(footpoint);
	}
	boost::progress_display progress(this -> pc -> size());
	
	#pragma omp parallel for
	for (unsigned int i = 0; i < this -> pc -> size(); ++i){
		this -> match_footpoint_to_element(tentative_footpoints[i]);
		++progress;
	}

	for (unsigned int i = 0; i < this -> pc -> size(); ++i){
		if (tentative_footpoints[i].element >= 0){
			footpoints.push_back(tentative_footpoints[i]);
		}
	}


	return footpoints;

}



void ShapeFitterBezier::match_footpoint_to_element(Footpoint & footpoint) const {
	

	// The prospective element this footpoint belongs to is found by ray-tracing the shape from the point cloud
	// along +/- the normal at the Ptilde

	Ray ray_plus(footpoint.Ptilde, footpoint.ntilde);
	Ray ray_minus(footpoint.Ptilde, -footpoint.ntilde);


	// The ShapeModelTri is ray-traced

	this -> psr_shape -> ray_trace(&ray_plus,false);
	this -> psr_shape -> ray_trace(&ray_minus);


	double distance_hit_plus = ray_plus.get_true_range();
	double distance_hit_minus = ray_minus.get_true_range();
	int element_hit_plus = ray_plus . get_hit_element();
	int element_hit_minus = ray_minus . get_hit_element();


	if (distance_hit_plus < distance_hit_minus && element_hit_plus != -1){

		const Bezier & patch = this -> shape_model -> get_element(element_hit_plus);
		ShapeFitterBezier::refine_footpoint_coordinates(patch,footpoint);

	}
	else if (distance_hit_plus > distance_hit_minus && element_hit_minus != -1){

		const Bezier & patch = this -> shape_model -> get_element(element_hit_minus);

		ShapeFitterBezier::refine_footpoint_coordinates(patch,footpoint);

	}

}

bool ShapeFitterBezier::refine_footpoint_coordinates(const Bezier & patch,Footpoint & footpoint){

	arma::mat::fixed<2,2> H = arma::zeros<arma::mat>(2,2);
	arma::vec::fixed<2> Y = arma::zeros<arma::vec>(2);
	arma::vec::fixed<2> dchi;
	arma::mat::fixed<3,2> dbezier_dchi;
	arma::vec::fixed<2> chi = {footpoint.u,footpoint.v};

	arma::vec::fixed<3> Pbar = patch.evaluate(chi(0),chi(1));
	unsigned int N_iter = 30;

	for (unsigned int i = 0; i < N_iter; ++i){
		dbezier_dchi = patch.partial_bezier(chi(0),chi(1));

		H.row(0) = dbezier_dchi.col(0).t() * dbezier_dchi - (footpoint.Ptilde - Pbar).t() * patch.partial_bezier_du(chi(0),chi(1));
		H.row(1) = dbezier_dchi.col(1).t() * dbezier_dchi - (footpoint.Ptilde - Pbar).t() * patch.partial_bezier_dv(chi(0),chi(1));

		Y(0) =  arma::dot(dbezier_dchi.col(0),footpoint.Ptilde - Pbar);
		Y(1) =  arma::dot(dbezier_dchi.col(1),footpoint.Ptilde - Pbar);

		dchi = arma::solve(H,Y);

		chi += dchi;
		Pbar = patch.evaluate(chi(0),chi(1));

		// If true, spurious footpoint search, abort
		if (arma::max(chi) > 0.99 || arma::min(chi) < 0.01 || arma::sum(chi) > 0.99 || arma::sum(chi) < 0.01 ){
			footpoint.element = -1;
			return false;
		}	

		arma::vec::fixed<3> normal = patch.get_normal_coordinates(chi(0),chi(1));
		double error = arma::norm(arma::cross(normal,arma::normalise(Pbar - footpoint.Ptilde)));
		
		if (error < 1e-5){

			footpoint . Pbar = Pbar;
			footpoint . u = chi(0);
			footpoint . v = chi(1);
			footpoint . n = normal;
			footpoint . element = patch.get_global_index();
			return true;

		}

	}
	footpoint.element = -1;
	return false;
}

void ShapeFitterBezier::train_shape_covariances(const std::vector<Footpoint> & footpoints){

	// The footpoints are assigned to the patches
	// First, patches that were seen are cleared
	// Then, the footpoints are added to the patches
	std::set<int> trained_patches;
	std::vector<int> trained_patches_vector;

	for (auto footpoint = footpoints.begin(); footpoint != footpoints.end(); ++footpoint){
		Bezier & patch = this -> shape_model -> get_element(footpoint -> element);
		
		if (trained_patches.find(patch.get_global_index()) == trained_patches.end()){
			patch.reset_footpoints();
			trained_patches.insert(patch.get_global_index());
		}

		patch.add_footpoint(*footpoint);
	}

	for (auto patch_index = trained_patches.begin(); patch_index != trained_patches.end(); ++patch_index ){
		trained_patches_vector.push_back(*patch_index);
	}
	
	// Once this is done, each patch is trained
	std::cout << "\n- Training "<< trained_patches_vector.size() <<  " patches ..." << std::endl;
	boost::progress_display progress(trained_patches_vector.size());
	
	#pragma omp parallel for
	for (int i = 0; i < trained_patches_vector.size(); ++i){
		Bezier & patch = this -> shape_model -> get_element(trained_patches_vector[i]);
		patch.train_patch_covariance();
		++progress;
	}
	
	std::cout << "- Done training " << trained_patches_vector.size() <<  " patches " << std::endl;


	// // The covariances are re-assigned to the control points
	// boost::progress_display progress_points(this -> shape_model -> get_NControlPoints());
	// std::cout << "- Assigning covariances to the  "<< this -> shape_model -> get_NControlPoints() <<  " control points ..." << std::endl;
	
	// auto control_points = this -> shape_model -> get_points();
	// for (auto point = control_points -> begin(); point != control_points -> end(); ++point){

	// 	auto elements = (*point) -> get_owning_elements();

	// 	Bezier * first_element = static_cast<Bezier *>(*elements.begin());
	// 	unsigned int first_element_index = first_element -> get_local_index(*point);

	// 	arma::mat P_C = first_element -> get_P_X().submat(
	// 		first_element_index,first_element_index,
	// 		first_element_index + 2, first_element_index + 2);

	// 	for (auto el = elements.begin(); el != elements.end(); ++el){

	// 		Bezier * element = static_cast<Bezier *>(*el);

	// 		unsigned int element_index = element -> get_local_index(*point);

	// 		arma::mat P = element -> get_P_X().submat(
	// 			element_index,element_index,
	// 			element_index + 2, element_index + 2);


	// 		if (P.max() > P_C.max()){
	// 			P_C = P;
	// 		}

	// 	}

	// 	(*point) -> set_covariance(P_C);


	// 	++progress_points;
	// }
	// std::cout << "- Done with the control points " << std::endl;


}


