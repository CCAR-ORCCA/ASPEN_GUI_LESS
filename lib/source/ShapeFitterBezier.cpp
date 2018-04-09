#include "ShapeFitterBezier.hpp"
#include "boost/progress.hpp"

ShapeFitterBezier::ShapeFitterBezier(ShapeModelBezier * shape_model,PC * pc) {
	
	this -> shape_model = shape_model;
	this -> pc = pc;
}



bool ShapeFitterBezier::fit_shape_batch(unsigned int N_iter, double ridge_coef){

	std::vector<Footpoint> footpoints;

	for (unsigned int i = 0; i < N_iter; ++i){
		this -> shape_model -> construct_kd_tree_control_points();
		
		// The footpoints are found
		footpoints = this -> find_footpoints_omp();
		
		// The shape is updated
		if (this -> update_shape(footpoints,ridge_coef)){

			break;
		}

	}

	std::cout << " - Done fitting. Recalculating footpoints\n";
	footpoints = this -> find_footpoints_omp();


	// The footpoints are assigned to the patches
	// First, patches that were seen are cleared
	// Then, the footpoints are added to the patches
	std::set<Bezier *> trained_patches;
	for (auto footpoint = footpoints.begin(); footpoint != footpoints.end(); ++footpoint){
		Bezier * patch = dynamic_cast<Bezier * >(footpoint -> element);
		
		if (trained_patches.find(patch) == trained_patches.end()){
			patch -> reset_footpoints();
			trained_patches.insert(patch);
		}
		patch -> add_footpoint(*footpoint);

	}


	// Once this is done, each patch is trained
	boost::progress_display progress(trained_patches.size());
	std::cout << "- Training "<< trained_patches.size() <<  " patches ..." << std::endl;
	for (auto patch = trained_patches.begin(); patch != trained_patches.end(); ++patch){
		(*patch) -> train_patch_covariance();
		(*patch) -> compute_range_biases();
		++progress;
	}
	std::cout << "- Done training "<< trained_patches.size() <<  " patches " << std::endl;



	// The covariances are re-assigned to the control points
	boost::progress_display progress_points(this -> shape_model -> get_NControlPoints());
	std::cout << "- Assigning covariances to the  "<< this -> shape_model -> get_NControlPoints() <<  " control points ..." << std::endl;
	

	auto control_points = this -> shape_model -> get_control_points();
	for (auto point = control_points -> begin(); point != control_points -> end(); ++point){
		
		auto elements = (*point) -> get_owning_elements();

		Bezier * first_element = static_cast<Bezier *>(*elements.begin());
		unsigned int first_element_index = first_element -> get_local_index(*point);

		arma::mat P_C = first_element -> get_P_X().submat(
			first_element_index,first_element_index,
			first_element_index + 2, first_element_index + 2);


		for (auto el = elements.begin(); el != elements.end(); ++el){

			Bezier * element = static_cast<Bezier *>(*el);

			unsigned int element_index = element -> get_local_index(*point);


			arma::mat P = element -> get_P_X().submat(
				element_index,element_index,
				element_index + 2, element_index + 2);


			if (P.max() > P_C.max()){
				P_C = P;
			}

		}

		(*point) -> set_covariance(P_C);


		++progress_points;
	}
	std::cout << "- Done with the control points " << std::endl;




	return false;

}


void ShapeFitterBezier::penalize_tangential_motion(std::vector<T>& coeffs,unsigned int N_measurements){


	auto control_points = this -> shape_model -> get_control_points();


	for (auto point =  control_points -> begin(); point !=  control_points -> end(); ++point){

		Bezier * patch = dynamic_cast<Bezier *>(    *((*point) -> get_owning_elements().begin()));
		auto indices = patch ->  get_local_indices(*point);
		unsigned int i = std::get<0>(indices);
		unsigned int j = std::get<1>(indices);
		double u = double(i) / double(patch -> get_degree());
		double v = double(j) / double(patch -> get_degree());

		arma::vec n = patch -> get_normal(u, v);
		arma::mat proj = double(N_measurements) / double( this -> shape_model -> get_NElements()) * (arma::eye<arma::mat>(3,3) - n * n.t());

		unsigned int index = this -> shape_model -> get_control_point_index(*point);

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

	std::cout << "\nUpdating shape from the " << footpoints.size()<<  " footpoints...\n";

	// The normal and information matrices are created
	unsigned int N = this -> shape_model -> get_NControlPoints();
	arma::sp_mat info_mat(3 * N,3 * N);
	arma::vec normal_mat = arma::zeros<arma::vec>(3 * N);
	arma::vec residuals = arma::zeros<arma::vec>(footpoints.size());

	std::vector<T> coefficients;            // list of non-zeros coefficients
	EigVec Nmat(3 * N); 
	Nmat.setZero();
	SpMat Lambda(3 * N, 3 * N);
	arma::mat dndCk;
	boost::progress_display progress(footpoints.size());

	// All the measurements are processed	
	for (unsigned int k = 0; k < footpoints.size(); ++k){
		++ progress;
		
		Footpoint footpoint = footpoints[k];
		Bezier * patch = dynamic_cast<Bezier *>(footpoint . element);

		auto control_points = patch -> get_control_points();
		


		std::vector<int> global_indices;

		std::vector<arma::rowvec> elements_to_add;

		// The different control points for this patch have their contribution added
		for (auto iter_points = control_points -> begin(); iter_points != control_points -> end(); ++iter_points){

			unsigned int global_point_index = this -> shape_model -> get_control_point_index(*iter_points);

			auto local_indices = patch -> get_local_indices(*iter_points);
			unsigned int i = std::get<0>(local_indices);
			unsigned int j = std::get<1>(local_indices);

			dndCk = patch -> partial_n_partial_Ck(footpoint . u,footpoint . v,i, j,patch -> get_degree());

			double B = Bezier::bernstein(footpoint . u,footpoint . v,i,j,patch -> get_degree());
			
			elements_to_add.push_back((B * footpoint . n.t() 
				- (footpoint . Ptilde - footpoint . Pbar).t() * dndCk ));


			global_indices.push_back(global_point_index);
		}
		
		double y = arma::dot(footpoint . n,footpoint . Ptilde
			- patch -> evaluate(footpoint . u,footpoint . v));

		this -> add_to_problem(coefficients,Nmat,y,elements_to_add,global_indices);

		residuals(k) = y;

	}


	this -> penalize_tangential_motion(coefficients,footpoints.size());



	// The information matrix is constructed
	Lambda.setFromTriplets(coefficients.begin(), coefficients.end());

	// MatrixXd dMat;

	// dMat = MatrixXd(Lambda);
	// Eigen::JacobiSVD<MatrixXd> svd(Lambda);
	// double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
	
	// std::cout << "conditioning: " << cond << std::endl;


	// The information matrix is regularized
	double trace = 0;
	for (int k=0; k<Lambda.outerSize(); ++k){
		for (SpMat::InnerIterator it(Lambda,k); it; ++it){
			if (it.row() == it.col()){
				trace += it.value();
			}
		}
	}

	// std::cout << "Trace : " << trace << std::endl;

	for (int k=0; k<Lambda.outerSize(); ++k){
		for (SpMat::InnerIterator it(Lambda,k); it; ++it){
			if (it.row() == it.col()){

				double & value = it.valueRef();
				value += ridge_coef * trace;
			}
		}
	}

	// std::cout << Lambda << std::endl;

	// std::cout << "RHS: " << Nmat << std::endl;


	// The cholesky decomposition of Lambda is computed
	Eigen::SimplicialCholesky<SpMat> chol(Lambda);  

	// The deviation is computed
	EigVec deviation = chol.solve(Nmat);    

	arma::vec dC(3*N);

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
	auto control_points = this -> shape_model -> get_control_points();

	for (auto iter_points = control_points -> begin(); iter_points != control_points -> end(); ++iter_points){
		unsigned int global_point_index = this -> shape_model -> get_control_point_index(*iter_points);
		(*iter_points) -> set_coordinates((*iter_points) -> get_coordinates()
			+ dC.rows(3 * global_point_index, 3 * global_point_index + 2));
	}


	return  (update_norm < 1e-5);




}




std::vector<Footpoint> ShapeFitterBezier::find_footpoints_omp() const{

	std::vector<Footpoint> pc_to_footpoint,footpoints;

	std::cout << "Finding footpoints ...";

	boost::progress_display progress(this -> pc -> get_size());

	for (unsigned int i = 0; i < this -> pc -> get_size(); ++i){
		Footpoint footpoint;
		footpoint.Ptilde = this -> pc -> get_point_coordinates(i);
		pc_to_footpoint.push_back(footpoint);
	}

	#pragma omp parallel for
	for (unsigned int i = 0; i < this -> pc -> get_size(); ++i){
		this -> find_footpoint_omp(pc_to_footpoint[i]);
		++progress;
	}

	for (unsigned int i = 0; i < this -> pc -> get_size(); ++i){
		if (pc_to_footpoint[i].element != nullptr){
			footpoints.push_back(pc_to_footpoint[i]);
		}
	}


	return footpoints;

}





void ShapeFitterBezier::find_footpoint_omp(Footpoint & footpoint) const {
	

	double distance = std::numeric_limits<double>::infinity();
	std::shared_ptr<ControlPoint> closest_control_point;

	this -> shape_model -> get_KDTreeControlPoints() -> closest_point_search(footpoint.Ptilde,
		this -> shape_model -> get_KDTreeControlPoints(),
		closest_control_point,
		distance);

	auto owning_elements = closest_control_point -> get_owning_elements();

		// The patches that this control point belongs to are searched
	for (auto el = owning_elements.begin(); el != owning_elements.end(); ++el){
		Bezier * patch = dynamic_cast<Bezier *> (*el);

		ShapeFitterBezier::find_footpoint_in_patch_omp(patch,footpoint);
		
	}


}

void ShapeFitterBezier::find_footpoint_in_patch_omp(Bezier * patch,Footpoint & footpoint){

	arma::mat::fixed<2,2> H = arma::zeros<arma::mat>(2,2);
	arma::vec::fixed<2> Y = arma::zeros<arma::vec>(2);
	arma::vec::fixed<2> dchi;
	arma::mat::fixed<3,2> dbezier_dchi;
	arma::vec::fixed<2> chi = {1./3,1./3};

	arma::vec Pbar = patch -> evaluate(chi(0),chi(1));
	unsigned int N_iter = 30;

	for (unsigned int i = 0; i < N_iter; ++i){

		dbezier_dchi = patch -> partial_bezier(chi(0),chi(1));

		H.row(0) = dbezier_dchi.col(0).t() * dbezier_dchi - (footpoint.Ptilde - Pbar).t() * patch -> partial_bezier_du(chi(0),chi(1));
		H.row(1) = dbezier_dchi.col(1).t() * dbezier_dchi - (footpoint.Ptilde - Pbar).t() * patch -> partial_bezier_dv(chi(0),chi(1));

		Y(0) =  arma::dot(dbezier_dchi.col(0),footpoint.Ptilde - Pbar);
		Y(1) =  arma::dot(dbezier_dchi.col(1),footpoint.Ptilde - Pbar);

		dchi = arma::solve(H,Y);

		chi += dchi;
		Pbar = patch -> evaluate(chi(0),chi(1));

		double error = arma::norm(arma::cross(patch -> get_normal(chi(0),chi(1)),arma::normalise(Pbar - footpoint.Ptilde)));
		
		if (error < 1e-5){
			std::cout << "found footpoint\n";
			if (footpoint.element != nullptr){

			// If true, then the previous footpoint was better
				if (arma::norm(footpoint . Ptilde - Pbar) > arma::norm(footpoint . Ptilde - footpoint . Pbar)){
					return ;
				}
			}

			// If true, spurious footpoint
			if (arma::max(chi) > 0.99 || arma::min(chi) < 0.01 || arma::sum(chi) > 0.99 || arma::sum(chi) < 0.01 ){
				return ;
			}	

			Bezier * patch = dynamic_cast<Bezier *>(footpoint.element);
			
			footpoint . Pbar = Pbar;
			footpoint . u = chi(0);
			footpoint . v = chi(1);
			footpoint . n = patch -> get_normal(chi(0),chi(1));
			footpoint . element = patch;
		}

	}

	

}




