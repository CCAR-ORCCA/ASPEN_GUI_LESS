#include "ShapeFitterBezier.hpp"
#include "boost/progress.hpp"

ShapeFitterBezier::ShapeFitterBezier(ShapeModelBezier * shape_model,PC * pc) : ShapeFitter(pc){
	
	this -> shape_model = shape_model;
}


std::vector<Footpoint> ShapeFitterBezier::fit_shape_KF(
	unsigned int index,
	unsigned int N_iter_outer, 
	double J,
	double los_noise_sd_base,
	const arma::vec & u_dir){


	double W = 1./(los_noise_sd_base * los_noise_sd_base);
	std::vector<Footpoint> footpoints;
	
	// fit_elements has the pointers to the elements to be fit
	for (unsigned int j = 0; j < N_iter_outer; ++j){

		std::cout << "\n\n- Outer iteration : " << j + 1 << "/" << N_iter_outer <<std::endl;

		// The footpoints are first found
		auto start = std::chrono::system_clock::now();
		
		#if USE_OMP_SHAPE_FITTER
		footpoints = this -> find_footpoints_omp();
		#else
		footpoints = this -> find_footpoints();
		#endif
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end-start;

		std::cout << "- Time elapsed finding footpoints: " << elapsed_seconds.count()<< " s"<< std::endl;
		
		std::map<Element *,std::vector<Footpoint> > fit_elements_to_footpoints;

		for (unsigned int i = 0; i < footpoints.size(); ++i){
			Footpoint footpoint = footpoints[i];

			if (fit_elements_to_footpoints.find(footpoint.element) == fit_elements_to_footpoints.end()){
				std::vector<Footpoint> element_footpoints;
				element_footpoints.push_back(footpoint);
				fit_elements_to_footpoints.insert(std::make_pair(footpoint.element,element_footpoints));
			}
			else{
				fit_elements_to_footpoints[footpoint.element].push_back(footpoint);
			}
		}



		if (footpoints.size() == 0){
			return footpoints;
		}

		arma::mat Pbar_mat = arma::mat(3,footpoints.size());

		for (unsigned int k = 0; k < footpoints.size(); ++k){		
			Pbar_mat.col(k) = footpoints[k].Pbar;
		}		

		arma::vec u = {1,0,0};		
		PC pc(u,Pbar_mat);	
		pc.save("../output/pc/Pbar_" + std::to_string(j) + "_"+std::to_string(index) + ".obj");

		if (j == 0){
			arma::mat Ptilde_mat = arma::mat(3,footpoints.size());		 		
			for (unsigned int k = 0; k < footpoints.size(); ++k){		
				Ptilde_mat.col(k) = footpoints[k].Ptilde;		
			}	
			PC pc_tilde(u,Ptilde_mat);		
			pc_tilde.save("../output/pc/Ptilde_" + std::to_string(j) + "_" + std::to_string(index) + ".obj");
		}



		for (auto element_pair = fit_elements_to_footpoints.begin(); element_pair != fit_elements_to_footpoints.end(); ++element_pair){

			std::cout << "--- Recomputing footpoints" << std::endl;

			element_pair -> second = this -> recompute_footpoints(element_pair -> second);

			if (j == N_iter_outer - 1){
				this -> update_element(element_pair -> first,
					element_pair -> second,true,W,u_dir);
			}
			else{
				this -> update_element(element_pair -> first,
					element_pair -> second,false,W,u_dir);
			}
			this -> shape_model -> construct_kd_tree_control_points();
			

		}


	}


	return footpoints;

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
	std::cout << "- Training patches... " << std::endl;

	for (auto patch = trained_patches.begin(); patch != trained_patches.end(); ++patch){
		(*patch) -> train_patch_covariance();
		++progress;

	}
	std::cout << "- Done training patches " << std::endl;



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
		arma::mat proj = N_measurements * (arma::eye<arma::mat>(3,3) - n * n.t());

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

			arma::mat dndCk = patch -> partial_n_partial_Ck(footpoint . u,footpoint . v,i, j,patch -> get_degree());

			double B = Bezier::bernstein(footpoint . u,footpoint . v,i,j,patch -> get_degree());
			
			elements_to_add.push_back((B * footpoint . n.t() 
				- 0 * (footpoint . Ptilde - footpoint . Pbar).t() * dndCk ));


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





std::vector<Footpoint> ShapeFitterBezier::recompute_footpoints(const std::vector<Footpoint> & footpoints)const {

	std::vector<Footpoint> new_footpoints_temp;

	for (unsigned int i = 0; i < footpoints.size(); ++i){

		Footpoint footpoint;
		footpoint.element = NULL;

		footpoint.Ptilde = footpoints[i].Ptilde;
		Element * element = footpoints[i].element;
		
		try{
			this -> find_footpoint(footpoint,element);

			if (footpoint.element == footpoints[i].element){
				new_footpoints_temp.push_back(footpoint);
			}
		}
		catch(const MissingFootpointException & e){
		}

	}

	arma::vec distances(new_footpoints_temp.size());

	for (unsigned int i = 0; i < new_footpoints_temp.size(); ++i){
		distances(i) = arma::norm(new_footpoints_temp[i].Ptilde - new_footpoints_temp[i].Pbar );
	}

	std::vector<Footpoint> new_footpoints;
	double std = arma::stddev(distances);


	for (unsigned int i = 0; i < new_footpoints_temp.size(); ++i){
		if (distances(i) < 3 * std){
			new_footpoints.push_back(new_footpoints_temp[i]);
		}
	}

	return new_footpoints;


}



std::vector<Footpoint> ShapeFitterBezier::find_footpoints_omp() const{

	std::vector<Footpoint> footpoints;

	std::cout << "Finding footpoints...";
	// Element * element_guess = nullptr;
	// std::map<Element *,arma::vec> element_quality;


	boost::progress_display progress(this -> pc -> get_size());

	std::vector<std::shared_ptr<Footpoint> > pc_to_footpoint;

	for (unsigned int i = 0; i < this -> pc -> get_size(); ++i){

		pc_to_footpoint.push_back(nullptr);
	}

	#pragma omp parallel for
	for (unsigned int i = 0; i < this -> pc -> get_size(); ++i){
		++progress;
		pc_to_footpoint[i] = this -> find_footpoint_omp(this -> pc -> get_point_coordinates(i));
	}

	for (unsigned int i = 0; i < this -> pc -> get_size(); ++i){
		if (pc_to_footpoint[i] != nullptr){
			footpoints.push_back(*pc_to_footpoint[i]);
		}
	}


	return footpoints;

}



std::vector<Footpoint> ShapeFitterBezier::find_footpoints() const{

	std::vector<Footpoint> footpoints;

	std::cout << "Finding footpoints...";
	Element * element_guess = nullptr;
	std::map<Element *,arma::vec> element_quality;


	boost::progress_display progress(this -> pc -> get_size());

	for (unsigned int i = 0; i < this -> pc -> get_size(); ++i){
		Footpoint footpoint;
		footpoint.element = NULL;

		++progress;

		footpoint.Ptilde = this -> pc -> get_point_coordinates(i);
		
		try {
			this -> find_footpoint(footpoint,element_guess);
			footpoints.push_back(footpoint);

		}

		catch(const MissingFootpointException & e){
			// std::cout << e.what() << std::endl;

		}

	}

	std::cout << "- Total footpoints: " << footpoints.size() << std::endl;

	return footpoints;
}

void ShapeFitterBezier::find_footpoint(Footpoint & footpoint,Element * & element_guess) const {

	if (this -> shape_model -> get_NElements() > 1){

		// if a guess is available, this footpoint is looked for in this element
		if (element_guess != nullptr){

			try{
				ShapeFitterBezier::find_footpoint_in_patch(dynamic_cast<Bezier *>(element_guess),footpoint);
				return;

			}


			// If the exception is thrown, this guess was the wrong one
			catch(const MissingFootpointException & e){
				element_guess = nullptr;
			}


		}


		// The closest control point to this measurement is looked for across the 
		// shape model using the KD tree

		double distance = std::numeric_limits<double>::infinity();
		std::shared_ptr<ControlPoint> closest_control_point;

		this -> shape_model -> get_KDTree_control_points() -> closest_point_search(
			footpoint.Ptilde,
			this -> shape_model -> get_KDTree_control_points(),
			closest_control_point,
			distance);

		auto owning_elements = closest_control_point -> get_owning_elements();

		// The patches that this control point belongs to are searched
		for (auto el = owning_elements.begin(); el != owning_elements.end(); ++el){
			Bezier * patch = dynamic_cast<Bezier *> (*el);

			try{
				ShapeFitterBezier::find_footpoint_in_patch(patch,footpoint);
				element_guess = patch;

				// break;
			}
			catch(const MissingFootpointException & e){

			}
		}

		if (footpoint.element == NULL){
			throw MissingFootpointException();
		}


	}

	// If the shape model only has one patch, this is trivial
	else {
		Bezier * patch = dynamic_cast<Bezier *> (this -> shape_model -> get_elements() -> front().get());
		ShapeFitterBezier::find_footpoint_in_patch(patch,footpoint);
	}


}




std::shared_ptr<Footpoint> ShapeFitterBezier::find_footpoint_omp(arma::vec P_tilde) const {
	std::shared_ptr<Footpoint> footpoint_temp = nullptr;

	if (this -> shape_model -> get_NElements() > 1){

		// The closest control point to this measurement is looked for across the 
		// shape model using the KD tree

		double distance = std::numeric_limits<double>::infinity();
		std::shared_ptr<ControlPoint> closest_control_point;

		this -> shape_model -> get_KDTree_control_points() -> closest_point_search(
			P_tilde,
			this -> shape_model -> get_KDTree_control_points(),
			closest_control_point,
			distance);

		auto owning_elements = closest_control_point -> get_owning_elements();

		// The patches that this control point belongs to are searched
		for (auto el = owning_elements.begin(); el != owning_elements.end(); ++el){
			Bezier * patch = dynamic_cast<Bezier *> (*el);

			try{
				ShapeFitterBezier::find_footpoint_in_patch_omp(P_tilde,patch,footpoint_temp);

			}
			catch(const MissingFootpointException & e){

			}
		}

		
	}

	// If the shape model only has one patch, this is trivial
	else {
		Bezier * patch = dynamic_cast<Bezier *> (this -> shape_model -> get_elements() -> front().get());
		ShapeFitterBezier::find_footpoint_in_patch_omp(P_tilde,patch,footpoint_temp);
	}

	return footpoint_temp;


}

void ShapeFitterBezier::find_footpoint_in_patch_omp(arma::vec P_tilde,Bezier * patch,std::shared_ptr<Footpoint> & footpoint){

	arma::mat H = arma::zeros<arma::mat>(2,2);
	arma::vec Y = arma::zeros<arma::vec>(2);
	arma::vec dchi;
	arma::mat dbezier_dchi;
	arma::vec chi = {1./3,1./3};

	arma::vec Pbar = patch -> evaluate(chi(0),chi(1));
	unsigned int N_iter = 30;

	for (unsigned int i = 0; i < N_iter; ++i){

		dbezier_dchi = patch -> partial_bezier(chi(0),chi(1));

		H.row(0) = dbezier_dchi.col(0).t() * dbezier_dchi - (P_tilde - Pbar).t() * patch -> partial_bezier_du(chi(0),chi(1));
		H.row(1) = dbezier_dchi.col(1).t() * dbezier_dchi - (P_tilde - Pbar).t() * patch -> partial_bezier_dv(chi(0),chi(1));

		Y(0) =  arma::dot(dbezier_dchi.col(0),P_tilde - Pbar);
		Y(1) =  arma::dot(dbezier_dchi.col(1),P_tilde - Pbar);

		dchi = arma::solve(H,Y);

		chi += dchi;
		Pbar = patch -> evaluate(chi(0),chi(1));

		double error = arma::norm(arma::cross(patch -> get_normal(chi(0),chi(1)),arma::normalise(Pbar - P_tilde)));
		if (error < 1e-5){


			if (footpoint != nullptr){
				Bezier * patch = dynamic_cast<Bezier *>(footpoint -> element);
				arma::vec P = patch -> evaluate(footpoint -> u,footpoint -> v);

				// If true, then the previous footpoint was better
				if (arma::norm(footpoint -> Ptilde - P) > arma::norm(footpoint -> Ptilde - footpoint -> Pbar)){
					return ;
				}

			}

			if (arma::max(chi) > 0.99 || arma::min(chi) < 0.01 || arma::sum(chi) > 0.99 || arma::sum(chi) < 0.01 ){
				return ;
			}	

			if (footpoint == nullptr){
				footpoint = std::make_shared<Footpoint>(Footpoint());
				footpoint -> Ptilde = P_tilde;
			}
			
			footpoint -> Pbar = Pbar;
			footpoint -> u = chi(0);
			footpoint -> v = chi(1);
			footpoint -> n = patch -> get_normal(chi(0),chi(1));
			footpoint -> element = patch;
			return;
		}

	}

	return;

}


void ShapeFitterBezier::find_footpoint_in_patch(Bezier * patch,Footpoint & footpoint){

	arma::mat H = arma::zeros<arma::mat>(2,2);
	arma::vec Y = arma::zeros<arma::vec>(2);
	arma::vec dchi;
	arma::mat dbezier_dchi;
	arma::vec chi = {1./3,1./3};

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
		if (error < 1e-3){


			if (footpoint.element != NULL){
				Bezier * patch = dynamic_cast<Bezier *>(footpoint.element);
				arma::vec P = patch -> evaluate(footpoint.u,footpoint.v);

				// If true, then the previous footpoint was better
				if (arma::norm(footpoint.Ptilde - P) > arma::norm(footpoint.Ptilde - footpoint.Pbar)){
					throw MissingFootpointException();
				}

			}

			if (arma::max(chi) > 0.99 || arma::min(chi) < 0.01 || arma::sum(chi) > 0.99 || arma::sum(chi) < 0.01 ){
				throw MissingFootpointException();
			}	
			footpoint.Pbar = Pbar;
			footpoint.u = chi(0);
			footpoint.v = chi(1);
			footpoint.n = patch -> get_normal(chi(0),chi(1));
			footpoint.element = patch;
			return;
		}

	}

	throw MissingFootpointException();

}



void ShapeFitterBezier::get_barycentric_coordinates(const arma::vec & Pbar,double & u, double & v, Facet * facet) const{


}

double ShapeFitterBezier::compute_residuals(std::vector<std::pair<arma::vec,Footpoint > > & measurement_pairs) const{

	return 1.;
}



void ShapeFitterBezier::save(std::string path, arma::mat & Pbar_mat) const {


}


bool ShapeFitterBezier::update_element(Element * element, 
	std::vector<Footpoint> & footpoints,
	bool store_info_mat,
	double W,
	const arma::vec & u_dir){


	double R = 1./W;



	if (footpoints.size() == 0){
		return false;
	}


	// Check if the info matrix has been initialized
	if (element -> get_info_mat_ptr() == nullptr){
		element -> initialize_info_mat();
		element -> initialize_dX_bar();
	}

	unsigned int N = 3 * element -> get_control_points() -> size();

	// The normal and information matrices are created
	arma::mat info_mat(*element -> get_info_mat_ptr());

	arma::vec normal_mat = info_mat * (*element -> get_dX_bar_ptr());
	arma::vec residuals = arma::zeros<arma::vec>(footpoints.size());

	Bezier * patch = dynamic_cast<Bezier *>(element);
	auto control_points = patch -> get_control_points();

	// All the measurements are processed	
	#pragma omp parallel for reduction (+:info_mat,normal_mat)
	
	for (unsigned int k = 0; k < footpoints.size(); ++k){

		Footpoint footpoint = footpoints[k];

		arma::mat Hi(1,N);






		// The different control points for this patch have their contribution added
		for (unsigned int point_index = 0; point_index < control_points -> size(); ++point_index){

			std::shared_ptr<ControlPoint> point = control_points -> at(point_index);

			auto local_indices = patch -> get_local_indices(point);

			unsigned int i = std::get<0>(local_indices);
			unsigned int j = std::get<1>(local_indices);
			unsigned int degree = patch -> get_degree();

			double B = Bezier::bernstein(footpoint . u,footpoint . v,i,j,degree);



			// partials assuming that n is constant
			Hi.cols(3 * point_index, 3 * point_index + 2) = B * footpoint . n.t();

			// partials accounting for the change in n
			Hi.cols(3 * point_index, 3 * point_index + 2) -= (footpoint.Ptilde - footpoint.Pbar).t() * patch -> partial_n_partial_Ck(footpoint . u,
				footpoint . v,
				i, j,
				degree);


		}


		double y = arma::dot(footpoint . n,footpoint . Ptilde - footpoint.Pbar);

		residuals(k) = y;

		double R_augmented = R ;
		double W_augmented = 1./R_augmented;



		normal_mat += Hi.t() * W_augmented * y;
		info_mat +=  Hi.t() * W_augmented * Hi;

	}


	info_mat += 1e-2 * arma::trace(info_mat) * arma::eye<arma::mat>(info_mat.n_rows,info_mat.n_rows);
	
	// The deviation is computed

	arma::vec dC = arma::solve(info_mat,normal_mat);

	// The a-priori deviation is adjusted
	*element -> get_dX_bar_ptr() = *element -> get_dX_bar_ptr() - dC;
	
	double update_norm = 0;
	unsigned int size = int(N / 3.);

	for (unsigned int k = 0; k < size; ++k){
		update_norm += arma::norm(dC.subvec(3 * k, 3 * k + 2))/size;

	}

	std::cout << "Updating element from the " << footpoints.size()<<  " footpoints...\n";

	std::cout << "- Maximum information: " << arma::abs(info_mat).max() << std::endl;
	std::cout << "- Information matrix conditioning: " << arma::cond(info_mat) << std::endl;
	std::cout << "- Information matrix determinant: " << arma::det(info_mat) << std::endl;
	std::cout << "- Average update norm: " << update_norm << std::endl;
	std::cout << "- Residuals: \n";
	std::cout << "--  Min: " << arma::min(arma::abs(residuals)) << std::endl;
	std::cout << "--  Max: " << arma::max(arma::abs(residuals)) << std::endl;
	std::cout << "--  Mean: " << arma::mean(residuals) << std::endl;
	std::cout << "--  Standard deviation: " << arma::stddev(residuals) << std::endl;
	


	// The deviations are added to the coordinates

	for (unsigned int k = 0; k < control_points -> size(); ++k){

		std::shared_ptr<ControlPoint> point = control_points -> at(k);

		auto local_indices = patch -> get_local_indices(point);

		unsigned int i = std::get<0>(local_indices);
		unsigned int j = std::get<1>(local_indices);
		unsigned int degree = patch -> get_degree();

		arma::vec n = patch -> get_normal(double(i) / degree,double(j) / degree);

		point -> set_coordinates(point -> get_coordinates()
			+ dC.rows(3 * k, 3 * k+ 2));
	}

	
	bool has_converged;
	if (std::abs(arma::mean(residuals)) < 1e-2){
		std::cout << "-- Element has converged (residuals)\n";
		has_converged = true;
	}
	else if (update_norm < 5e-2){
		std::cout << "-- Element has converged (update norm)\n";
		has_converged = true;
	}

	else{
		has_converged = false;
	}

	if (store_info_mat){
		// The information matrix is stored

		// if (arma::det(info_mat) > 1){
		// 	arma::mat M_reg = 1e-7 * arma::trace(info_mat) * arma::eye<arma::mat>(N,N);
		// 	double M = std::abs(arma::mean(residuals));

		// 	arma::mat Q = M * M * arma::eye<arma::mat>(N,N);
		// 	std::cout << "- Information matrix eigenvalues before regularization: \n";
		// 	std::cout << arma::eig_sym( info_mat ).t() << std::endl;

		// 	info_mat = arma::inv( arma::inv(info_mat + M_reg) + Q) - M_reg;

		// 	std::cout << "- Information matrix eigenvalues after regularization: \n";
		// 	std::cout << arma::eig_sym( info_mat ).t() << std::endl;
		// }

		(*element -> get_info_mat_ptr()) = info_mat;
		element -> get_dX_bar_ptr() -> fill(0);
		std::cout << "--- Done with this patch\n" << std::endl;		
	}
	return has_converged;

}




