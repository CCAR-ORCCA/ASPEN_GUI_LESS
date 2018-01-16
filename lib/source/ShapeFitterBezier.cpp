#include "ShapeFitterBezier.hpp"
#include "boost/progress.hpp"


ShapeFitterBezier::ShapeFitterBezier(ShapeModelBezier * shape_model,PC * pc) : ShapeFitter(pc){
	
	this -> shape_model = shape_model;
}


bool ShapeFitterBezier::fit_shape_KF(
	unsigned int index,
	unsigned int N_iter, 
	double J,
	const arma::mat & DS, 
	const arma::vec & X_DS,
	double los_noise_sd_base){


	double W = 1./(los_noise_sd_base * los_noise_sd_base);

	
	// The footpoints are first found
	std::vector<Footpoint> 	footpoints = this -> find_footpoints();
	bool element_has_converged  = false;

	std::map<Element *,std::vector<Footpoint> > fit_elements_to_footpoints;
	std::map<Element *,bool  > fit_elements_to_stalling_status;


	for (unsigned int i = 0; i < footpoints.size(); ++i){
		Footpoint footpoint = footpoints[i];

		if (fit_elements_to_footpoints.find(footpoint.element) == fit_elements_to_footpoints.end()){
			std::vector<Footpoint> element_footpoints;
			element_footpoints.push_back(footpoint);
			fit_elements_to_footpoints.insert(std::make_pair(footpoint.element,element_footpoints));
			fit_elements_to_stalling_status.insert(std::make_pair(footpoint.element,false));
		}
		else{
			fit_elements_to_footpoints[footpoint.element].push_back(footpoint);
		}
	}


	// fit_elements has the pointers to the elements to be fit
	for (unsigned int j = 0; j < N_iter; ++j){

		std::cout << "- Outer iteration : " << j + 1<< "/" << N_iter - 1 <<std::endl;

		
		
	// Only footpoints over "good" elements are kept
		if (footpoints.size() == 0){
			return false;
		}

		if (j == 0){
			arma::mat Pbar_mat = arma::mat(3,footpoints.size());
			arma::mat Ptilde_mat = arma::mat(3,footpoints.size());		 		

			for (unsigned int k = 0; k < footpoints.size(); ++k){		
				Pbar_mat.col(k) = footpoints[k].Pbar;
				Ptilde_mat.col(k) = footpoints[k].Ptilde;		

			}		

			arma::vec u = {1,0,0};		
			// PC pc(u,Pbar_mat);	
			// PC pc_tilde(u,Ptilde_mat);		

			// pc.save("../output/pc/Pbar_" +std::to_string(index) + ".obj");
			// pc_tilde.save("../output/pc/Ptilde_" +std::to_string(index) + ".obj");

		}


		for (auto element_pair = fit_elements_to_footpoints.begin(); 
			element_pair != fit_elements_to_footpoints.end(); ++element_pair){

			for (unsigned int i = 0; i < N_iter; ++i){
				std::cout << "-- Inner iteration : " << i + 1 << "/" << N_iter - 1 <<std::endl;
				std::cout << "--- Recomputing footpoints" << std::endl;
				
				element_pair -> second = this -> recompute_footpoints(element_pair -> second);

				element_has_converged = this -> update_element(element_pair -> first,
					element_pair -> second,false,W);
				
				if (element_has_converged){
					std::cout << "--- Element has converged\n";
					break;
				}

			}

		}

		if (element_has_converged){
			std::cout << "- Fit elements have all converged\n";
			break;
		}


	}

	// The information matrix of each patch is updated
	for (auto element_pair = fit_elements_to_footpoints.begin(); element_pair != fit_elements_to_footpoints.end(); ++element_pair){
		
		element_has_converged = this -> update_element(element_pair -> first,
			element_pair -> second,true,W);
		
	}


	return false;

}


std::vector<Footpoint> ShapeFitterBezier::recompute_footpoints(const std::vector<Footpoint> & footpoints)const {

	std::vector<Footpoint> new_footpoints;

	for (unsigned int i = 0; i < footpoints.size(); ++i){

		Footpoint footpoint;
		footpoint.element = NULL;

		footpoint.Ptilde = footpoints[i].Ptilde;
		Element * element = footpoints[i].element;
		
		try{
			this -> find_footpoint(footpoint,element);

			if (footpoint.element == footpoints[i].element){
				new_footpoints.push_back(footpoint);
			}
		}
		catch(const MissingFootpointException & e){
		}

	}
	return new_footpoints;


}


std::vector<Footpoint> ShapeFitterBezier::find_footpoints() const{

	std::vector<Footpoint> footpoints;

	std::cout << "Finding footpoints...\n";
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

			if (element_quality.find(footpoint.element) == element_quality.end()){

				arma::vec quality = {
					footpoint.u,
					footpoint.u,
					footpoint.v,
					footpoint.v
				};

				element_quality.insert(std::make_pair(footpoint.element,quality));

			}
			else{
				arma::vec quality = element_quality[footpoint.element];
				quality(0) = std::min(footpoint.u,quality(0));
				quality(1) = std::max(footpoint.u,quality(1));
				quality(2) = std::min(footpoint.v,quality(2));
				quality(3) = std::max(footpoint.v,quality(3));
				element_quality[footpoint.element] = quality;
			}

		}

		catch(const MissingFootpointException & e){
			// std::cout << e.what() << std::endl;

		}

	}


	// Only the footpoints whose footpoints 
    // have been sufficiently covered are kept
	std::vector<Footpoint> footpoints_clean;

	for (unsigned int i =0; i < footpoints.size(); ++i){
		arma::vec quality = element_quality[footpoints[i].element];

		double area = (quality(1) - quality(0)) * (quality(3) - quality(2));

		// If at least two thirds of the unit triangle has been seen
		if (std::abs(area - 1) < 0.7){
			footpoints_clean.push_back(footpoints[i]);
		}

	}

	std::cout << "- Total footpoints: " << footpoints.size() << std::endl;
	std::cout << "- Kept footpoints: " << footpoints_clean.size() << std::endl;




	return footpoints_clean;
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
		if (error < 1e-4){


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
	double W){

	if (footpoints.size() == 0){
		return false;
	}

	std::cout << "Updating element from the " << footpoints.size()<<  " footpoints...\n";

	// Check if the info matrix has been initialized
	if (element -> get_info_mat_ptr() == nullptr){
		element -> initialize_info_mat();
		element -> initialize_dX_bar();
	}

	unsigned int N = 3 * element -> get_control_points() -> size();

	// The normal and information matrices are created
	arma::mat info_mat(*element -> get_info_mat_ptr());

	// arma::mat info_mat = arma::zeros<arma::mat>(N,N);



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

			Hi.cols(3 * point_index, 3 * point_index + 2) = B * footpoint . n.t();
		}

		double y = arma::dot(footpoint . n,footpoint . Ptilde
			- patch -> evaluate(footpoint . u,footpoint . v));

		residuals(k) = y;

		normal_mat += Hi.t() * W * y;
		info_mat +=  Hi.t() * W * Hi;

	}

	arma::mat regularized_info_mat = info_mat + 1e-1 * arma::trace(info_mat) * arma::eye<arma::mat>(info_mat.n_cols,info_mat.n_cols);

	// The deviation is computed
	arma::vec dC = 0.5 * arma::solve(regularized_info_mat,normal_mat);

	// The a-priori deviation is adjusted
	*element -> get_dX_bar_ptr() = *element -> get_dX_bar_ptr() - dC;
	

	double update_norm = 0;
	unsigned int size = int(N / 3.);

	for (unsigned int k = 0; k < size; ++k){
		update_norm += arma::norm(dC.subvec(3 * k, 3 * k + 2))/size;

	}

	std::cout << "\nSignificant figures: " << std::endl;
	std::cout << "\n- Maximum information: " << arma::abs(info_mat).max() << std::endl;
	std::cout << "\n- Information matrix conditioning: " << arma::cond(info_mat) << std::endl;
	std::cout << "\n- Information matrix determinant: " << arma::det(info_mat) << std::endl;
	std::cout << "\n- Average update norm: " << update_norm << std::endl;
	std::cout << "\n- Residuals: \n";
	std::cout << "--  Mean: " << arma::mean(residuals) << std::endl;
	std::cout << "--  Standard deviation: " << arma::stddev(residuals) << std::endl;

	// The information matrix is stored
	if (store_info_mat){

		arma::mat eigvec;
		arma::vec eigval;
		double M = std::abs(arma::mean(residuals));

		arma::eig_sym(eigval,eigvec,info_mat);
		std::cout << "-- Information matrix eigenvalues: \n";
		
		std::cout << eigval.t() << std::endl;

		for (unsigned int i = 0; i < eigval.n_rows; ++i){
			if (eigval(i) > 1./(M * M + 1./W)){
				eigval(i) = 1./(M * M + 1./W);
			}
		}

		info_mat = eigvec * arma::diagmat(eigval) * eigvec.t();

		(*element -> get_info_mat_ptr()) = info_mat;
		element -> get_dX_bar_ptr() -> fill(0);
		std::cout << "--- Done with this patch\n" << std::endl;
		return true;
	}


	// The deviations are added to the coordinates


	for (unsigned int k = 0; k < control_points -> size(); ++k){

		std::shared_ptr<ControlPoint> point = control_points -> at(k);

		auto local_indices = patch -> get_local_indices(point);

		unsigned int i = std::get<0>(local_indices);
		unsigned int j = std::get<1>(local_indices);
		unsigned int degree = patch -> get_degree();

		arma::vec n = patch -> get_normal(double(i) / degree,double(j) / degree);

		// point -> set_coordinates(point -> get_coordinates()
		// 	+ arma::dot(n,dC.rows(3 * k, 3 * k+ 2)) * n);

		point -> set_coordinates(point -> get_coordinates()
			+ dC.rows(3 * k, 3 * k+ 2));
	}


	if (std::abs(arma::mean(residuals)) < 1e-2){
		return true;
	}
	else if (update_norm < 5e-2){
		return true;
	}

	else{
		return false;
	}

}



// arma::mat ShapeFitterBezier::update_shape(std::vector<Footpoint> & footpoints,
// 	bool & has_converged){

// 	std::cout << "Updating shape from the " << footpoints.size()<<  " footpoints...\n";

// 	// Check if the info matrix has been initialized
// 	if (this -> shape_model -> get_info_mat_ptr() == nullptr){
// 		this -> shape_model -> initialize_info_mat();
// 		this -> shape_model -> initialize_dX_bar();
// 		this -> shape_model -> initialize_index_table();
// 	}

// 	// The normal and information matrices are created
// 	arma::mat info_mat(*this -> shape_model -> get_info_mat_ptr());

// 	arma::vec normal_mat = info_mat * (*this -> shape_model -> get_dX_bar_ptr());
// 	arma::vec residuals = arma::zeros<arma::vec>(footpoints.size());
// 	unsigned int N = info_mat.n_cols;

// 	boost::progress_display progress(footpoints.size());

// 	// All the measurements are processed	
// 	#pragma omp parallel for reduction (+:info_mat,normal_mat)
// 	for (unsigned int k = 0; k < footpoints.size(); ++k){
// 		++ progress;

// 		Footpoint footpoint = footpoints[k];

// 		arma::mat Hi(1,N);
// 		Bezier * patch = dynamic_cast<Bezier *>(footpoint . element);

// 		auto control_points = patch -> get_control_points();

// 		// The different control points for this patch have their contribution added
// 		for (auto iter_points = control_points -> begin(); iter_points != control_points -> end(); ++iter_points){

// 			unsigned int global_point_index = this -> shape_model -> get_control_point_index(*iter_points);

// 			auto local_indices = patch -> get_local_indices(*iter_points);

// 			unsigned int i = std::get<0>(local_indices);
// 			unsigned int j = std::get<1>(local_indices);
// 			unsigned int degree = patch -> get_degree();

// 			double B = Bezier::bernstein(footpoint . u,footpoint . v,i,j,degree);

// 			Hi.cols(3 * global_point_index, 3 * global_point_index + 2) = B * footpoint . n.t();
// 		}

// 		double y = arma::dot(footpoint . n,footpoint . Ptilde
// 			- patch -> evaluate(footpoint . u,footpoint . v));

// 		residuals(k) = y;

// 		normal_mat += Hi.t() * y;
// 		info_mat +=  Hi.t() * Hi;

// 	}

// 	// The information matrix is regularized
// 	arma::mat regularized_info_mat = info_mat;
// 	regularized_info_mat +=  1e-3 * arma::trace(info_mat) * arma::eye<arma::mat>(info_mat.n_rows,info_mat.n_cols);

// 	// The deviation is computed
// 	arma::vec dC = arma::solve(regularized_info_mat,normal_mat);

// 	// The a-priori deviation is adjusted
// 	*this -> shape_model -> get_dX_bar_ptr() = *this -> shape_model -> get_dX_bar_ptr() - dC;

// 	double update_norm = 0;
// 	unsigned int size = int(N / 3.);

// 	for (unsigned int k = 0; k < size; ++k){
// 		update_norm += arma::norm(dC.subvec(3 * k, 3 * k + 2))/size;

// 	}



// 	std::cout << "\nSignificant figures: " << std::endl;
// 	std::cout << "\n- Maximum information: " << arma::abs(info_mat).max() << std::endl;
// 	std::cout << "\n- Information matrix conditioning: " << arma::cond(info_mat) << std::endl;
// 	std::cout << "\n- Information matrix determinant: " << arma::det(info_mat) << std::endl;
// 	std::cout << "\n- Average update norm: " << update_norm << std::endl;
// 	std::cout << "\n- Residuals: \n";
// 	std::cout << "--  Mean: " << arma::mean(residuals) << std::endl;
// 	std::cout << "--  Standard deviation: " << arma::stddev(residuals) << std::endl;


// 	if (update_norm < 1e-2){
// 		has_converged = true;
// 	}

// 	// The deviations are added to the coordinates
// 	auto control_points = this -> shape_model -> get_control_points();

// 	for (auto iter_points = control_points -> begin(); iter_points != control_points -> end(); ++iter_points){
// 		unsigned int global_point_index = this -> shape_model -> get_control_point_index(*iter_points);
// 		(*iter_points) -> set_coordinates((*iter_points) -> get_coordinates()
// 			+ dC.rows(3 * global_point_index, 3 * global_point_index + 2));
// 	}



// 	return info_mat;

// }

