#include "ShapeFitterBezier.hpp"
#include "boost/progress.hpp"


ShapeFitterBezier::ShapeFitterBezier(ShapeModelBezier * shape_model,PC * pc) : ShapeFitter(pc){
	
	this -> shape_model = shape_model;
}


bool ShapeFitterBezier::fit_shape_KF(
	unsigned int index,
	unsigned int N_iter_outer, 
	double J,
	const arma::mat & DS, 
	const arma::vec & X_DS,
	double los_noise_sd_base,
	const arma::vec & u_dir){


	double W = 1./(los_noise_sd_base * los_noise_sd_base);

	
	// fit_elements has the pointers to the elements to be fit
	for (unsigned int j = 0; j < N_iter_outer; ++j){

		std::cout << "\n\n- Outer iteration : " << j + 1 << "/" << N_iter_outer <<std::endl;

		// The footpoints are first found
		std::vector<Footpoint> footpoints = this -> find_footpoints();

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
			return false;
		}

		arma::mat Pbar_mat = arma::mat(3,footpoints.size());
		arma::mat Ptilde_mat = arma::mat(3,footpoints.size());		 		

		for (unsigned int k = 0; k < footpoints.size(); ++k){		
			Pbar_mat.col(k) = footpoints[k].Pbar;
			Ptilde_mat.col(k) = footpoints[k].Ptilde;		

		}		

		arma::vec u = {1,0,0};		
		PC pc(u,Pbar_mat);	
		PC pc_tilde(u,Ptilde_mat);		

		pc.save("../output/pc/Pbar_" + std::to_string(j) + "_"+std::to_string(index) + ".obj");
		pc_tilde.save("../output/pc/Ptilde_" + std::to_string(j) + "_"+std::to_string(index) + ".obj");


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
			

		}


	}




	return false;

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
			Hi.cols(3 * point_index, 3 * point_index + 2) -= (footpoint.Ptilde - footpoint.Pbar).t() * patch -> partial_n_partial_C(footpoint . u,
				footpoint . v,
				i, j,
				degree);


		}


		double y = arma::dot(footpoint . n,footpoint . Ptilde
			- patch -> evaluate(footpoint . u,footpoint . v));

		residuals(k) = y;


		// The orthogonal of the ray direction is constructed
		// its exact orientation does not matter if one assumes that
		// the transverse noises have equal standard deviations
		// and are uncorrelated

		arma::vec rand = arma::randu<arma::vec>(3);
		arma::vec v_dir = arma::normalise(arma::cross(u_dir,rand));
		arma::vec w_dir = arma::cross(u_dir,v_dir);

		// double R_augmented = R * ( std::pow(arma::dot(footpoint . n,u_dir),2) 
		// 	+ 1e-3 *  (std::pow(arma::dot(footpoint . n,v_dir),2) + std::pow(arma::dot(footpoint . n,w_dir),2)));

		double R_augmented = R 
		double W_augmented = 1./R_augmented;

		normal_mat += Hi.t() * W_augmented * y;
		info_mat +=  Hi.t() * W_augmented * Hi;

	}


	arma::mat regularized_info_mat;

	if (arma::det(info_mat) < std::numeric_limits<double>::infinity()){
		regularized_info_mat = info_mat + 1e-2 * arma::trace(info_mat) * arma::eye<arma::mat>(info_mat.n_cols,info_mat.n_cols);
	}
	else{
		regularized_info_mat = info_mat;
	}


	// The deviation is computed
	arma::vec dC = arma::solve(regularized_info_mat,normal_mat);

	// Only the normal component is kept
	for (unsigned int k = 0; k < control_points -> size(); ++k){

		std::shared_ptr<ControlPoint> point = control_points -> at(k);

		auto local_indices = patch -> get_local_indices(point);

		unsigned int i = std::get<0>(local_indices);
		unsigned int j = std::get<1>(local_indices);
		unsigned int degree = patch -> get_degree();

		arma::vec n = patch -> get_normal(double(i) / degree,double(j) / degree);

		dC.rows(3 * k, 3 * k+ 2) = arma::dot(n,dC.rows(3 * k, 3 * k+ 2)) * n;
	}

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

		// point -> set_coordinates(point -> get_coordinates()
		// 	+ arma::dot(n,dC.rows(3 * k, 3 * k+ 2)) * n);

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




