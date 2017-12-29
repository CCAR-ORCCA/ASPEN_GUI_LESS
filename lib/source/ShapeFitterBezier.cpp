#include "ShapeFitterBezier.hpp"
#include "boost/progress.hpp"


ShapeFitterBezier::ShapeFitterBezier(ShapeModelBezier * shape_model,PC * pc) : ShapeFitter(pc){
	
	this -> shape_model = shape_model;
}

bool ShapeFitterBezier::fit_shape_batch(
	unsigned int N_iter, 
	double J,
	const arma::mat & DS, 
	const arma::vec & X_DS){


	arma::sp_mat info_mat;
	std::vector<Footpoint> footpoints;
	bool has_converged  = false;

	for (unsigned int i = 0; i < N_iter; ++i){
		
		// The foodpoints are found
		footpoints = this -> find_footpoints();
		
		// The shape is updated
		info_mat = this -> update_shape(footpoints,has_converged);
		if (has_converged){
			break;
		}

	}

	// The information matrix is updated
	*this -> shape_model -> get_info_mat_ptr() = info_mat;
	this -> shape_model -> get_dX_bar_ptr() -> fill(0);

	return false;

}

std::vector<Footpoint> ShapeFitterBezier::find_footpoints() const{

	std::vector<Footpoint> footpoints;

	std::cout << "Finding footpoints...\n";

	boost::progress_display progress(this -> pc -> get_size());


	for (unsigned int i = 0; i < this -> pc -> get_size(); ++i){
		Footpoint footpoint;
		footpoint.element = NULL;

		++progress;

		footpoint.Ptilde = this -> pc -> get_point_coordinates(i);
		
		try {
			this -> find_footpoint(footpoint);
			footpoints.push_back(footpoint);
		}

		catch(const MissingFootpointException & e){
			// std::cout << e.what() << std::endl;

		}

	}

	return footpoints;
}

void ShapeFitterBezier::find_footpoint(Footpoint & footpoint) const {

	if (this -> shape_model -> get_NElements() > 1){

		// The closest control to this measurement is found
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
				break;
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
		if (error < 1e-8){

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

arma::sp_mat ShapeFitterBezier::update_shape(std::vector<Footpoint> & footpoints,
	bool & has_converged){

	std::cout << "Updating shape from the " << footpoints.size()<<  " footpoints...\n";

	// Check if the info matrix has been initialized
	if (this -> shape_model -> get_info_mat_ptr() == nullptr){
		this -> shape_model -> initialize_info_mat();
		this -> shape_model -> initialize_dX_bar();
		this -> shape_model -> initialize_index_table();
	}

	// The normal and information matrices are created
	arma::sp_mat info_mat(*this -> shape_model -> get_info_mat_ptr());
	arma::vec normal_mat = info_mat * (*this -> shape_model -> get_dX_bar_ptr());
	unsigned int N = info_mat.n_cols;

	boost::progress_display progress(footpoints.size());

	// All the measurements are processed
	
	for (auto footpoint = footpoints.begin(); footpoint != footpoints.end(); ++footpoint){

		++ progress;

		arma::sp_mat Hi(1,N);
		Bezier * patch = dynamic_cast<Bezier *>(footpoint -> element);

		auto control_points = patch -> get_control_points();

		// The different control points for this patch have their contribution added
		for (auto iter_points = control_points -> begin(); iter_points != control_points -> end(); ++iter_points){

			unsigned int global_point_index = this -> shape_model -> get_control_point_index(*iter_points);

			auto local_indices = patch -> get_local_indices(*iter_points);

			unsigned int i = std::get<0>(local_indices);
			unsigned int j = std::get<1>(local_indices);
			unsigned int degree = patch -> get_degree();

			double bezier = Bezier::bernstein(footpoint -> u,footpoint -> v,i,j,degree);
			
			Hi.cols(3 * global_point_index, 3 * global_point_index + 2) = bezier * footpoint -> n.t();
		}
		
		double y = (arma::dot(footpoint -> n,footpoint -> Ptilde)
			- arma::dot(footpoint -> n,patch -> evaluate(footpoint -> u,footpoint -> v)));

		normal_mat += Hi.t() * y;
		
		info_mat +=  Hi.t() * Hi;

	}
	
	// The information matrix is regularized
	info_mat += 0.001 * arma::trace(info_mat) * arma::eye<arma::mat>(info_mat.n_rows,info_mat.n_cols);
	
	// The deviation is computed
	arma::vec dC = arma::spsolve(info_mat,normal_mat);

	// The a-priori deviation is adjusted
	*this -> shape_model -> get_dX_bar_ptr() = *this -> shape_model -> get_dX_bar_ptr() - dC;

	double update_norm = 0;
	unsigned int size = int(N / 3.);

	for (unsigned int k = 0; k < size; ++k){
		update_norm += arma::norm(dC.subvec(3 * k, 3 * k + 2))/size;

	}

	std::cout << "Average update norm: " << update_norm << std::endl;
	
	if (update_norm < 1e-1){
		has_converged = true;
	}

	// The deviations are added to the coordinates
	auto control_points = this -> shape_model -> get_control_points();
	
	for (auto iter_points = control_points -> begin(); iter_points != control_points -> end(); ++iter_points){
		unsigned int global_point_index = this -> shape_model -> get_control_point_index(*iter_points);
		(*iter_points) -> set_coordinates((*iter_points) -> get_coordinates()
			+ dC.rows(3 * global_point_index, 3 * global_point_index + 2));
	}



	return info_mat;

}

