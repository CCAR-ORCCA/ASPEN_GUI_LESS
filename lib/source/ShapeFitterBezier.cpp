#include "ShapeFitterBezier.hpp"
#include "boost/progress.hpp"


ShapeFitterBezier::ShapeFitterBezier(ShapeModelBezier * shape_model,PC * pc) : ShapeFitter(pc){
	
	this -> shape_model = shape_model;
}

bool ShapeFitterBezier::fit_shape_batch(unsigned int N_iter, double J,const arma::mat & DS, const arma::vec & X_DS){


	arma::sp_mat info_mat;
	std::vector<Footpoint> footpoints;

	for (unsigned int i = 0; i < N_iter; ++i){
		
		// The foodpoints are found
		footpoints = this -> find_footpoints();
		arma::mat Pbar_mat = arma::mat(3,footpoints.size());
		for (unsigned int k = 0; k < footpoints.size(); ++k){
			Pbar_mat.col(k) = footpoints[k].Pbar;
		}

		arma::vec u = {1,0,0};
		PC pc(u,Pbar_mat);
		pc.save("Pbar_" + std::to_string(i) + ".obj");


		// The shape is updated
		info_mat = this -> update_shape(footpoints);

	}

	// The information matrices of the patches that were seen are updated
	this -> update_patches_info_matrices(info_mat,footpoints);


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
		
		try{
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





arma::sp_mat ShapeFitterBezier::update_shape(std::vector<Footpoint> & footpoints){


	std::cout << "Updating shape from the " << footpoints.size()<<  " footpoints...\n";

	// The forward/backward look up tables are created
	std::map<unsigned int, std::shared_ptr<ControlPoint> > global_index_to_pointer;
	std::map<std::shared_ptr<ControlPoint> ,unsigned int> pointer_to_global_index;

	for (auto iter = footpoints.begin(); iter != footpoints.end(); ++iter){

		auto control_points = iter -> element -> get_control_points();
		for (auto control_point = control_points -> begin(); control_point != control_points -> end(); ++ control_point){
			if (pointer_to_global_index.find(*control_point) == pointer_to_global_index.end()){
				pointer_to_global_index[*control_point] = pointer_to_global_index.size();
				global_index_to_pointer[pointer_to_global_index.size()] = *control_point;
			}
		}
	}

	// The normal and information matrices are created
	unsigned int N = 3 * global_index_to_pointer.size();
	arma::sp_mat info_mat(N,N);
	arma::vec normal_mat = arma::zeros<arma::vec>(N);


	boost::progress_display progress(footpoints.size());

	// All the measurements are processed
	
	for (auto footpoint = footpoints.begin(); footpoint != footpoints.end(); ++footpoint){

		++ progress;

		arma::sp_mat Hi(1,N);
		Bezier * patch = dynamic_cast<Bezier *>(footpoint -> element);

		auto control_points = patch -> get_control_points();

		// The different control points for this patch have their contribution added
		for (auto iter_points = control_points -> begin(); iter_points != control_points -> end(); ++iter_points){

			unsigned int global_point_index = pointer_to_global_index[*iter_points];

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

		arma::sp_mat Li = Hi.t() * Hi;
		
		info_mat = info_mat + Li;

	}

	
	// The information matrix is regularized
	info_mat += 0.001 * arma::trace(info_mat) * arma::eye<arma::mat>(info_mat.n_rows,info_mat.n_cols);
	// arma::vec dC = arma::solve(eigvec * arma::diagmat(eigval) * eigvec.t(),normal_mat);

	arma::vec dC = arma::spsolve(info_mat,normal_mat);


	std::cout << "Update norm: " << arma::norm(dC) << std::endl;

	// The deviations are added to the coordinates
	for (auto iter_points = pointer_to_global_index.begin(); iter_points != pointer_to_global_index.end(); ++iter_points){
		unsigned int global_point_index = iter_points -> second;
		iter_points -> first -> set_coordinates(iter_points -> first -> get_coordinates()
			+ dC.rows(3 * global_point_index, 3 * global_point_index + 2));
	}



	return info_mat;

}

void ShapeFitterBezier::update_patches_info_matrices(arma::sp_mat & info_mat,std::vector<Footpoint> & footpoints){

	std::map<std::shared_ptr<ControlPoint> ,unsigned int> pointer_to_global_index;
	std::set<Bezier *> seen_patches;

	for (auto iter = footpoints.begin(); iter != footpoints.end(); ++iter){
		auto control_points = iter -> element -> get_control_points();
		for (auto control_point = control_points -> begin(); control_point != control_points -> end(); ++ control_point){
			if (pointer_to_global_index.find(*control_point) == pointer_to_global_index.end()){
				pointer_to_global_index[*control_point] = pointer_to_global_index.size();
			}
		}
		seen_patches.insert(dynamic_cast<Bezier * >(iter -> element));
	}

	// Each patch's information matrix is updated
	// Note that this formulation does not account for the 
	// neighbours to the patch as only the cross correlations of 
	// points within the same patch are tracked
	for (auto patch = seen_patches.begin(); patch != seen_patches.end(); ++patch){
		arma::mat * patch_info_mat = (*patch) -> get_info_mat();

		unsigned int points_in_patch = (*patch) -> get_control_points() -> size();

		for (unsigned int i = 0; i < points_in_patch; ++ i){
			unsigned int i_global = pointer_to_global_index[(*patch) -> get_control_points() -> at(i)];
			
			for (unsigned int j = 0; j <= i; ++ j){
				unsigned int j_global = pointer_to_global_index[(*patch) -> get_control_points() -> at(j)];
				
				patch_info_mat -> submat( 3 * i, 3 * j, 3 * i + 2, 3 * j + 2 ) += info_mat.submat(
					3 * i_global,3 * j_global, 3 * i_global + 2, 3 * j_global + 2 );
				
				if (i != j){
					patch_info_mat -> submat( 3 * j, 3 * i, 3 * j + 2, 3 * i + 2 ) += info_mat.submat(
						3 * j_global,3 * i_global, 3 * j_global + 2, 3 * i_global + 2 );
				}
			}
		}
	}

}
