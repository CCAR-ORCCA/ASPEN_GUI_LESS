#include "ShapeFitter.hpp"





ShapeFitter::ShapeFitter(ShapeModelTri * shape_model,PC * pc){
	this -> shape_model = shape_model;
	this -> pc = pc;

}

bool ShapeFitter::fit_shape_batch(double J,const arma::mat & DS, const arma::vec & X_DS){

	std::set<Element *> seen_facets;
	std::map<std::shared_ptr<ControlPoint> , unsigned int> seen_vertices;
	std::vector<std::pair<arma::vec,Footpoint > > measurement_pairs;


	// Each measurement is associated to the closest facet on the shape
	for (unsigned int i = 0; i < this -> pc -> get_size(); ++i){
		

		arma::vec Ptilde = DS * this -> pc -> get_point_coordinates(i) + X_DS;

		Footpoint footpoint;
		this -> find_footpoint(Ptilde,footpoint);
		if(std::abs(arma::dot(footpoint.n,Ptilde - footpoint.Pbar))< J){
			continue;
		}

		seen_facets.insert(footpoint.facet);
		measurement_pairs.push_back(std::make_pair(Ptilde,footpoint));

		if (seen_vertices.find(footpoint.facet -> get_control_points() -> at(0)) == seen_vertices.end()){
			seen_vertices[footpoint.facet -> get_control_points() -> at(0)] = seen_vertices.size();
		}

		if (seen_vertices.find(footpoint.facet -> get_control_points() -> at(1)) == seen_vertices.end()){
			seen_vertices[footpoint.facet -> get_control_points() -> at(1)] = seen_vertices.size();
		}

		if (seen_vertices.find(footpoint.facet -> get_control_points() -> at(2)) == seen_vertices.end()){
			seen_vertices[footpoint.facet -> get_control_points() -> at(2)] = seen_vertices.size();
		}

	}

	std::cout << "Seen vertices: " << seen_vertices.size() << std::endl;
	std::cout << "Measurements: " << measurement_pairs.size() << std::endl;
	std::cout << "Residuals: " << this -> compute_residuals(measurement_pairs) << " m" << std::endl;



	double W = 5;

	arma::sp_mat info_mat = W * arma::eye<arma::sp_mat>(3 * seen_vertices.size(), 3 * seen_vertices.size());
	arma::vec normal_mat = arma::zeros<arma::vec>( 3 * seen_vertices.size());
	

	arma::vec Pbar(3);
	arma::vec n(3);
	arma::vec C(3);
	double u,v;

	// arma::mat Pbar_mat(3,measurement_pairs.size());
	// unsigned int i = 0;

	for (auto iter = measurement_pairs.begin(); iter < measurement_pairs.end(); ++ iter){

    	// The Pbar - Ptilde  is not exactly colinear to the surface normal. It is thus corrected
		n =  iter -> second.n;

		Pbar = iter -> second.Pbar;

		u = iter -> second.u;
		v = iter -> second.v;


		unsigned int iu = seen_vertices[iter -> second .facet ->  get_control_points() -> at(0)];
		unsigned int iv = seen_vertices[iter -> second .facet ->  get_control_points() -> at(1)];
		unsigned int iw  = seen_vertices[iter -> second .facet ->  get_control_points() -> at(2)];

		

		
		// Pbar_mat.col(i) = Pbar;
		// ++i;

		arma::rowvec Hii = arma::zeros<arma::rowvec>( 9 );

		Hii.cols(0,2) = n.t() * u;
		Hii.cols(3,5) = n.t() * v;
		Hii.cols(6,8) = n.t() * (1 - u - v);


		arma::sp_mat Mi(9,3 * seen_vertices.size());

		Mi.submat(0,3 * iu,2,3 * iu + 2) = arma::eye<arma::mat>(3,3);
		Mi.submat(3,3 * iv,5,3 * iv + 2) = arma::eye<arma::mat>(3,3);
		Mi.submat(6,3 * iw,8,3 * iw + 2) = arma::eye<arma::mat>(3,3);

		arma::sp_mat Hi(1,3 * seen_vertices.size());

		Hi = Hii * Mi;
		info_mat += Hi.t() * Hi;
		normal_mat += Hi.t() * arma::dot(n, iter -> first - Pbar);

	}

	
	// The new control point coordinates are found
	arma::vec X = arma::spsolve(info_mat,normal_mat);

	std::cout << "Update norm (average): " << arma::norm(X / seen_vertices.size()) << std::endl;

	for (auto iter = seen_vertices.begin(); iter != seen_vertices.end(); ++iter){
		unsigned int i = seen_vertices[iter -> first];
		(iter -> first -> get_coordinates()) = (iter -> first -> get_coordinates()) + X.rows(3 *i, 3 * i + 2);
	}

	this -> shape_model -> update_facets();
	this -> shape_model -> update_mass_properties();

	return false;

}





void ShapeFitter::find_footpoint(const arma::vec & Ptilde, 
	Footpoint & footpoint){

	double distance = std::numeric_limits<double>::infinity();
	double u,v;
	std::shared_ptr<ControlPoint> closest_point;

	this -> shape_model -> get_KDTree_control_points() -> closest_point_search(
		Ptilde,
		this -> shape_model -> get_KDTree_control_points(),
		closest_point,
		distance);

	std::set< Element *  > owning_elements = closest_point -> get_owning_elements();

	Element * closest_facet =  * owning_elements.begin();
	double max_distance = std::abs(arma::dot(closest_facet -> get_normal(),
		Ptilde - closest_point -> get_coordinates()));

	// The closest vertex belongs to a number of facets. the one yielding the smallest
	// projection distance is chosen

	for (auto iter = owning_elements.begin();iter != owning_elements.end(); ++iter){

		arma::vec n = (*iter) -> get_normal();
		double distance = std::abs(arma::dot(n,
			Ptilde - closest_point -> get_coordinates()));

		double u,v;
		arma::vec Ptilde_in_facet = Ptilde - arma::dot(n,
			Ptilde - closest_point -> get_coordinates()) * (n);

		this -> get_barycentric_coordinates(Ptilde_in_facet , u,v,dynamic_cast<Facet*>(*iter));
		if (u < 0.99 && v < 0.99 && u > 0.01 && v > 0.01 && u + v > 0.01 && u + v < 0.99){
			closest_facet = *iter;
			break;
		}

		if (distance > max_distance){
			closest_facet = *iter;
			max_distance = distance;
		}
		
	}

	


	footpoint.n = closest_facet -> get_normal();
	footpoint.Pbar = (Ptilde 
		- arma::dot(footpoint.n,Ptilde - closest_facet -> get_center()) * (footpoint.n));


	this -> get_barycentric_coordinates(footpoint.Pbar, u,v,dynamic_cast<Facet *>(closest_facet));
	footpoint.u = u;
	footpoint.v = v;
	footpoint.facet = closest_facet;

}









void ShapeFitter::get_barycentric_coordinates(const arma::vec & Pbar,double & u, double & v, Facet * facet){


	// The barycentric coordinates of Pbar are found
	arma::vec C0 = facet -> get_control_points() -> at(0) -> get_coordinates();
	arma::vec C1 = facet -> get_control_points() -> at(1) -> get_coordinates();
	arma::vec C2 = facet -> get_control_points() -> at(2) -> get_coordinates();

	arma::mat A(3,2);
	arma::vec B(2);

	A.col(0) = C0 - C2;
	A.col(1) = C1 - C2;

	B(0) = arma::dot(Pbar - C2, A.col(0));
	B(1) = arma::dot(Pbar - C2, A.col(1));

	arma::vec X = arma::solve(A.t() * A, B);

	u = X(0);
	v = X(1);


}

double ShapeFitter::compute_residuals(std::vector<std::pair<arma::vec,Footpoint > > & measurement_pairs) const{

	double res = 0;

	arma::vec C(3);
	arma::vec n(3);

	for (auto iter = measurement_pairs.begin(); iter != measurement_pairs.end(); ++iter){


		C =  iter -> second.facet -> get_center();
		n =  iter -> second.facet -> get_normal();

		res += std::pow(arma::dot(C - iter -> first,n),2);

	}

	return std::sqrt(res / measurement_pairs.size());


}



void ShapeFitter::save(std::string path, arma::mat & Pbar_mat) const {


	std::ofstream shape_file;
	shape_file.open(path);

	

	for (unsigned int vertex_index = 0;
		vertex_index < Pbar_mat.n_cols;
		++vertex_index) {

		arma::vec p = Pbar_mat.col(vertex_index);

	shape_file << "v " << p(0) << " " << p(1) << " " << p(2) << std::endl;
	
}


}







