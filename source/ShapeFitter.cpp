#include "ShapeFitter.hpp"


ShapeFitter::ShapeFitter(ShapeModelTri * shape_model,PC * pc){
	this -> shape_model = shape_model;
	this -> pc = pc;

}

void ShapeFitter::fit_shape(){

	std::set<Facet *> seen_facets;
	std::set<Facet *> under_observed_facets;
	std::map<std::shared_ptr<ControlPoint> , unsigned int> seen_vertices;
	std::vector<std::pair<arma::vec,Facet * > > measurement_pairs;

	// Each measurement is associated to the closest facet on the shape
	for (unsigned int i = 0; i < this -> pc -> get_size(); ++i){
		
		double distance = std::numeric_limits<double>::infinity();
		std::shared_ptr<Element> closest_point;

		arma::vec Ptilde = this -> pc -> get_point_coordinates(i);

		this -> shape_model -> get_kdtree_facet() -> closest_point_search(
			Ptilde,
			this -> shape_model -> get_kdtree_facet(),
			closest_point,
			distance);
		Facet * facet = dynamic_cast<Facet *>(closest_point.get());
		
		seen_facets.insert(facet);
		measurement_pairs.push_back(std::make_pair(Ptilde,facet));

		if (seen_vertices.find(facet -> get_control_points() -> at(0)) == seen_vertices.end()){
			seen_vertices[facet -> get_control_points() -> at(0)] = seen_vertices.size();
		}
		
		if (seen_vertices.find(facet -> get_control_points() -> at(1)) == seen_vertices.end()){
			seen_vertices[facet -> get_control_points() -> at(1)] = seen_vertices.size();
		}

		if (seen_vertices.find(facet -> get_control_points() -> at(2)) == seen_vertices.end()){
			seen_vertices[facet -> get_control_points() -> at(2)] = seen_vertices.size();
		}


	}

	// arma::sp_mat info_mat( 3 * seen_vertices.size(), 3 * seen_vertices.size());
	arma::mat info_mat( 3 * seen_vertices.size(), 3 * seen_vertices.size());
	arma::vec normal_mat = arma::zeros<arma::vec>( 3 * seen_vertices.size());
	arma::mat Pbar_mat(3,measurement_pairs.size());
	
	info_mat.fill(0);
	normal_mat.fill(0);

	// The shape is pre-constrained so as to prevent 
	// a singular information matrix

	for (auto iter = seen_facets.begin(); iter != seen_facets.end(); ++ iter){
		this -> freeze_facet(*iter, 
		seen_vertices,
		info_mat,
		normal_mat);
	}
	
	for (auto iter = measurement_pairs.begin(); iter < measurement_pairs.end(); ++ iter){

		arma::vec C = * iter -> second -> get_facet_center();
		arma::vec n =  * iter -> second -> get_facet_normal();

    	// The Pbar - Ptilde  is not exactly colinear to the surface normal. It is thus corrected
		arma::vec Pbar = iter -> first - arma::dot(n,iter -> first - C) * n;


    	// The barycentric coordinates of Pbar are found
		arma::vec C0 = * iter -> second -> get_control_points() -> at(0) -> get_coordinates();
		arma::vec C1 = * iter -> second -> get_control_points() -> at(1) -> get_coordinates();
		arma::vec C2 = * iter -> second -> get_control_points() -> at(2) -> get_coordinates();

		arma::mat A(3,2);
		arma::vec B(2);
		
		Pbar_mat.col(std::distance(measurement_pairs.begin(),iter)) = Pbar;

		A.col(0) = C0 - C2;
		A.col(1) = C1 - C2;

		B(0) = arma::dot(Pbar - C2, A.col(0));
		B(1) = arma::dot(Pbar - C2, A.col(1));

		arma::vec X = arma::solve(A.t() * A, B);

		double u = X(0);
		double v = X(1);

		unsigned int iu = seen_vertices[iter -> second -> get_control_points() -> at(0)];
		unsigned int iv = seen_vertices[iter -> second -> get_control_points() -> at(1)];
		unsigned int iw = seen_vertices[iter -> second -> get_control_points() -> at(2)];

		arma::rowvec Hii = arma::zeros<arma::rowvec>( 9 );

		Hii.cols(0,2) = n.t() * u;
		Hii.cols(3,5) = n.t() * v;
		Hii.cols(6,8) = n.t() * (1 - u - v);
		
		arma::sp_mat Hi(1,3 * seen_vertices.size());

		arma::sp_mat Mi(9,3 * seen_vertices.size());

		Mi.submat(0,3 * iu,2,3 * iu + 2) = arma::eye<arma::mat>(3,3);
		Mi.submat(3,3 * iv,5,3 * iv + 2) = arma::eye<arma::mat>(3,3);
		Mi.submat(6,3 * iw,8,3 * iw + 2) = arma::eye<arma::mat>(3,3);

		Hi = Hii * Mi;
		info_mat += Hi.t() * Hi;
		normal_mat += Hi.t() * arma::dot(n, iter -> first);

	}

	arma::vec los = {1,0,0};
	// PC Pbar_pc(los,Pbar_mat);
	// Pbar_pc.save("../output/pc/Pbar_pc.obj");


	arma::vec X = arma::solve(info_mat,normal_mat);

	arma::mat X_mat(3,int(X.n_rows / 3) );

	for (int i = 0; i < int(X.n_rows / 3); ++i){
		X_mat.col(i) = X.rows(3 * i, 3 * i + 2);
	}

	PC X_pc(los,X_mat);
	X_pc.save("../output/pc/X_pc.obj");


	throw;



}

void ShapeFitter::freeze_facet(Facet * facet, 
	std::map<std::shared_ptr<ControlPoint> , unsigned int> & seen_vertices,
	arma::mat & info_mat,arma::vec & normal_mat){

	arma::vec n = * facet -> get_facet_normal();
	double u,v;

    // The barycentric coordinates of Pbar are found
	arma::vec C0 = * facet -> get_control_points() -> at(0) -> get_coordinates();
	arma::vec C1 = * facet -> get_control_points() -> at(1) -> get_coordinates();
	arma::vec C2 = * facet -> get_control_points() -> at(2) -> get_coordinates();

	unsigned int iu = seen_vertices[facet -> get_control_points() -> at(0)];
	unsigned int iv = seen_vertices[facet -> get_control_points() -> at(1)];
	unsigned int iw = seen_vertices[facet -> get_control_points() -> at(2)];
	
	arma::vec n0 = arma::cross(n,arma::normalise(C0 - C1));
	arma::vec n1 = arma::cross(n,arma::normalise(C1 - C2));
	arma::vec n2 = arma::cross(n,arma::normalise(C2 - C0));

	arma::sp_mat Mi(9,3 * seen_vertices.size());

	Mi.submat(0,3 * iu,2,3 * iu + 2) = arma::eye<arma::mat>(3,3);
	Mi.submat(3,3 * iv,5,3 * iv + 2) = arma::eye<arma::mat>(3,3);
	Mi.submat(6,3 * iw,8,3 * iw + 2) = arma::eye<arma::mat>(3,3);

	// First edge e0 : C0 -- C1
	u = 0.4;
	v = 0.6;
	this -> apply_constraint_to_facet(C0,C1,C2,n0,u,v,
		info_mat,
		normal_mat,
		Mi);

	u = 0.6;
	v = 0.4;
	this -> apply_constraint_to_facet(C0,C1,C2,n0,u,v,
		info_mat,
		normal_mat,
		Mi);

	// Second edge e1 : C1 -- C2
	u = 0.;
	v = 0.6;
	this -> apply_constraint_to_facet(C0,C1,C2,n1,u,v,
		info_mat,
		normal_mat,
		Mi);

	u = 0.;
	v = 0.4;
	this -> apply_constraint_to_facet(C0,C1,C2,n1,u,v,
		info_mat,
		normal_mat,
		Mi);

	// Third edge e1 : C2 -- C0
	u = 0.4;
	v = 0.;
	this -> apply_constraint_to_facet(C0,C1,C2,n2,u,v,
		info_mat,
		normal_mat,
		Mi);

	u = 0.6;
	v = 0.;
	this -> apply_constraint_to_facet(C0,C1,C2,n2,u,v,
		info_mat,
		normal_mat,
		Mi);

	// Facet : 

	u = 0.6;
	v = 0.2;
	this -> apply_constraint_to_facet(C0,C1,C2,n,u,v,
		info_mat,
		normal_mat,
		Mi);

	u = 0.2;
	v = 0.6;
	this -> apply_constraint_to_facet(C0,C1,C2,n,u,v,
		info_mat,
		normal_mat,
		Mi);

	u = 0.2;
	v = 0.2;
	this -> apply_constraint_to_facet(C0,C1,C2,n,u,v,
		info_mat,
		normal_mat,
		Mi);

}

void ShapeFitter::apply_constraint_to_facet(const arma::vec & C0, 
	const arma::vec & C1,
	const arma::vec & C2,
	const arma::vec & n, 
	double u, 
	double v,
	arma::mat & info_mat,
	arma::vec & normal_mat,
	const arma::sp_mat & Mi){

	arma::rowvec Hii = arma::zeros<arma::rowvec>( 9 );

	arma::vec Pbar = u * C0 + v * C1 + ( 1 - u - v) * C2;

	Hii.cols(0,2) = n.t() * u;
	Hii.cols(3,5) = n.t() * v;
	Hii.cols(6,8) = n.t() * (1 - u - v);

	arma::sp_mat Hi(1,info_mat.n_cols);

	Hi = Hii * Mi;
	info_mat += Hi.t() * Hi;
	normal_mat += Hi.t() * arma::dot(n, Pbar);

}







