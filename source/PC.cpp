#include "PC.hpp"

PC::PC(std::vector<std::shared_ptr<Ray> > * focal_plane,
	FrameGraph * frame_graph) {

	std::vector< std::shared_ptr<PointNormal> > points_normals;

	// The valid measurements used to form the point cloud are extracted
	for (unsigned int pixel = 0; pixel < focal_plane -> size(); ++pixel) {

		if (focal_plane -> at(pixel) -> get_true_range() < std::numeric_limits<double>::infinity()) {

			arma::vec impact_point_instrument = focal_plane -> at(pixel) -> get_impact_point();

			points_normals.push_back(std::make_shared<PointNormal>(PointNormal(impact_point_instrument)));
			
		}
	}

	arma::vec los = {1,0,0};

	this -> construct_kd_tree(points_normals);
	this -> construct_normals(los);

}



PC::PC(arma::vec los_dir, arma::mat & points) {

	std::vector< std::shared_ptr<PointNormal> > points_normals;

	for (unsigned int index = 0; index < points . n_cols; ++index) {

		points_normals.push_back(std::make_shared<PointNormal>(PointNormal(points . col(index))));
	}
	this -> construct_kd_tree(points_normals);
	this -> construct_normals(los_dir);

}



PC::PC(ShapeModelTri * shape_model) {

	std::vector< std::shared_ptr<PointNormal> > points_normals;


	// The shape model is used to create the point cloud
	// The center points to each facets are used
	// The normals of each facet are directly used
	for (unsigned int facet_index = 0; facet_index < shape_model -> get_NElements(); ++facet_index) {
		Facet * facet = dynamic_cast<Facet *>(shape_model -> get_elements() -> at(facet_index).get());

		arma::vec C = *facet -> get_facet_center();

		arma::vec C0 = *facet -> get_control_points() -> at(0) -> get_coordinates();
		arma::vec C1 = *facet -> get_control_points() -> at(1) -> get_coordinates();
		arma::vec C2 = *facet -> get_control_points() -> at(2) -> get_coordinates();

		points_normals.push_back(std::make_shared<PointNormal>(PointNormal(C, *facet -> get_facet_normal())));
		points_normals.push_back(std::make_shared<PointNormal>(PointNormal( 0.5 * (C0 + C), *facet -> get_facet_normal())));
		points_normals.push_back(std::make_shared<PointNormal>(PointNormal( 0.5 * (C1 + C), *facet -> get_facet_normal())));
		points_normals.push_back(std::make_shared<PointNormal>(PointNormal( 0.5 * (C2 + C), *facet -> get_facet_normal())));


	}

	for (unsigned int vertex_index = 0; vertex_index < shape_model -> get_NControlPoints(); ++vertex_index) {
		
		std::shared_ptr<ControlPoint> control_point = shape_model -> get_control_points() -> at(vertex_index);

		arma::vec n = {0,0,0};

		std::set< Element *  >  owning_elements = control_point -> get_owning_elements();
		
		for (auto iter = owning_elements.begin() ; iter != owning_elements.end(); ++iter){
			n +=  * (dynamic_cast<Facet *>(*iter)) -> get_facet_normal();
		}

		n = arma::normalise(n);

		points_normals.push_back(std::make_shared<PointNormal>(PointNormal(* control_point -> get_coordinates(), n)));
		


	}






	this -> construct_kd_tree(points_normals);

}


PC::PC(arma::mat & dcm,
	arma::vec & x,
	std::shared_ptr<PC> destination_pc,
	std::shared_ptr<PC> source_pc,
	FrameGraph * frame_graph) {


	std::vector<arma::vec> points;
	std::vector< std::shared_ptr<PointNormal> > points_normals;


	// The inclusion counter of each source is decremented
	// If this counter is greater or equal to zero, this point
	// can be reused
	// Otherwise it is discarded. This prevents the point cloud from growing too much
	for (unsigned int i = 0; i < source_pc -> get_size(); ++i) {

		source_pc -> get_point(i) -> decrement_inclusion_counter();
		if (source_pc -> get_point(i) -> get_inclusion_counter() >= 0) {
			points.push_back(dcm * source_pc -> get_point_coordinates(i) + x);

			points_normals.push_back(std::make_shared<PointNormal>(PointNormal(dcm * source_pc -> get_point_coordinates(i) + x, source_pc -> get_point(i) -> get_inclusion_counter())));

		}

	}

	// all the destination points are used
	for (unsigned int i = 0; i < destination_pc -> get_size(); ++i) {
		points.push_back(destination_pc -> get_point_coordinates(i));

		points_normals.push_back(std::make_shared<PointNormal>(PointNormal(destination_pc -> get_point_coordinates(i))));


	}


	arma::vec u = {1, 0, 0};
	arma::vec converted_los = frame_graph -> convert(u, "L", "N", true);

	this -> construct_kd_tree(points_normals);
	this -> construct_normals(converted_los);




}




unsigned int PC::get_size() const {
	return this -> kd_tree -> points_normals.size();
}

arma::vec PC::get_point_coordinates(unsigned int index) const {
	return *this -> kd_tree -> points_normals[index] -> get_point();
}

std::shared_ptr<PointNormal> PC::get_point(unsigned int index) const {
	return this -> kd_tree -> points_normals[index];
}



arma::vec PC::get_point_normal(unsigned int index) const {
	return *this -> kd_tree -> points_normals[index] -> get_normal();
}







void PC::construct_kd_tree(std::vector< std::shared_ptr<PointNormal> > & points_normals) {

	// The KD Tree is now constructed

	this -> kd_tree = std::make_shared<KDTree_pc>(KDTree_pc());
	this -> kd_tree = this -> kd_tree -> build(points_normals, 0, false);
}

std::shared_ptr<PointNormal> PC::get_closest_point(arma::vec & test_point) const {

	double distance = std::numeric_limits<double>::infinity();
	std::shared_ptr<PointNormal> closest_point;

	this -> kd_tree -> closest_point_search(test_point,
		this -> kd_tree,
		closest_point,
		distance);

	return closest_point;

}

std::vector<std::shared_ptr<PointNormal> > PC::get_closest_N_points(
	arma::vec & test_point, unsigned int N) const {

	std::vector<std::shared_ptr<PointNormal> > closest_points;

	for (unsigned int i = 0; i < N; ++i) {

		double distance = std::numeric_limits<double>::infinity();
		std::shared_ptr<PointNormal> closest_point;

		this -> kd_tree -> closest_point_search(test_point,
			this -> kd_tree,
			closest_point,
			distance,
			closest_points);

		closest_points.push_back(closest_point);
	}

	return closest_points;

}

// void PC::save(std::string path, bool format_like_obj) const {

// 	std::ofstream shape_file;
// 	shape_file.open(path);

// 	for (unsigned int vertex_index = 0;
// 	        vertex_index < this -> get_size();
// 	        ++vertex_index) {

// 		arma::vec p = this -> get_point_coordinates(vertex_index);
// 		if (format_like_obj) {
// 			shape_file << "v " << p(0) << " " << p(1) << " " << p(2) << std::endl;
// 		}
// 		else {
// 			shape_file << p(0) << " " << p(1) << " " << p(2) << std::endl;

// 		}
// 	}

// }


void  PC::save(std::string path, arma::mat dcm, arma::vec x, bool save_normals,
	bool format_like_obj) const {


	std::ofstream shape_file;
	shape_file.open(path);

	if (save_normals == format_like_obj && save_normals == true) {
		throw (std::runtime_error("save can't be called with those arguments!"));
	}

	for (unsigned int vertex_index = 0;
		vertex_index < this -> get_size();
		++vertex_index) {

		arma::vec p = dcm * this -> get_point_coordinates(vertex_index) + x;

	if (format_like_obj) {
		shape_file << "v " << p(0) << " " << p(1) << " " << p(2) << std::endl;
	}
	else if (save_normals) {

		arma::vec n = dcm * this -> get_point_normal(vertex_index);

		shape_file << p(0) << " " << p(1) << " " << p(2) << " " << n(0) << " " << n(1) << " " << n(2) << std::endl;
	}
	else {
		shape_file << p(0) << " " << p(1) << " " << p(2) << std::endl;

	}
}


}




std::shared_ptr<PointNormal> PC::get_closest_point_index_brute_force(arma::vec & test_point) const {

	double distance = std::numeric_limits<double>::infinity();

	unsigned int pc_size = this -> kd_tree -> get_size();
	int index_closest = 0;

	for (unsigned int i = 0 ; i < pc_size ; ++i) {
		double new_distance = arma::norm(test_point - *this -> kd_tree ->
			get_points_normals() -> at(i) -> get_point());

		if (new_distance < distance) {
			index_closest = i;
			distance = new_distance;
		}

	}

	return this -> kd_tree -> get_points_normals() -> at(index_closest);
}

std::vector<std::shared_ptr<PointNormal> > PC::get_closest_N_points_brute_force(arma::vec & test_point, unsigned int N) const {

	unsigned int pc_size = this -> kd_tree -> get_size();

	if (N > pc_size) {
		throw "Point cloud has fewer points than requested neighborhood size";
	}

	if (pc_size == 0) {
		throw "Empty point cloud";
	}

	std::map<double, unsigned int> distance_map;
	distance_map[std::numeric_limits<double>::infinity()] = 0;

	for (unsigned int i = 0 ; i < pc_size ; ++i) {

		double new_distance = arma::norm(test_point - *this -> kd_tree ->
			get_points_normals() -> at(i) -> get_point());

		if (new_distance > std::prev(distance_map.end()) -> first) {
			continue;
		}

		else  {

			distance_map[new_distance] = i;

			if (distance_map.size() > N) {
				distance_map.erase(std::prev(distance_map.end()));
			}

		}

	}

	std::vector<std::shared_ptr<PointNormal> > closest_points;

	for (auto it = distance_map.begin(); it != distance_map.end(); ++it ) {
		closest_points.push_back(this -> kd_tree -> get_points_normals() -> at(it -> second));
	}

	return closest_points;
}


void PC::construct_normals(arma::vec & los_dir) {

	#pragma omp parallel for if (USE_OMP_PC)
	for (unsigned int i = 0; i < this -> kd_tree -> get_points_normals() -> size(); ++i) {

		std::shared_ptr<PointNormal> pn = this -> kd_tree -> get_points_normals() -> at(i);

		// Get the N nearest neighbors to this point
		unsigned int N = 5;
		std::vector< std::shared_ptr<PointNormal > > closest_points = this -> get_closest_N_points(*pn -> get_point(), N);

		// This N nearest neighbors are used to get the normal
		arma::mat points_augmented(N, 4);
		points_augmented.col(3) = arma::ones<arma::vec>(N);

		for (unsigned int j = 0; j < N; ++j) {
			points_augmented.row(j).cols(0, 2) = closest_points[j] -> get_point() -> t();
		}

		// The eigenvalue problem is solved
		arma::vec eigval;
		arma::mat eigvec;

		arma::eig_sym(eigval, eigvec, points_augmented.t() * points_augmented);
		arma::vec n = arma::normalise(eigvec.col(arma::abs(eigval).index_min()).rows(0, 2));

		// The normal is flipped to make sure it is facing the los
		if (arma::dot(n, los_dir) < 0) {
			this -> kd_tree -> get_points_normals() -> at(i) -> set_normal(n);
		}
		else {
			this -> kd_tree -> get_points_normals() -> at(i) -> set_normal(-n);
		}

	}

}

arma::vec PC::get_center() const{

	double c_x = 0;
	double c_y = 0;
	double c_z = 0;

	unsigned int size = this -> kd_tree -> get_points_normals() -> size();

	#pragma omp parallel for reduction(+:c_x,c_y,c_z) if (USE_OMP_PC)
	for (unsigned int i = 0; i < size; ++i) {

		c_x += this -> kd_tree -> get_points_normals() -> at(i) -> get_point() -> at(0) / size;
		c_y += this -> kd_tree -> get_points_normals() -> at(i) -> get_point() -> at(1) / size;
		c_z += this -> kd_tree -> get_points_normals() -> at(i) -> get_point() -> at(2) / size;

	}
	arma::vec C = {c_x,c_y,c_z};


	return C;
}


arma::vec PC::get_bbox_center() const{

	double bbox_x_min = std::numeric_limits<double>::infinity();
	double bbox_y_min = std::numeric_limits<double>::infinity();
	double bbox_z_min = std::numeric_limits<double>::infinity();

	double bbox_x_max = - bbox_x_min;
	double bbox_y_max = - bbox_y_min;
	double bbox_z_max = - bbox_z_min;

	unsigned int size = this -> kd_tree -> get_points_normals() -> size();

	#pragma omp parallel for reduction(min:bbox_x_min,bbox_y_min,bbox_z_min) reduction(max:bbox_x_max,bbox_y_max,bbox_z_max) if (USE_OMP_PC)
	for (unsigned int i = 0; i < size; ++i) {

		bbox_x_max = std::max(bbox_x_max,this -> kd_tree -> get_points_normals() -> at(i) -> get_point() -> at(0));
		bbox_y_max = std::max(bbox_y_max,this -> kd_tree -> get_points_normals() -> at(i) -> get_point() -> at(1));
		bbox_z_max = std::max(bbox_z_max,this -> kd_tree -> get_points_normals() -> at(i) -> get_point() -> at(2));

		bbox_x_min = std::min(bbox_x_min,this -> kd_tree -> get_points_normals() -> at(i) -> get_point() -> at(0));
		bbox_y_min = std::min(bbox_y_min,this -> kd_tree -> get_points_normals() -> at(i) -> get_point() -> at(1));
		bbox_z_min = std::min(bbox_z_min,this -> kd_tree -> get_points_normals() -> at(i) -> get_point() -> at(2));


	}

	arma::vec C = {(bbox_x_max + bbox_x_min)/2,(bbox_y_max + bbox_y_min)/2,(bbox_z_max + bbox_z_min)/2};
	

	return C;
}

double PC::get_bbox_diagonal() const{

	double bbox_x_min = std::numeric_limits<double>::infinity();
	double bbox_y_min = std::numeric_limits<double>::infinity();
	double bbox_z_min = std::numeric_limits<double>::infinity();

	double bbox_x_max = - bbox_x_min;
	double bbox_y_max = - bbox_y_min;
	double bbox_z_max = - bbox_z_min;

	unsigned int size = this -> kd_tree -> get_points_normals() -> size();

	#pragma omp parallel for reduction(min:bbox_x_min,bbox_y_min,bbox_z_min) reduction(max:bbox_x_max,bbox_y_max,bbox_z_max) if (USE_OMP_PC)
	for (unsigned int i = 0; i < size; ++i) {

		bbox_x_max = std::max(bbox_x_max,this -> kd_tree -> get_points_normals() -> at(i) -> get_point() -> at(0));
		bbox_y_max = std::max(bbox_y_max,this -> kd_tree -> get_points_normals() -> at(i) -> get_point() -> at(1));
		bbox_z_max = std::max(bbox_z_max,this -> kd_tree -> get_points_normals() -> at(i) -> get_point() -> at(2));

		bbox_x_min = std::min(bbox_x_min,this -> kd_tree -> get_points_normals() -> at(i) -> get_point() -> at(0));
		bbox_y_min = std::min(bbox_y_min,this -> kd_tree -> get_points_normals() -> at(i) -> get_point() -> at(1));
		bbox_z_min = std::min(bbox_z_min,this -> kd_tree -> get_points_normals() -> at(i) -> get_point() -> at(2));


	}
	arma::vec top = {bbox_x_max,bbox_y_max,bbox_z_max};
	arma::vec bottom = {bbox_x_min,bbox_y_min,bbox_z_min};


	return arma::norm(top - bottom);

}




