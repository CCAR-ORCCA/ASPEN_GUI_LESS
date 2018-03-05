#include "ShapeModel.hpp"
#include <chrono>


ShapeModel::ShapeModel() {

}

ShapeModel::ShapeModel(std::string ref_frame_name,
	FrameGraph * frame_graph) {
	this -> frame_graph = frame_graph;
	this -> ref_frame_name = ref_frame_name;
}


std::string ShapeModel::get_ref_frame_name() const {
	return this -> ref_frame_name;
}


arma::mat ShapeModel::get_inertia() const {
	return this -> inertia;

}

double ShapeModel::get_volume() const{
	return this -> volume;
}



double ShapeModel::get_surface_area() const {
	return this -> surface_area;
}


arma::vec ShapeModel::get_center_of_mass() const{
	return this -> cm;
}


void ShapeModel::shift_to_barycenter() {

	arma::vec x = - this -> get_center_of_mass();

	// The vertices are shifted
	#pragma omp parallel for if(USE_OMP_SHAPE_MODEL)
	for (unsigned int vertex_index = 0;
		vertex_index < this -> get_NControlPoints();
		++vertex_index) {

		this -> control_points[vertex_index] -> set_coordinates(this -> control_points[vertex_index] -> get_coordinates() + x);

}

this -> cm = 0 * this -> cm;

}

void ShapeModel:: align_with_principal_axes() {


	this -> compute_inertia();

	std::cout << "Non-dimensional inertia: " << std::endl;
	std::cout << this -> inertia << std::endl;

	arma::vec moments;
	arma::mat axes;

	this -> get_principal_inertias(axes,moments);

	this -> rotate(axes.t());

	this -> inertia = arma::diagmat(moments);

}






void ShapeModel::construct_kd_tree_control_points() {

	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	this -> kdt_control_points = std::make_shared<KDTree_control_points>(KDTree_control_points());
	this -> kdt_control_points = this -> kdt_control_points -> build(this -> control_points, 0);

	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;


	std::cout << "\n Elapsed time during KDTree construction : " << elapsed_seconds.count() << "s\n\n";

}

void ShapeModel::set_ref_frame_name(std::string ref_frame_name) {

	this -> ref_frame_name = ref_frame_name;
}


std::vector<std::shared_ptr< ControlPoint> > * ShapeModel::get_control_points() {
	return &this -> control_points;
}


std::shared_ptr< ControlPoint>  ShapeModel::get_control_point(unsigned int i) const {
	return this -> control_points.at(i);
}


std::shared_ptr<KDTree_control_points> ShapeModel::get_KDTree_control_points() const {
	return this -> kdt_control_points;
}


unsigned int ShapeModel::get_NElements() const {
	return this -> elements . size();
}

unsigned int ShapeModel::get_NControlPoints() const {
	return this -> control_points . size();
}

void ShapeModel::initialize_index_table(){
	// The forward look up table is created
	for (auto iter = this -> control_points.begin(); iter != this -> control_points.end(); ++iter){
		this -> pointer_to_global_index[*iter] = this -> pointer_to_global_index.size();
		
	}
}


std::shared_ptr<KDTree_shape> ShapeModel::get_KDTree_shape() const {
	return this -> kdt_facet;
}


arma::vec ShapeModel::get_center() const{
	arma::vec center = {0,0,0};
	unsigned int N = this -> get_NControlPoints();

	for (auto point = this -> control_points.begin(); point != this -> control_points.end(); ++point){
		center += (*point) -> get_coordinates() / N;
	}
	return center;
}

void ShapeModel::translate(arma::vec x){

	for (auto point = this -> control_points.begin(); point != this -> control_points.end(); ++point){
		arma::vec coords = (*point) -> get_coordinates();
		(*point) -> set_coordinates(coords + x) ;
	}
}

void ShapeModel::rotate(arma::mat M){

	for (auto point = this -> control_points.begin(); point != this -> control_points.end(); ++point){
		arma::vec coords = (*point) -> get_coordinates();
		(*point) -> set_coordinates(M*coords) ;
	}
}


void ShapeModel::get_bounding_box(double * bounding_box,arma::mat M) const {

	arma::vec bbox_min = M * this -> control_points[0] -> get_coordinates();
	arma::vec bbox_max = M * this -> control_points[0] -> get_coordinates();

	for ( unsigned int vertex_index = 0; vertex_index < this -> get_NControlPoints(); ++ vertex_index) {
		bbox_min = arma::min(bbox_min,M * this -> control_points[vertex_index] -> get_coordinates());
		bbox_max = arma::max(bbox_max,M * this -> control_points[vertex_index] -> get_coordinates());

	}

	bounding_box[0] = bbox_min(0);
	bounding_box[1] = bbox_min(1);
	bounding_box[2] = bbox_min(2);
	bounding_box[3] = bbox_max(0);
	bounding_box[4] = bbox_max(1);
	bounding_box[5] = bbox_max(2);

	std::cout << "- Bounding box : " << std::endl;
	std::cout << "-- xmin : " << bbox_min(0) << std::endl;
	std::cout << "-- xmax : " << bbox_max(0) << std::endl;

	std::cout << "-- ymin : " << bbox_min(1) << std::endl;
	std::cout << "-- ymax : " << bbox_max(1) << std::endl;


	std::cout << "-- zmin : " << bbox_min(2) << std::endl;
	std::cout << "-- zmax : " << bbox_max(2) << std::endl;


}


void ShapeModel::get_principal_inertias(arma::mat & axes,arma::vec & moments) const{


	arma::eig_sym(moments,axes,this -> inertia);
	

	// Check if two axes are flipped here

	// The longest distance measured from the center of mass along the first principal
	// axis to the surface should be measured along +x

	double bbox[6];
	
	arma::vec e0 = axes.col(0);
	arma::vec e1 = axes.col(1);
	arma::vec e2 = axes.col(2);


	if (arma::det(axes) < 0){
		e0 = -e0;
	}

	axes = arma::join_rows(e0,arma::join_rows(axes.col(1),axes.col(2)));


	this -> get_bounding_box(bbox,axes.t());
	arma::vec x_max = {bbox[3],bbox[4],bbox[5]}; 
	arma::vec x_min = {bbox[0],bbox[1],bbox[2]}; 

	arma::mat M0 = arma::eye<arma::mat>(3,3);
	arma::mat M1 = {{1,0,0},{0,-1,0},{0,0,-1}};
	arma::mat M2 = {{-1,0,0},{0,1,0},{0,0,-1}};
	arma::mat M3 = {{-1,0,0},{0,-1,0},{0,0,1}};

	if (std::abs(arma::dot(x_max,e0)) > std::abs(arma::dot(x_min,e0))){

		if(std::abs(arma::dot(x_max,e1)) > std::abs(arma::dot(x_min,e1))){
			axes = axes * M0;
		}

		else{
			axes = axes * M1;
		}


	}
	else{
		if(std::abs(arma::dot(x_max,e1)) > std::abs(arma::dot(x_min,e1))){
			axes = axes * M2;
		}

		else{

			axes = axes * M3;
		}
	}
	

	std::cout << "Principal axes: " << std::endl;
	std::cout << axes << std::endl;



	std::cout << "Non-dimensional principal moments: " << std::endl;
	std::cout << moments << std::endl;


}



unsigned int ShapeModel::get_control_point_index(std::shared_ptr<ControlPoint> point) const{
	return this -> pointer_to_global_index.at(point);
}


std::vector<std::shared_ptr<Element> > * ShapeModel::get_elements() {
	return &this -> elements;
}


void ShapeModel::add_element(std::shared_ptr<Element> el) {
	this -> elements.push_back(el);
}


void ShapeModel::add_control_point(std::shared_ptr<ControlPoint> vertex) {
	this -> control_points.push_back(vertex);
}

