#include "PC.hpp"
#include "PointDescriptor.hpp"
#include "SPFH.hpp"
#include "PFH.hpp"
#include "FPFH.hpp"

#define PC_DEBUG_FLAG 1

PC::PC(std::vector<std::shared_ptr<Ray> > * focal_plane, int label_) {

	std::vector< std::shared_ptr<PointNormal> > points_normals;

	// The valid measurements used to form the point cloud are extracted
	for (unsigned int pixel = 0; pixel < focal_plane -> size(); ++pixel) {

		if (focal_plane -> at(pixel) -> get_true_range() < std::numeric_limits<double>::infinity()) {

			arma::vec impact_point_instrument = focal_plane -> at(pixel) -> get_impact_point();

			points_normals.push_back(std::make_shared<PointNormal>(PointNormal(impact_point_instrument)));
			
		}
	}

	this -> los = {1,0,0};
	
	this -> construct_kd_tree(points_normals);
	this -> construct_normals(this -> los);
	this -> label = std::to_string(label_);

}



PC::PC(arma::vec los_dir, arma::mat & points) {
	std::vector< std::shared_ptr<PointNormal> > points_normals;

	for (unsigned int index = 0; index < points . n_cols; ++index) {
		points_normals.push_back(std::make_shared<PointNormal>(PointNormal(points . col(index))));
	}
	this -> construct_kd_tree(points_normals);

	this -> construct_normals(los_dir);

}

PC::PC(arma::mat & points,arma::mat & normals) {

	std::vector< std::shared_ptr<PointNormal> > points_normals;

	for (unsigned int index = 0; index < points . n_cols; ++index) {
		points_normals.push_back(std::make_shared<PointNormal>(PointNormal(points . col(index),
			normals.col(index))));
	}

	this -> construct_kd_tree(points_normals);

}

PC::PC(std::vector< std::shared_ptr<PointNormal> > points_normals) {
	
	this -> construct_kd_tree(points_normals);

}


PC::PC(std::vector< std::shared_ptr<PC> > & pcs,int points_retained){

	std::vector< std::shared_ptr<PointNormal> > points_normals;
	double downsampling_factor = 1;
	int N_points_total = 0;
	for (unsigned int i = 0; i < pcs.size();++i){

		N_points_total += pcs[i] -> get_size();
	}

	if (points_retained > 0){
		downsampling_factor = double(points_retained) / N_points_total;
	}

	for (unsigned int i = 0; i < pcs.size();++i){

		std::vector< std::shared_ptr<PointNormal> > points_from_pc = pcs[i] -> get_points();

		arma::uvec random_order =  arma::regspace< arma::uvec>(0,  pcs[i] -> get_size() - 1);		
		random_order = arma::shuffle(random_order);	

		int points_to_keep = (int)	(downsampling_factor *  pcs[i] -> get_size());

		for (unsigned int p = 0; p < points_to_keep; ++p){
			points_normals.push_back(points_from_pc[random_order(p)]);
		}

	}
	
	this -> construct_kd_tree(points_normals);

}


PC::PC(std::string filename) {
	std::cout << " Reading " << filename << std::endl;

	std::ifstream ifs(filename);

	if (!ifs.is_open()) {
		std::cout << "There was a problem opening the input file!\n";
		throw;
	}

	std::string line;
	std::vector<arma::vec> points;
	std::vector<std::vector<unsigned int> > shape_patch_indices;


	while (std::getline(ifs, line)) {

		std::stringstream linestream(line);


		char type;
		linestream >> type;

		if (type == '#' || type == 's'  || type == 'o' || type == 'm' || type == 'u' || line.size() == 0) {
			continue;
		}

		else if (type == 'v') {
			double vx, vy, vz;
			linestream >> vx >> vy >> vz;
			arma::vec vertex = {vx, vy, vz};
			points.push_back(vertex);

		}

		else {
			throw(std::runtime_error(" unrecognized type: "  + std::to_string(type)));
		}

	}


	std::vector< std::shared_ptr<PointNormal> > points_normals;
	arma::vec los_dir = {1,0,0};

	for (unsigned int index = 0; index < points.size(); ++index) {
		points_normals.push_back(std::make_shared<PointNormal>(PointNormal(points[index])));
	}
	this -> construct_kd_tree(points_normals);

	this -> construct_normals(los_dir);

}




void PC::transform(const arma::mat & dcm, const arma::vec & x){

	std::vector< std::shared_ptr<PointNormal> > points_normals;

	// The valid measurements used to form the point cloud are extracted
	for (unsigned int i = 0; i < this -> kdt_points  -> get_points_normals() -> size(); ++i) {

		std::shared_ptr<PointNormal> pn = this -> kdt_points  -> get_points_normals() -> at(i);

		pn -> set_point(dcm * pn -> get_point() + x);
		pn -> set_normal(dcm * pn -> get_normal());

		points_normals.push_back(pn);
	}

	this -> construct_kd_tree(points_normals);


}


std::vector< std::shared_ptr<PointNormal> > PC::get_points() const{
	return this -> kdt_points -> points_normals;
}



PC::PC(ShapeModelTri * shape_model) {

	std::vector< std::shared_ptr<PointNormal> > points_normals;


	// The shape model is used to create the point cloud
	// The center points to each facets are used
	// The normals of each facet are directly used
	for (unsigned int facet_index = 0; facet_index < shape_model -> get_NElements(); ++facet_index) {
		Facet * facet = dynamic_cast<Facet *>(shape_model -> get_elements() -> at(facet_index).get());

		arma::vec C = facet -> get_center();

		arma::vec C0 = facet -> get_control_points() -> at(0) -> get_coordinates();
		arma::vec C1 = facet -> get_control_points() -> at(1) -> get_coordinates();
		arma::vec C2 = facet -> get_control_points() -> at(2) -> get_coordinates();

		points_normals.push_back(std::make_shared<PointNormal>(PointNormal(C, facet -> get_normal())));
		points_normals.push_back(std::make_shared<PointNormal>(PointNormal( 0.5 * (C0 + C), facet -> get_normal())));
		points_normals.push_back(std::make_shared<PointNormal>(PointNormal( 0.5 * (C1 + C), facet -> get_normal())));
		points_normals.push_back(std::make_shared<PointNormal>(PointNormal( 0.5 * (C2 + C), facet -> get_normal())));


	}

	for (unsigned int vertex_index = 0; vertex_index < shape_model -> get_NControlPoints(); ++vertex_index) {
		
		std::shared_ptr<ControlPoint> control_point = shape_model -> get_control_points() -> at(vertex_index);

		arma::vec n = {0,0,0};

		std::set< Element *  >  owning_elements = control_point -> get_owning_elements();
		
		for (auto iter = owning_elements.begin() ; iter != owning_elements.end(); ++iter){
			n +=  (*iter) -> get_normal();
		}

		n = arma::normalise(n);

		points_normals.push_back(std::make_shared<PointNormal>(PointNormal(control_point -> get_coordinates(), n)));
		


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
	return this -> kdt_points  -> points_normals.size();
}

arma::vec PC::get_point_coordinates(unsigned int index) const {
	return this -> kdt_points  -> points_normals[index] -> get_point();
}

std::shared_ptr<PointNormal> PC::get_point(unsigned int index) const {
	return this -> kdt_points  -> points_normals[index];
}



arma::vec PC::get_point_normal(unsigned int index) const {
	return this -> kdt_points  -> points_normals[index] -> get_normal();
}




void PC::construct_kd_tree(std::vector< std::shared_ptr<PointNormal> > & points_normals) {

	// The KD Tree is now constructed
	this -> kdt_points  = std::make_shared<KDTreePC>(KDTreePC());
	this -> kdt_points  = this -> kdt_points  -> build(points_normals, 0);
}

std::shared_ptr<PointNormal> PC::get_closest_point(const arma::vec & test_point) const {

	double distance = std::numeric_limits<double>::infinity();
	std::shared_ptr<PointNormal> closest_point;

	this -> kdt_points  -> closest_point_search(test_point,
		this -> kdt_points,
		closest_point,
		distance);

	return closest_point;

}

std::vector<std::shared_ptr<PointNormal> > PC::get_points_in_sphere(
	arma::vec test_point, const double & radius) const {

	std::vector<std::shared_ptr<PointNormal> > closest_points;

	this -> kdt_points  -> radius_point_search(test_point,this -> kdt_points,radius,closest_points);


	#if PC_DEBUG_FLAG

	std::cout << "Search within sphere of radius " << radius <<  " returned " << closest_points.size() << " points\n";

	#endif


	return closest_points;

}


std::map<double,std::shared_ptr<PointNormal> > PC::get_closest_N_points(const arma::vec & test_point, 
	const unsigned int & N) const {

	std::map<double,std::shared_ptr<PointNormal> > closest_points;
	double distance = std::numeric_limits<double>::infinity();

	this -> kdt_points -> closest_N_point_search(test_point,N,this -> kdt_points,distance,closest_points);

	return closest_points;

}

void  PC::save(std::string path, 
	arma::mat dcm, arma::vec x, 
	bool save_normals,
	bool format_like_obj) const {

	std::ofstream shape_file;
	shape_file.open(path);

	if (save_normals == format_like_obj && save_normals == true) {
		throw (std::runtime_error("save can't be called with those arguments!"));
	}

	for (unsigned int vertex_index = 0;vertex_index < this -> get_size();++vertex_index) {
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

void PC::save(arma::mat & points,std::string path) {


	std::ofstream shape_file;
	shape_file.open(path);

	for (unsigned int e = 0;e < points.n_cols;++e) {
		shape_file << "v " << points(0,e) << " " << points(1,e) << " " << points(2,e) << std::endl;
	}

}




void PC::construct_normals(arma::vec los_dir) {


	// This stores the average size of the ball enclosing all the neighbors of a given point
	// during the surface computation phase
	arma::vec neighborhood_sizes = arma::zeros<arma::vec>(this -> kdt_points  -> get_points_normals() -> size());

	// #pragma omp parallel for if (USE_OMP_PC)
	for (unsigned int i = 0; i < this -> kdt_points  -> get_points_normals() -> size(); ++i) {

		std::shared_ptr<PointNormal> pn = this -> kdt_points  -> get_points_normals() -> at(i);

		// Get the N nearest neighbors to this point
		unsigned int N = 5;
		std::map<double,std::shared_ptr<PointNormal> > closest_points = this -> get_closest_N_points(pn -> get_point(), N);

		// This N nearest neighbors are used to get the normal
		arma::mat points_augmented(N, 4);
		points_augmented.col(3) = arma::ones<arma::vec>(N);

		unsigned int j = 0;
		for (auto it = closest_points.begin(); it != closest_points.end(); ++it) {

			points_augmented.row(j).cols(0, 2) = it -> second -> get_point().t();

			++j;

		}

		neighborhood_sizes(i) = (--closest_points.end()) -> first ;


		// The eigenvalue problem is solved
		arma::vec eigval;
		arma::mat eigvec;

		arma::eig_sym(eigval, eigvec, points_augmented.t() * points_augmented);
		arma::vec n = arma::normalise(eigvec.col(arma::abs(eigval).index_min()).rows(0, 2));

		// The normal is flipped to make sure it is facing the los
		if (arma::dot(n, los_dir) < 0) {
			this -> kdt_points  -> get_points_normals() -> at(i) -> set_normal(n);
		}
		else {
			this -> kdt_points  -> get_points_normals() -> at(i) -> set_normal(-n);
		}

	}

	// double neighborhood_sizes_mean = arma::mean(neighborhood_sizes);
	// double neighborhood_sizes_sd = arma::stddev(neighborhood_sizes);


	// // Points that are too sparsely surrounded are flagged as irrelevant 
	// unsigned int discarded_points = 0;
	// for (unsigned int i = 0; i < this -> kdt_points  -> get_points_normals() -> size(); ++i) {
	// 	if (std::abs(neighborhood_sizes(i) - neighborhood_sizes_mean) > 3. * neighborhood_sizes_sd){
	// 		++discarded_points;
	// 		this -> kdt_points  -> get_points_normals() -> at(i) -> set_is_unique_feature(false);
	// 	}
	// }



}




arma::vec::fixed<3> PC::get_center() const{

	double c_x = 0;
	double c_y = 0;
	double c_z = 0;

	unsigned int size = this -> kdt_points  -> get_points_normals() -> size();

	#pragma omp parallel for reduction(+:c_x,c_y,c_z) if (USE_OMP_PC)
	for (unsigned int i = 0; i < size; ++i) {

		arma::vec point = this -> kdt_points  -> get_points_normals() -> at(i) -> get_point();

		c_x += point(0) / size;
		c_y += point(1) / size;
		c_z += point(2) / size;

	}
	arma::vec center = {c_x,c_y,c_z};
	assert(arma::norm(center - C) == 0);


	return center;
}


arma::mat::fixed<3,3> PC::get_principal_axes() const{

	arma::mat::fixed<3,3> P,E;
	P.fill(0);

	int size = (int)(this -> kdt_points  -> get_points_normals() -> size());

	arma::vec::fixed<3> center = this -> get_center();

	for (unsigned int i = 0; i < size; ++i) {
		arma::vec point = this -> kdt_points  -> get_points_normals() -> at(i) -> get_point();
		P += 1./(size - 1) * (point - center) * (point - center).t();
	}
	
	arma::vec eigval;
	arma::eig_sym( eigval, E, P );

	if (arma::det(E) < 0){
		E.col(0) *= -1;
	}

	return E;
}



arma::vec PC::get_bbox_center() const{

	
	unsigned int size = this -> kdt_points  -> get_points_normals() -> size();

	arma::vec bbox_min = this -> kdt_points  -> get_points_normals() -> at(0) -> get_point();
	arma::vec bbox_max = this -> kdt_points  -> get_points_normals() -> at(0) -> get_point();

	// #pragma omp parallel for reduction(min:bbox_x_min,bbox_y_min,bbox_z_min) reduction(max:bbox_x_max,bbox_y_max,bbox_z_max) if (USE_OMP_PC)
	for (unsigned int i = 0; i < size; ++i) {
		arma::vec point = this -> kdt_points  -> get_points_normals() -> at(i) -> get_point();
		bbox_max = arma::max(bbox_max,point);
		bbox_min = arma::min(bbox_min,point);

	}
	
	arma::vec C = 0.5 * (bbox_max + bbox_min);
	

	return C;
}

arma::vec PC::get_bbox_dim() const{

	
	unsigned int size = this -> kdt_points  -> get_points_normals() -> size();

	arma::vec bbox_max = this -> kdt_points  -> get_points_normals() -> at(0) -> get_point();
	arma::vec bbox_min = this -> kdt_points  -> get_points_normals() -> at(0) -> get_point();


	// #pragma omp parallel for reduction(min:bbox_x_min,bbox_y_min,bbox_z_min) reduction(max:bbox_x_max,bbox_y_max,bbox_z_max) if (USE_OMP_PC)
	for (unsigned int i = 0; i < size; ++i) {
		arma::vec point = this -> kdt_points  -> get_points_normals() -> at(i) -> get_point();

		bbox_max = arma::max(bbox_max,point);
		bbox_min = arma::min(bbox_min,point);

	}
	

	return 0.5 *(bbox_max - bbox_min);
}

double PC::get_bbox_diagonal() const{

	unsigned int size = this -> kdt_points  -> get_points_normals() -> size();

	arma::vec bbox_max = this -> kdt_points  -> get_points_normals() -> at(0) -> get_point();
	arma::vec bbox_min = this -> kdt_points  -> get_points_normals() -> at(0) -> get_point();

	// #pragma omp parallel for reduction(min:bbox_x_min,bbox_y_min,bbox_z_min) reduction(max:bbox_x_max,bbox_y_max,bbox_z_max) if (USE_OMP_PC)
	for (unsigned int i = 0; i < size; ++i) {
		arma::vec point = this -> kdt_points  -> get_points_normals() -> at(i) -> get_point();

		bbox_max = arma::max(bbox_max,point);
		bbox_min = arma::min(bbox_min,point);

	}

	return arma::norm(bbox_max - bbox_min);

}

std::string PC::get_label() const{
	return this -> label;
}



void PC::compute_feature_descriptors(int type,bool keep_correlations,int N_bins,double neighborhood_radius){


	std::vector<std::shared_ptr<PointNormal> > relevant_point_with_descriptors;
	unsigned int size = this -> kdt_points  -> get_points_normals() -> size();


	std::vector<double> radii = {neighborhood_radius,1.25*neighborhood_radius,1.5 *neighborhood_radius,1.75 * neighborhood_radius};

	for (unsigned int index = 0; index < radii.size(); ++index){
		
		double test_radius = radii[index];

		#if PC_DEBUG_FLAG
		std::cout << "Computing surface descriptors at r_" << index << " = " << test_radius << " ... \n";
		#endif
		
		if (type == 0){
			this -> compute_PFH(keep_correlations,N_bins,test_radius);
		}
		else{
			this -> compute_FPFH(keep_correlations,N_bins,test_radius);
		}
		this -> compute_mean_feature_histogram();
		
		this -> prune_features();

		#if PC_DEBUG_FLAG
		std::cout << "saving active features...\n";
		this -> save_active_features(index);
		#endif

	}


	// Only the unique features are kept
	for (int i = 0 ; i < size; ++i){
		std::shared_ptr<PointNormal> query_point = this -> kdt_points  -> get_points_normals() -> at(i);
		if (query_point -> get_is_unique_feature()) {
			relevant_point_with_descriptors.push_back(query_point);
		}
	}


	#if PC_DEBUG_FLAG
	std::cout << "Creating KDTree with active features...\n";
	#endif

	this -> kdt_descriptors = std::make_shared<KDTreeDescriptors>(KDTreeDescriptors());
	this -> kdt_descriptors = this -> kdt_descriptors  -> build(relevant_point_with_descriptors, 0);

}



void PC::save_active_features(int index) const{


	unsigned int size = this -> kdt_points  -> get_points_normals() -> size();
	unsigned int histo_size = this -> kdt_points  -> get_points_normals() -> at(0) -> get_histogram_size();
	std::vector<std::shared_ptr<PointNormal> > active_features;

	arma::mat active_features_histograms = arma::zeros<arma::mat>(histo_size,size);
	arma::mat active_features_histograms_spfh = arma::zeros<arma::mat>(histo_size,size);

	
	for (unsigned int k  = 0; k < size; ++k){
		if(this -> kdt_points  -> get_points_normals() -> at(k) -> get_is_unique_feature()){
			active_features.push_back(this -> kdt_points  -> get_points_normals() -> at(k));
			active_features_histograms.col(k) = this -> kdt_points  -> get_points_normals() -> at(k) -> get_descriptor_histogram();
			active_features_histograms_spfh.col(k) = this -> kdt_points  -> get_points_normals() -> at(k) -> get_spfh_histogram();
			
		};
	}

	PC active_features_pc(active_features);

	active_features_pc.save("active_features_" + std::to_string(index) + ".obj");
	
	this -> mean_feature_histogram.save("mean_histogram_" + std::to_string(index) + ".txt",arma::raw_ascii);
	active_features_histograms.save("active_features_histograms_"+ std::to_string(index) + ".txt",arma::raw_ascii);
	active_features_histograms_spfh.save("active_features_histograms_spfh_"+ std::to_string(index) + ".txt",arma::raw_ascii);

}






void PC::compute_PFH(bool keep_correlations,int N_bins,double neighborhood_radius){

	unsigned int size = this -> kdt_points  -> get_points_normals() -> size();

	if (neighborhood_radius < 0){
		throw(std::runtime_error("neighborhood_radius is negative"));
	}

	for (unsigned int i = 0; i < size; ++i) {
		std::shared_ptr<PointNormal> query_point = this -> kdt_points  -> get_points_normals() -> at(i);
		std::vector<std::shared_ptr<PointNormal> > neighbors_inclusive = this -> get_points_in_sphere(query_point -> get_point(),neighborhood_radius);
		query_point -> set_descriptor(PFH(neighbors_inclusive,keep_correlations,N_bins));
	}	



}


void PC::compute_FPFH(bool keep_correlations,int N_bins,double neighborhood_radius){

	unsigned int size = this -> kdt_points  -> get_points_normals() -> size();

	if (neighborhood_radius < 0){
		throw(std::runtime_error("neighborhood_radius is negative"));
	}

	for (unsigned int i = 0; i < size; ++i) {
		std::shared_ptr<PointNormal> query_point = this -> kdt_points  -> get_points_normals() -> at(i);
		std::vector<std::shared_ptr<PointNormal> > neighbors_inclusive = this -> get_points_in_sphere(query_point -> get_point(),neighborhood_radius);

		query_point -> set_SPFH(SPFH(query_point,neighbors_inclusive,keep_correlations,N_bins));
	}

	for (unsigned int i = 0; i < size; ++i) {
		std::shared_ptr<PointNormal> query_point = this -> kdt_points  -> get_points_normals() -> at(i);

		query_point -> set_descriptor(FPFH(query_point));
	}

}


void PC::save_point_descriptors(std::string path) const{
	unsigned int size = this -> kdt_points  -> get_points_normals() -> size();
	unsigned int histo_size = this -> kdt_points  -> get_points_normals() -> at(0)-> get_histogram_size();
	arma::mat active_features_histograms = arma::zeros<arma::mat>(histo_size,size);
	int relevant_feature_count = 0;
	
	for (unsigned int i = 0; i < size; ++i) {
		
		active_features_histograms.col(i) = this -> kdt_points  -> get_points_normals() -> at(i) -> get_descriptor_histogram();

		if (this -> kdt_points  -> get_points_normals() -> at(i) -> get_is_unique_feature()){
			++relevant_feature_count;
		}

	}

	std::cout << relevant_feature_count << " features found to be unique in " << path << std::endl;

	active_features_histograms.save(path,arma::raw_ascii);

}



std::vector<PointPair>  PC::find_pch_matches_kdtree(std::shared_ptr<PC> pc0,std::shared_ptr<PC> pc1){


	PointDescriptor descriptor;
	std::vector<PointPair> matches;	
	std::vector<PointPair> all_matched_pairs;

	std::set<std::shared_ptr< PointNormal > > used_destination_points;


	std::set<unsigned int> pc1_used_indices;

	arma::vec match_distances(pc0 -> get_size());

	for (unsigned int i = 0; i < pc0 -> get_size(); ++i)  {

		double distance;
		auto best_match = pc1 -> get_best_match_feature_point(pc0 -> get_point(i),distance);

		match_distances(i) = distance;
		all_matched_pairs.push_back(std::make_pair(pc0 -> get_point(i),best_match));
		
	}

	double mean_distance_between_matches = arma::mean(match_distances);
	double std_distance_between_matches = arma::stddev(match_distances);


	// Excluding matches that yield an error more than X sigma away from the mean error
	// and only allowing destination points to be paired once
	for (unsigned int i = 0; i < pc0 -> get_size(); ++i)  {
		if (std::abs(match_distances(i) - mean_distance_between_matches) < 2 * std_distance_between_matches){

			if(used_destination_points.find(all_matched_pairs[i].second) == used_destination_points.end()){
				matches.push_back(all_matched_pairs[i]);
				used_destination_points.insert(all_matched_pairs[i].second);
			}

		}
	}

	return matches;

}



void PC::save_pch_matches(const std::multimap<double,std::pair<int,int> > matches,std::string path){


	arma::mat matches_mat(matches.size(), 3);
	int count = 0;
	for (auto it = matches.begin();it != matches.end(); ++it ){

		matches_mat(count,0)= it -> first;
		matches_mat(count,1)= it -> second.first;
		matches_mat(count,2)= it -> second.second;

		++count;
	}

	matches_mat.save(path,arma::raw_ascii);

}


std::shared_ptr<PointNormal> PC::get_best_match_feature_point(std::shared_ptr<PointNormal> other_point,double & distance) const{

	distance = std::numeric_limits<double>::infinity();
	std::shared_ptr<PointNormal> closest_point;

	this -> kdt_descriptors  -> closest_point_search(other_point,
		this -> kdt_descriptors,
		closest_point,
		distance);

	return closest_point;

}


void PC::compute_mean_feature_histogram(){

	unsigned int size = this -> kdt_points  -> get_points_normals() -> size();
	unsigned int histo_size = this -> kdt_points  -> get_points_normals() -> at(0) -> get_histogram_size();
	this -> mean_feature_histogram = arma::zeros<arma::vec>(histo_size);

	for (unsigned int k = 0; k < size; ++k){
		this -> mean_feature_histogram += this -> kdt_points  -> get_points_normals() -> at(k) -> get_descriptor_histogram();
	}

	this -> mean_feature_histogram = this -> mean_feature_histogram / arma::max(this -> mean_feature_histogram);

	


}

void PC::prune_features() {

	// The distance between each point histogram and the mean histogram is computed
	unsigned int size = this -> kdt_points  -> get_points_normals() -> size();
	unsigned int histo_size = this -> kdt_points  -> get_points_normals() -> at(0) -> get_histogram_size();
	auto all_points = this -> kdt_points  -> get_points_normals();
	arma::vec distances = arma::zeros<arma::vec>(size);
	
	#pragma omp parallel for
	for (unsigned int k = 0; k < size; ++k){
		for (int i = 0; i < histo_size; ++i){
			double mu_i = this -> mean_feature_histogram(i);
			double p_i = all_points -> at(k) -> get_histogram_value(i);
			distances(k) += std::pow(mu_i - p_i,2) / (mu_i + p_i);
		}
	}

	distances = distances / arma::stddev(distances);

	// Points whose feature descriptor is less than 1.25 standard deviations away from the mean are considered as 
	// inliers, thus irrelevant as discriminative features
	
	unsigned int irrelevant_features_count = 0;

	for (unsigned int k = 0; k < size; ++k){
		if (distances(k) < 1.75){
			all_points -> at(k) -> set_is_unique_feature(false);
			++ irrelevant_features_count;
		}
		
	}

	#if PC_DEBUG_FLAG
	std::cout << "Discarded " << irrelevant_features_count << " redundant features \n  ";
	#endif
}


