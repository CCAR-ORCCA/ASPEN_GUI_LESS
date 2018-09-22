#include "PC.hpp"
#include "PointDescriptor.hpp"
#include "SPFH.hpp"
#include "PFH.hpp"
#include "FPFH.hpp"


#define PC_DEBUG_FLAG 1

PC::PC(std::vector<std::shared_ptr<Ray> > * focal_plane, int label_) {

	this -> points_normals.clear();
	
	// The valid measurements used to form the point cloud are extracted
	for (unsigned int pixel = 0; pixel < focal_plane -> size(); ++pixel) {

		if (focal_plane -> at(pixel) -> get_true_range() < std::numeric_limits<double>::infinity()) {

			arma::vec impact_point_instrument = focal_plane -> at(pixel) -> get_impact_point();

			this -> points_normals.push_back(PointNormal(impact_point_instrument,pixel));

		}
	}

	this -> label = std::to_string(label_);

}

PC::PC(arma::mat & points,arma::mat & normals) {

	std::vector< std::shared_ptr<PointNormal> > points_normals;

	for (unsigned int index = 0; index < points . n_cols; ++index) {
		this -> points_normals.push_back(PointNormal(points . col(index),normals.col(index),index));
	}


}

PC::PC(const std::vector< PointNormal > & points_normals) {

	this -> points_normals.clear();
	// The valid measurements used to form the point cloud are extracted
	for (unsigned int i = 0; i < points_normals.size(); ++i) {

		this -> points_normals[i].set_global_index(i);
		this -> points_normals.push_back(points_normals[i]);

	}

}

PC::PC(std::string filename){

	std::cout << "Reading " << filename << std::endl;

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
			throw(std::runtime_error(" unrecognized character in input file : "  + std::to_string(type)));
		}

	}

	this -> points_normals.clear();
	for (unsigned int index = 0; index < points.size(); ++index) {
		this -> points_normals.push_back(PointNormal(points[index],index));
	}


}



void PC::transform(const arma::mat & dcm, const arma::vec & x){

	// The valid measurements used to form the point cloud are extracted
	#pragma omp parallel for
	for (unsigned int i = 0; i < this -> points_normals .  size(); ++i) {
		PointNormal & p = this -> points_normals.at(i);
		p. set_point_coordinates(dcm * p. get_point_coordinates() + x);
		p. set_normal_coordinates(dcm * p. get_normal_coordinates());
	}

	this -> build_kd_tree_points();

}


PC::PC(ShapeModelTri * shape_model) {

	this -> points_normals.clear();


	// The shape model is used to create the point cloud
	// The center points to each facets are used
	// The normals of each facet are directly used
	for (unsigned int facet_index = 0; facet_index < shape_model -> get_NElements(); ++facet_index) {
		Facet * facet = dynamic_cast<Facet *>(shape_model -> get_elements() -> at(facet_index).get());

		arma::vec C = facet -> get_center();
		arma::vec C0 = facet -> get_control_points() -> at(0) -> get_coordinates();
		arma::vec C1 = facet -> get_control_points() -> at(1) -> get_coordinates();
		arma::vec C2 = facet -> get_control_points() -> at(2) -> get_coordinates();

		this -> points_normals.push_back(PointNormal(C, facet -> get_normal_coordinates(),this -> points_normals.size()));
		this -> points_normals.push_back(PointNormal( 0.5 * (C0 + C), facet -> get_normal_coordinates(),this -> points_normals.size()));
		this -> points_normals.push_back(PointNormal( 0.5 * (C1 + C), facet -> get_normal_coordinates(),this -> points_normals.size()));
		this -> points_normals.push_back(PointNormal( 0.5 * (C2 + C), facet -> get_normal_coordinates(),this -> points_normals.size()));

	}

	for (unsigned int vertex_index = 0; vertex_index < shape_model -> get_NControlPoints(); ++vertex_index) {

		std::shared_ptr<ControlPoint> control_point = shape_model -> get_control_points() -> at(vertex_index);

		arma::vec n = {0,0,0};

		std::set< Element *  >  owning_elements = control_point -> get_owning_elements();

		for (auto iter = owning_elements.begin() ; iter != owning_elements.end(); ++iter){
			n +=  (*iter) -> get_normal_coordinates();
		}

		n = arma::normalise(n);

		this -> points_normals.push_back(PointNormal(control_point -> get_coordinates(), n,this -> points_normals.size()));

	}


}



int PC::get_closest_point(const arma::vec & test_point) const {

	double distance = std::numeric_limits<double>::infinity();
	int closest_point_index = -1;

	this -> kdt_points  -> closest_point_search(test_point,
		this -> kdt_points,
		closest_point_index,
		distance);

	return closest_point_index;

}



std::map<double,int > PC::get_closest_N_points(const arma::vec & test_point, 
	const unsigned int & N) const {

	std::map<double,int > closest_points;
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


	if (format_like_obj){

		if (save_normals){
			for (unsigned int vertex_index = 0;vertex_index < this -> size();++vertex_index) {
				arma::vec p = dcm * this -> get_point_coordinates(vertex_index) + x;
				shape_file << "v " << p(0) << " " << p(1) << " " << p(2) << std::endl;
			}

			for (unsigned int vertex_index = 0;vertex_index < this -> size();++vertex_index) {
				arma::vec n = dcm * this -> get_normal_coordinates(vertex_index);
				shape_file << "vn " << n(0) << " " << n(1) << " " << n(2) << std::endl;
			}
		}

		else{

			for (unsigned int vertex_index = 0;vertex_index < this -> size();++vertex_index) {
				arma::vec p = dcm * this -> get_point_coordinates(vertex_index) + x;
				shape_file << "v " << p(0) << " " << p(1) << " " << p(2) << std::endl;
			}
		}

	}
	else{

		for (unsigned int vertex_index = 0;vertex_index < this -> size();++vertex_index) {
			arma::vec p = dcm * this -> get_point_coordinates(vertex_index) + x;

			if (save_normals) {
				arma::vec n = dcm * this -> get_normal_coordinates(vertex_index);
				shape_file << p(0) << " " << p(1) << " " << p(2) << " " << n(0) << " " << n(1) << " " << n(2) << std::endl;
			}

			else {
				shape_file << p(0) << " " << p(1) << " " << p(2) << std::endl;
			}
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




void PC::construct_normals(arma::vec los_dir,unsigned int N) {

	#pragma omp parallel for if (USE_OMP_PC)
	for (unsigned int i = 0; i < this -> points_normals .  size(); ++i) {

		// Get the N nearest neighbors to this point
		auto closest_points = this -> get_closest_N_points(this -> get_point_coordinates(i), N);

		arma::mat::fixed<3,3> covariance = arma::zeros<arma::mat>(3,3);
		arma::vec::fixed<3> centroid = {0,0,0};

		for (auto it = closest_points.begin(); it != closest_points.end(); ++it) {
			centroid += this -> get_point_coordinates(it -> second)/closest_points.size();
		}

		for (auto it = closest_points.begin(); it != closest_points.end(); ++it) {
			auto p = this -> get_point_coordinates(it -> second);
			covariance += 1./(closest_points.size() - 1) * (p - centroid) * (p - centroid).t();
		}

		// The eigenvalue problem is solved
		arma::vec eigval;
		arma::mat eigvec;

		arma::eig_sym(eigval, eigvec, covariance);
		arma::vec n = arma::normalise(eigvec.col(arma::abs(eigval).index_min()).rows(0, 2));

		// The normal is flipped to make sure it is facing the los
		if (arma::dot(n, los_dir) < 0) {
			this -> points_normals .  at(i). set_normal_coordinates(n);
		}
		else {
			this -> points_normals .  at(i). set_normal_coordinates(-n);
		}
	}
}



void PC::construct_normals(arma::vec los_dir, double radius) {

	unsigned int size = this -> size();

	#pragma omp parallel for 
	for (unsigned int i = 0; i < size; ++i) {

		// Get the N nearest neighbors to this point
		auto closest_points = this -> get_nearest_neighbors_radius(this -> get_point_coordinates(i), radius);

		// arma::mat::fixed<3,3> covariance = arma::zeros<arma::mat>(3,3);
		// arma::vec::fixed<3> centroid = {0,0,0};

		// for (auto it = closest_points.begin(); it != closest_points.end(); ++it) {
		// 	centroid += (*it) -> get_point()/closest_points.size();
		// }

		// for (auto it = closest_points.begin(); it != closest_points.end(); ++it) {
		// 	auto p = (*it) -> get_point();
		// 	covariance += 1./(closest_points.size() - 1) * (p - centroid) * (p - centroid).t();
		// }

		// The eigenvalue problem is solved
		// arma::vec eigval;
		// arma::mat eigvec;

		// arma::eig_sym(eigval, eigvec, covariance);
		// arma::vec n = arma::normalise(eigvec.col(arma::abs(eigval).index_min()).rows(0, 2));

		// // The normal is flipped to make sure it is facing the los
		// if (arma::dot(n, los_dir) < 0) {
		// 	this -> points_normals .  at(i) -> set_normal_coordinates(n);
		// }
		// else {
		// 	this -> points_normals .  at(i) -> set_normal_coordinates(-n);
		// }
	}
}





arma::vec::fixed<3> PC::get_center() const{

	double c_x = 0;
	double c_y = 0;
	double c_z = 0;

	unsigned int size = this -> size();

	#pragma omp parallel for reduction(+:c_x,c_y,c_z)
	for (unsigned int i = 0; i < size; ++i) {

		arma::vec point = this -> get_point_coordinates(i);

		c_x += point(0) / size;
		c_y += point(1) / size;
		c_z += point(2) / size;

	}
	arma::vec center = {c_x,c_y,c_z};

	return center;
}


arma::mat::fixed<3,3> PC::get_principal_axes() const{

	arma::mat::fixed<3,3> P,E;
	P.fill(0);

	int size = (int)(this -> points_normals .  size());

	arma::vec::fixed<3> center = this -> get_center();

	for (unsigned int i = 0; i < size; ++i) {
		arma::vec point = this  -> get_point_coordinates(i);
		P += 1./(size - 1) * (point - center) * (point - center).t();
	}

	arma::vec eigval;
	arma::eig_sym( eigval, E, P );

	if (arma::det(E) < 0){
		E.col(0) *= -1;
	}

	return E;
}



// arma::vec PC::get_bbox_center() const{


// 	unsigned int size = this -> points_normals .  size();

// 	arma::vec bbox_min = this -> points_normals .  at(0) -> get_point();
// 	arma::vec bbox_max = this -> points_normals .  at(0) -> get_point();

// 	// #pragma omp parallel for reduction(min:bbox_x_min,bbox_y_min,bbox_z_min) reduction(max:bbox_x_max,bbox_y_max,bbox_z_max) if (USE_OMP_PC)
// 	for (unsigned int i = 0; i < size; ++i) {
// 		arma::vec point = this -> points_normals .  at(i) -> get_point();
// 		bbox_max = arma::max(bbox_max,point);
// 		bbox_min = arma::min(bbox_min,point);

// 	}

// 	arma::vec C = 0.5 * (bbox_max + bbox_min);


// 	return C;
// }

// arma::vec PC::get_bbox_dim() const{


// 	unsigned int size = this -> points_normals .  size();

// 	arma::vec bbox_max = this -> points_normals .  at(0) -> get_point();
// 	arma::vec bbox_min = this -> points_normals .  at(0) -> get_point();


// 	// #pragma omp parallel for reduction(min:bbox_x_min,bbox_y_min,bbox_z_min) reduction(max:bbox_x_max,bbox_y_max,bbox_z_max) if (USE_OMP_PC)
// 	for (unsigned int i = 0; i < size; ++i) {
// 		arma::vec point = this -> points_normals .  at(i) -> get_point();

// 		bbox_max = arma::max(bbox_max,point);
// 		bbox_min = arma::min(bbox_min,point);

// 	}


// 	return 0.5 *(bbox_max - bbox_min);
// }

// double PC::get_bbox_diagonal() const{

// 	unsigned int size = this -> points_normals .  size();

// 	arma::vec bbox_max = this -> points_normals .  at(0) -> get_point();
// 	arma::vec bbox_min = this -> points_normals .  at(0) -> get_point();

// 	// #pragma omp parallel for reduction(min:bbox_x_min,bbox_y_min,bbox_z_min) reduction(max:bbox_x_max,bbox_y_max,bbox_z_max) if (USE_OMP_PC)
// 	for (unsigned int i = 0; i < size; ++i) {
// 		arma::vec point = this -> points_normals .  at(i) -> get_point();

// 		bbox_max = arma::max(bbox_max,point);
// 		bbox_min = arma::min(bbox_min,point);

// 	}

// 	return arma::norm(bbox_max - bbox_min);

// }

std::string PC::get_label() const{
	return this -> label;
}



void PC::compute_feature_descriptors(int type,bool keep_correlations,int N_bins,double neighborhood_radius,std::string pc_name){


	std::vector<int> relevant_point_with_descriptors;
	unsigned int size = this -> points_normals .  size();


	// std::vector<double> radii = {neighborhood_radius,1.25 * neighborhood_radius,1.5 *neighborhood_radius,1.75 *  neighborhood_radius};

	std::vector<double> radii = {neighborhood_radius};


	throw(std::runtime_error("Not finished here"));

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
		std::cout << "Saving active features...\n";
		this -> save_active_features(index,pc_name);
		#endif

	}

	// Only the unique features are kept
	for (int i = 0 ; i < size; ++i){
		const PointNormal & query_point = this -> points_normals.at(i);
		if (query_point . get_is_valid_feature()) {
			relevant_point_with_descriptors.push_back(i);
		}
	}



		#if PC_DEBUG_FLAG
	std::cout << "Creating KDTree with active features...\n";
	#endif

	// this -> kdt_descriptors = std::make_shared<KDTreeDescriptors>(KDTreeDescriptors());
	// this -> kdt_descriptors = this -> kdt_descriptors  -> build(relevant_point_with_descriptors, 0);



}


void PC::save_active_features(int index,std::string pc_name) const{

	unsigned int size = this -> points_normals .  size();
	unsigned int histo_size = this -> points_normals .  at(0) . get_histogram_size();
	
	std::vector<PointNormal > active_features;

	arma::mat active_features_histograms = arma::zeros<arma::mat>(histo_size,size);

	arma::mat active_features_histograms_spfh = arma::zeros<arma::mat>(histo_size,size);

	for (unsigned int k  = 0; k < size; ++k){

		if(this -> points_normals .  at(k) .get_is_valid_feature()){
			active_features.push_back(this -> get_point(k));
			active_features_histograms.col(k) = this -> points_normals .  at(k) . get_descriptor_histogram();
			if (this -> get_point(0).get_descriptor_ptr() -> get_type() == 1)
				active_features_histograms_spfh.col(k) = this -> points_normals .  at(k).get_spfh_histogram();
		};

	}

	PC active_features_pc(active_features);

	active_features_pc.save(pc_name + "_active_features_" + std::to_string(index) + ".obj");

	this -> mean_feature_histogram.save(pc_name + "_mean_histogram_" + std::to_string(index) + ".txt",arma::raw_ascii);
	active_features_histograms.save(pc_name + "_active_features_histograms_"+ std::to_string(index) + ".txt",arma::raw_ascii);
	
	if (this -> get_point(0).get_descriptor_ptr() -> get_type() == 1)
		active_features_histograms_spfh.save(pc_name + "_active_features_histograms_spfh_"+ std::to_string(index) + ".txt",arma::raw_ascii);

}



void PC::compute_PFH(bool keep_correlations,int N_bins,double neighborhood_radius){

	// unsigned int size = this -> points_normals .  size();

	// if (neighborhood_radius < 0){
	// 	throw(std::runtime_error("neighborhood_radius is negative"));
	// }

	// #pragma omp parallel for
	// for (unsigned int i = 0; i < size; ++i) {

	// 	PointNormal & query_point = this -> points_normals.at(i);

	// 	std::vector<int> neighborhood;
	// 	auto query_point_neighborhood = query_point.get_neighborhood(neighborhood_radius);

	// 	for (int k = 0; k < query_point_neighborhood.size(); ++k ){
	// 		neighborhood.push_back(this -> get_point(query_point_neighborhood[k]));
	// 	}

	// 	query_point.set_descriptor(PFH(neighborhood,keep_correlations,N_bins));
	// }	


}



void PC::compute_FPFH(bool keep_correlations,int N_bins,double neighborhood_radius){

	unsigned int size = this -> points_normals .  size();

	// if (neighborhood_radius < 0){
	// 	throw(std::runtime_error("neighborhood_radius is negative"));
	// }

	// #pragma omp parallel for
	// for (unsigned int i = 0; i < size; ++i) {

	// 	std::shared_ptr<PointNormal> query_point = this -> points_normals .  at(i);

	// 	std::vector<std::shared_ptr<PointNormal> > neighborhood;
	// 	auto query_point_neighborhood = query_point -> get_neighborhood(neighborhood_radius);

	// 	for (int k = 0; k < query_point_neighborhood.size(); ++k ){
	// 		neighborhood.push_back(this -> get_point(query_point_neighborhood[k]));
	// 	}

	// 	query_point -> set_SPFH(SPFH(query_point,neighborhood,keep_correlations,N_bins));
	// }



	// #pragma omp parallel for
	// for (unsigned int i = 0; i < size; ++i) {
	// 	std::shared_ptr<PointNormal> query_point = this -> points_normals .  at(i);
	// 	std::vector<std::shared_ptr<PointNormal> > neighborhood;
	// 	auto query_point_neighborhood = query_point -> get_neighborhood(neighborhood_radius);

	// 	for (int k = 0; k < query_point_neighborhood.size(); ++k ){
	// 		neighborhood.push_back(this -> get_point(query_point_neighborhood[k]));
	// 	}

	// 	query_point -> set_descriptor(FPFH(query_point,neighborhood));
	// 	if (arma::max(query_point -> get_descriptor_histogram()) == 0){
	// 		query_point -> set_is_valid_feature(false);
	// 	}

	// }

}


void PC::save_point_descriptors(std::string path) const{
	unsigned int size = this -> size();
	unsigned int histo_size = this -> points_normals .  at(0).get_histogram_size();


	std::vector<arma::vec> active_features_histograms_arma;

	for (unsigned int i = 0; i < size; ++i) {

		// active_features_histograms.col(i) = this -> points_normals .  at(i) -> get_descriptor_histogram();
		auto histogram = this -> points_normals.at(i).get_descriptor_histogram();

		if (this -> points_normals.at(i).get_is_valid_feature()){
			active_features_histograms_arma.push_back(histogram);
		}

	}

	arma::mat active_features_histograms = arma::zeros<arma::mat>(histo_size,active_features_histograms_arma.size());

	std::cout << active_features_histograms_arma.size() << " features found to be unique in " << path << std::endl;

	for (unsigned int i = 0; i < active_features_histograms_arma.size(); ++i) {
		active_features_histograms.col(i) = active_features_histograms_arma[i];

	}

	active_features_histograms.save(path,arma::raw_ascii);

}



std::vector<PointPair>  PC::find_pch_matches_kdtree(const std::shared_ptr<PC> & pc_source,const std::shared_ptr<PC> & pc_destination){


// 	PointDescriptor descriptor;
	std::vector<PointPair> matches;	
// 	std::vector<PointPair> all_matched_pairs;

// 	std::set<std::shared_ptr< PointNormal > > used_destination_points;


// 	std::set<unsigned int> pc1_used_indices;
// 	std::map<std::shared_ptr< PointNormal >, std::map<double,std::shared_ptr<PointNormal> > >  pc0_to_pc1_potential_matches;


// 	int N_potential_matches = 10;
// 	int N_potential_neighbors = 20;
// 	int N_draws = 100;




// 	for (unsigned int i = 0; i < pc_source -> get_size(); ++i)  {	
// 		if (pc_source -> get_point(i) -> get_is_valid_feature()){

// 			auto closest_features = pc_destination -> get_closest_N_features(pc_source -> get_point(i),N_potential_matches);

// 			pc0_to_pc1_potential_matches[pc_source -> get_point(i)] = closest_features;
// 		}	
// 	}

// 	// We now have potential correspondances. Looking back at pc0_to_pc1_potential_matches, we need to find neighborhoods of features
// 	std::map<std::shared_ptr< PointNormal >, std::shared_ptr< PointNormal > > point_to_neighborhood_center;
// 	std::map<std::shared_ptr< PointNormal >, std::vector<std::shared_ptr< PointNormal > >  > neighborhood_center_to_points;

// 	for (auto it = pc0_to_pc1_potential_matches.begin(); it != pc0_to_pc1_potential_matches.end(); ++it){

// 		if (point_to_neighborhood_center.find(it -> first) == point_to_neighborhood_center.end()){

// 			point_to_neighborhood_center[it -> first] = it -> first;
// 			neighborhood_center_to_points[it -> first].push_back(it -> first);

// 			auto closest_neighbors = pc_source -> get_closest_N_points(it -> first -> get_point(),N_potential_neighbors);

// 			for (auto n_it = closest_neighbors.begin(); n_it != closest_neighbors.end(); ++n_it){
// 				if (point_to_neighborhood_center.find(n_it -> second) == point_to_neighborhood_center.end() && n_it -> second -> get_is_valid_feature() ){
// 					point_to_neighborhood_center[n_it -> second] = it -> first;
// 					neighborhood_center_to_points[it -> first].push_back(n_it -> second);
// 				}
// 			}

// 		}

// 	}

// 	// The match geometry is optimized in each neighborhood
// 	for (auto it = neighborhood_center_to_points.begin(); it != neighborhood_center_to_points.end(); ++it){

// 		#if PC_DEBUG_FLAG
// 		std::cout << "\nNeighborhood size: "  << it -> second.size() << std::endl;
// 		#endif

// 		if (it -> second.size() < 2){
// 			#if PC_DEBUG_FLAG
// 			std::cout << "\nSkipping" << std::endl;
// 			#endif
// 			continue;
// 		}

// 		std::vector<PointPair> best_correspondance_table;

// 		for (int i = 0; i < it -> second.size(); ++i ){

// 			std::shared_ptr<PointNormal> point_in_neighborhood = it -> second[i];

// 			std::map<double,std::shared_ptr<PointNormal> > potential_matches = pc0_to_pc1_potential_matches[point_in_neighborhood];
// 			best_correspondance_table.push_back(std::make_pair(point_in_neighborhood,potential_matches.begin() -> second));


// 		}

// 		#if PC_DEBUG_FLAG
// 		std::cout << "Optimizing geometry in neighborhood " << std::distance(neighborhood_center_to_points.begin(),it) << " / " << neighborhood_center_to_points.size() << std::endl;
// 		#endif


// 		double best_log_likelihood = PC::compute_neighborhood_consensus_ll(best_correspondance_table);

// 		#if PC_DEBUG_FLAG
// 		std::cout << "Starting with L = " << best_log_likelihood << std::endl;
// 		#endif

// 		// The optimal correspondance table is obtained by brute-forcing the correspondances
// 		// from this neighborhood to the destination point cloud
// 		for (int i = 0; i < N_draws; ++i){

// 		// Generate random correspondance table

// 			#if PC_DEBUG_FLAG
// 			std::cout << "\n\tDraw " << i + 1<< " / "  << N_draws  << "... \n";
// 			std::cout << "\tGenerating random correspondance table ... \n";
// 			#endif


// 			auto point_in_neighborhood = *(it -> second.begin());
// 			auto potential_matches = pc0_to_pc1_potential_matches.at(point_in_neighborhood);

// 			auto correspondance_table = PC:: generate_random_correspondance_table(it -> second,pc0_to_pc1_potential_matches);

// 			// Evaluate the log-likelihood at this correspondance table

// 			#if PC_DEBUG_FLAG
// 			std::cout << "\tComputing log-likelihood... \n";
// 			#endif

// 			double log_likelihood = PC::compute_neighborhood_consensus_ll(correspondance_table);

// 			#if PC_DEBUG_FLAG
// 			std::cout << "\tLog-likelihood: " << log_likelihood << " \n";
// 			#endif

// 			if (log_likelihood > best_log_likelihood){
// 				best_log_likelihood = log_likelihood;
// 				best_correspondance_table = correspondance_table;

// 				#if PC_DEBUG_FLAG
// 				std::cout << "\tNew optimal ll: " << best_log_likelihood << std::endl;
// 				#endif

// 			}
// 		}

// 		for (int k = 0; k < best_correspondance_table.size(); ++k){
// 			best_correspondance_table[k].first -> set_match(best_correspondance_table[k].second.get());
// 			best_correspondance_table[k].second -> set_match(best_correspondance_table[k].first.get());

// 			matches.push_back(best_correspondance_table[k]);
// 		}

// 	}

// 	#if PC_DEBUG_FLAG
// 	std::cout << "Total number of features left after optimization: " << matches.size() << std::endl;
// 	#endif



	return matches;

}



void PC::find_N_closest_pch_matches_kdtree(const std::shared_ptr<PC> & pc_source,
	const std::shared_ptr<PC> & pc_destination,const int N_closest_matches,
	std::vector< int > & active_source_points,
	std::map<int , std::vector<int > > & possible_matches){

	for (unsigned int i = 0; i < pc_source -> size(); ++i)  {	

		if (pc_source -> get_point(i).get_is_valid_feature()){
			active_source_points.push_back(static_cast<int>(i));
			auto closest_features = pc_destination -> get_closest_N_features(pc_source -> get_point_coordinates(i),N_closest_matches);

			std::vector<int> closest_features_points;

			for (auto it = closest_features.begin(); it != closest_features.end(); ++it){
				closest_features_points.push_back(it -> second);
			}

			possible_matches[i] = closest_features_points;
		}	

	}



}







PC::PC(std::vector< std::shared_ptr<PC> > & pcs,int points_retained){

	this -> points_normals.clear();
	double downsampling_factor = 1;
	int N_points_total = 0;
	for (unsigned int i = 0; i < pcs.size();++i){

		N_points_total += pcs[i] -> size();
	}

	if (points_retained > 0){
		downsampling_factor = double(points_retained) / N_points_total;
	}

	for (unsigned int i = 0; i < pcs.size();++i){

		arma::uvec random_order =  arma::regspace< arma::uvec>(0,  pcs[i] -> size() - 1);		
		random_order = arma::shuffle(random_order);	

		int points_to_keep = (int)	(downsampling_factor *  pcs[i] -> size());

		for (unsigned int p = 0; p < points_to_keep; ++p){
			this -> points_normals.push_back(pcs[i] -> get_point(random_order(p)));
			this -> points_normals.back().set_global_index((int)(this -> points_normals.size()) - 1);
		}

	}
	

}





// double PC::compute_neighborhood_consensus_ll(const std::vector<PointPair> & correspondance_table){


// 	arma::vec::fixed<2> mean_angles = {0,0};
// 	arma::mat::fixed<2,2> covariance = arma::zeros<arma::mat>(2,2);
// 	std::vector<arma::vec> angles_distribution;

// 	// Computing the angle distribution
// 	for (int i = 0 ; i < correspondance_table.size(); ++i){
// 		auto pair = correspondance_table[i];
// 		arma::vec pair_direction = arma::normalise(pair.second -> get_point() - pair.first -> get_point() );
// 		double alpha = std::atan2(pair_direction(1),pair_direction(0));
// 		double beta = std::atan2(pair_direction(2),arma::norm(pair_direction.subvec(0,1)));
// 		arma::vec angles = {alpha,beta};
// 		mean_angles += angles;
// 		angles_distribution.push_back(angles);

// 	}

// 	// Computing the mean/covariance of the distribution
// 	mean_angles = mean_angles / angles_distribution.size();

// 	for (int k = 0; k < angles_distribution.size(); ++k){
// 		covariance += 1./(angles_distribution.size() - 1) * (angles_distribution[k] - mean_angles) * (angles_distribution[k] - mean_angles).t();
// 	}


// 	// Evaluating the log-likelihood of the distribution
// 	double ll = 0;

// 	for (int k = 0; k < angles_distribution.size(); ++k){
// 		ll += - 0.5 * std::log(2 * arma::datum::pi * std::abs(arma::det(covariance)))  - 0.5 * arma::dot(angles_distribution[k] - mean_angles,arma::inv(covariance)*(angles_distribution[k] - mean_angles));
// 	}

// 	return ll;

// }

// std::vector<PointPair> PC::generate_random_correspondance_table(const std::vector<std::shared_ptr< PointNormal > > & neighborhood,
// 	const std::map<std::shared_ptr< PointNormal >, std::map<double,std::shared_ptr<PointNormal> > > & pc0_to_pc1_potential_matches){

// 	std::vector<PointPair> pairs;

// 	for (int i = 0; i < neighborhood.size(); ++i ){

// 		auto point_in_neighborhood = neighborhood[i];

// 		auto potential_matches = pc0_to_pc1_potential_matches.at(point_in_neighborhood);

// 		arma::ivec random_integer = arma::randi(1,arma::distr_param(0,potential_matches.size() - 1));
// 		int random_index = random_integer(0);

// 		pairs.push_back(std::make_pair(point_in_neighborhood,
// 			std::next(potential_matches.begin(),random_index) -> second));

// 	}


// 	return pairs;
// }










// void PC::save_pch_matches(const std::multimap<double,std::pair<int,int> > matches,std::string path){


// 	arma::mat matches_mat(matches.size(), 3);
// 	int count = 0;
// 	for (auto it = matches.begin();it != matches.end(); ++it ){

// 		matches_mat(count,0)= it -> first;
// 		matches_mat(count,1)= it -> second.first;
// 		matches_mat(count,2)= it -> second.second;

// 		++count;
// 	}

// 	matches_mat.save(path,arma::raw_ascii);

// }


// std::vector< std::shared_ptr<PointNormal> > * PC::get_points_with_features() const{
// 	return this -> kdt_descriptors -> get_points_with_descriptors();
// }


// std::shared_ptr<PointNormal> PC::get_closest_feature(std::shared_ptr<PointNormal> other_point,double & distance) const{

// 	distance = std::numeric_limits<double>::infinity();
// 	std::shared_ptr<PointNormal> closest_point;

// 	this -> kdt_descriptors  -> closest_point_search(other_point,
// 		this -> kdt_descriptors,
// 		closest_point,
// 		distance);

// 	return closest_point;

// }



std::map<double,int > PC::get_closest_N_features(const arma::vec & histogram,const int & N) const{

	double distance = std::numeric_limits<double>::infinity();
	std::map<double,int > closest_points_with_features;

	// this -> kdt_descriptors  -> closest_N_point_search(other_point,
	// 	N,
	// 	this -> kdt_descriptors,
	// 	distance,
	// 	closest_points_with_features);

	return closest_points_with_features;

}


void PC::compute_mean_feature_histogram(){

	unsigned int size = this -> points_normals .  size();
	unsigned int histo_size = this -> get_point(0). get_histogram_size();

	this -> mean_feature_histogram = arma::zeros<arma::vec>(histo_size);

	for (unsigned int k = 0; k < size; ++k){
		this -> mean_feature_histogram += this -> get_point(k).get_descriptor_histogram();
	}

	this -> mean_feature_histogram = this -> mean_feature_histogram / arma::max(this -> mean_feature_histogram) * 100;

}

void PC::prune_features() {

	// The distance between each point histogram and the mean histogram is computed
	// unsigned int size = this -> points_normals .  size();
	// auto all_points = this -> kdt_points  -> get_points_normals();
	// arma::vec distances = arma::zeros<arma::vec>(size);

	// #pragma omp parallel for
	// for (unsigned int k = 0; k < size; ++k){

	// 	distances(k) = all_points -> at(k) -> features_similarity_distance(this -> mean_feature_histogram);

	// }

	// distances = arma::abs(distances - arma::mean(distances)) / arma::stddev(distances);

	// // Points whose feature descriptor is less than 1.25 standard deviations away from the mean are considered as 
	// // inliers, thus irrelevant as discriminative features

	// unsigned int relevant_features_count = 0;

	// for (unsigned int k = 0; k < size; ++k){
	// 	if (distances(k) < 1.75){
	// 		all_points -> at(k) -> set_is_valid_feature(false);
	// 	}

	// 	if (all_points -> at(k) -> get_is_valid_feature()){
	// 		++ relevant_features_count;
	// 	}
	// }

	// #if PC_DEBUG_FLAG
	// std::cout << "Keeping " << relevant_features_count << " non-redundant features from the original " <<all_points -> size() << " ones \n";
	// #endif
}




// void PC::compute_neighborhoods(double radius){
// 	int size = this -> size();
// 	// First, the neighborhood of each point is computed using the largest provided radius

// 	#pragma omp parallel for
// 	for (int i = 0 ; i < size; ++i){
// 		auto p = this -> kdt_points  -> get_normal_coordinates(i);
// 		p -> set_neighborhood(this -> get_points_in_sphere(p -> get_point(),radius));
// 	}

// }


/**
	Returns queried point
	@param index Index of the queried point
	@return queried point
	*/
const PointNormal & PC::get_point(unsigned int index) const{
	return this -> points_normals[index];
}

	/**
	Returns queried point coordinates
	@param index Index of the queried point
	@return queried point coordinates
	*/
arma::vec PC::get_point_coordinates(unsigned int index) const{
	return this -> points_normals[index].get_point_coordinates();
}


arma::vec PC::get_normal_coordinates(unsigned int index) const{
	return this -> points_normals[index].get_normal_coordinates();
}

unsigned int PC::size() const{
	return this -> points_normals.size();
}

void PC::build_kd_tree_points(){
	std::vector<int> indices;
	for (int i =0; i < this -> size(); ++i){
		indices.push_back(i);
	}

	this -> kdt_points = std::make_shared<KDTree<PC>>(KDTree<PC>(this));
	this -> kdt_points -> build(indices,0);
}


std::vector<int> PC::get_nearest_neighbors_radius(const arma::vec & test_point, const double & radius) const{
	std::vector< int > neighbors_indices;
	this -> kdt_points -> radius_point_search(test_point,this -> kdt_points,radius,neighbors_indices);
	return neighbors_indices;
}



