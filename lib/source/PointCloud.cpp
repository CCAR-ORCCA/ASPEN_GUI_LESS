#include <PointCloud.hpp>
#include <PointNormal.hpp>
#include <KDTree.hpp>
#include <Ray.hpp>

#define PC_DEBUG_FLAG 1


template <class PointType> PointCloud<PointType>::PointCloud(){
}

template <class PointType> PointCloud<PointType>::PointCloud(int size){
this -> points.resize(size);
}



template <>
PointCloud<PointNormal>::PointCloud(std::vector<std::shared_ptr<Ray> > * focal_plane){

	for (int i = 0; i < focal_plane -> size(); ++i){

		if (focal_plane -> at(i) -> get_hit_element() >= 0){

			arma::vec::fixed<3> impact_point = focal_plane -> at(i) -> get_impact_point();

			PointNormal point(impact_point,this -> points.size());
			this -> points.push_back(point);

		}

	}

}




template <class PointType> 
PointCloud<PointType>::PointCloud(const std::vector< PointType > & points) {

	this -> points.clear();
	// The valid measurements used to form the point cloud are extracted
	for (unsigned int i = 0; i < points.size(); ++i) {

		this -> points[i].set_global_index(i);
		this -> points.push_back(points[i]);

	}

}

template <class PointType> int PointCloud<PointType>::get_closest_point(const arma::vec & test_point) const {

double distance = std::numeric_limits<double>::infinity();
int closest_point_index = -1;

this -> kdt  -> closest_point_search(test_point,
	this -> kdt,
	closest_point_index,
	distance);

return closest_point_index;

}


template <class PointType> std::map<double,int > PointCloud<PointType>::get_closest_N_points(const arma::vec & test_point, 
const unsigned int & N) const {

	std::map<double,int > closest_points;
	double distance = std::numeric_limits<double>::infinity();

	this -> kdt -> closest_N_point_search(test_point,N,this -> kdt,distance,closest_points);

	return closest_points;

}


template <class PointType> PointCloud<PointType>::PointCloud(std::vector< std::shared_ptr<PointCloud< PointType > > > & pcs,int points_retained){

this -> points.clear();
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
		this -> points.push_back(pcs[i] -> get_point(random_order(p)));
		this -> points.back().set_global_index((int)(this -> points.size()) - 1);
	}

}


}


template <class PointType> const PointType & PointCloud<PointType>::get_point(unsigned int index) const{
return this -> points[index];
}

template <class PointType>  PointType & PointCloud<PointType>::get_point(unsigned int index) {
return this -> points[index];
}


template <class PointType>  unsigned int PointCloud<PointType>::size() const{
return this -> points.size();
}


template <class PointType> std::vector<int> PointCloud<PointType>::get_nearest_neighbors_radius(const arma::vec & test_point, const double & radius) const{
std::vector< int > neighbors_indices;
this -> kdt -> radius_point_search(test_point,this -> kdt,radius,neighbors_indices);
return neighbors_indices;
}


template <class PointType> void PointCloud<PointType>::push_back(const PointType & point){
this -> points.push_back(point);
}

template <> const arma::vec & PointCloud<PointNormal>::get_point_coordinates(int i) const{
return this -> points[i].get_point_coordinates();
}


template <> const arma::vec & PointCloud<PointDescriptor>::get_point_coordinates(int i) const{
return this -> points[i].get_histogram();
}


template <> const arma::vec & PointCloud<PointNormal>::get_normal_coordinates(int i) const{
return this -> points[i].get_normal_coordinates();
}


template <> 
const arma::vec & PointCloud<PointDescriptor>::get_normal_coordinates(int i) const{
	throw(std::runtime_error("PointCloud<PointDescriptor>::get_normal_coordinates(int i) is not defined"));
	// The following will never be returned, it's merely to silence a compiler warning
	return this -> points[i].get_histogram();
}




template <> 
PointCloud<PointNormal>::PointCloud(std::string filename){

	std::cout << "Reading " << filename << std::endl;

	std::ifstream ifs(filename);

	if (!ifs.is_open()) {
		std::cout << "There was a problem opening the input file!\n";
		throw;
	}

	std::string line;
	std::vector<arma::vec::fixed<3>> vertices,normals;
	std::vector<std::vector<unsigned int> > shape_patch_indices;

	while (std::getline(ifs, line)) {

		std::stringstream linestream(line);


		std::string type;
		linestream >> type;

		if (type == "#" || type == "s"  || type == "o" || type == "m" || type == "u" || line.size() == 0) {
			continue;
		}

		else if (type == "v") {
			double vx, vy, vz;
			linestream >> vx >> vy >> vz;
			arma::vec::fixed<3> vertex = {vx, vy, vz};
			vertices.push_back(vertex);

		}
		else if (type == "vn"){
			double nx,ny,nz;
			linestream >> nx >> ny >> nz;
			arma::vec::fixed<3> normal = {nx, ny, nz};

			normals.push_back(normal);

		}

		else {
			throw(std::runtime_error(" unrecognized string in input file : "  + type));
		}

	}

	this -> points.clear();
	if (normals.size()!= 0){
		assert(normals.size() == vertices.size());
		for (unsigned int index = 0; index < vertices.size(); ++index) {
			this -> points.push_back(PointNormal(vertices[index],normals[index],index));
		}
	}
	else{
		for (unsigned int index = 0; index < vertices.size(); ++index) {
			this -> points.push_back(PointNormal(vertices[index],index));
		}
	}


}


template <class PointType> 
void PointCloud<PointType>::build_kdtree(bool verbose){

	auto start = std::chrono::system_clock::now();


	std::vector<int> indices;
	for (int i =0; i < this -> size(); ++i){
		if (this -> check_if_point_valid(i)){
			indices.push_back(i);
		}
	}

	this -> kdt = std::make_shared< KDTree<PointCloud,PointType> >(KDTree< PointCloud,PointType> (this));
	this -> kdt -> build(indices,0);

	auto end = std::chrono::system_clock::now();

	if (verbose){
		std::chrono::duration<double> elapsed_seconds = end-start;
		std::cout << "- Time elapsed building PointCloud kdtree: " << elapsed_seconds.count()<< " (s)"<< std::endl;
	}
	
}


template <class PointType> 
PointType & PointCloud<PointType>::operator[] (const int index){
	return this -> points[index];
}

template <>
void PointCloud<PointNormal>::transform(const arma::mat::fixed<3,3> & dcm,const arma::vec::fixed<3> & x){

	// The valid measurements used to form the point cloud are extracted
	#pragma omp parallel for
	for (unsigned int i = 0; i < this -> size(); ++i) {
		PointNormal & p = this -> points.at(i);
		p.set_point_coordinates(dcm * p. get_point_coordinates() + x);
		p.set_normal_coordinates(dcm * p. get_normal_coordinates());
	}

	// The KDTree is rebuilt

	this -> build_kdtree();

}


template <>
bool PointCloud<PointNormal>::check_if_point_valid(int i) const{
	return true;
}


template <>
bool PointCloud<PointDescriptor>::check_if_point_valid(int i) const{
	return this -> points[i].get_is_valid_feature();
}


template class PointCloud<PointNormal>;
template class PointCloud<PointDescriptor>;


