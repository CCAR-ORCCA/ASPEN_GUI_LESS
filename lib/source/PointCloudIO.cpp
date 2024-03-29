#include <PointCloudIO.hpp>
#include <PointNormal.hpp>
#include <PointDescriptor.hpp>

template<>
void PointCloudIO<PointNormal>::save_to_obj(
	const PointCloud<PointNormal> & pc, 
	std::string savepath,
	const arma::mat::fixed<3,3> & dcm, 
	const arma::vec::fixed<3> & x){


	std::ofstream shape_file;
	shape_file.open(savepath);

	for (unsigned int vertex_index = 0;vertex_index < pc.size();++vertex_index) {
		arma::vec::fixed<3> p = dcm * pc.get_point_coordinates(vertex_index) + x;
		shape_file << "v " << p(0) << " " << p(1) << " " << p(2) << std::endl;
	}	

	// Only save the normals if they have been calculated
	if (pc.get_point(0).get_normal_coordinates().size() > 0){
		for (unsigned int vertex_index = 0;vertex_index < pc.size();++vertex_index) {
			arma::vec::fixed<3> n = dcm * pc.get_point(vertex_index).get_normal_coordinates();
			shape_file << "vn " << n(0) << " " << n(1) << " " << n(2) << std::endl;
		}
	}
	

}

template<>
void PointCloudIO<PointNormal>::save_to_txt(const PointCloud<PointNormal> & pc, 
	std::string savepath,const arma::mat::fixed<3,3> & dcm, const arma::vec::fixed<3> & x){

	std::ofstream shape_file;
	shape_file.open(savepath);

	if (pc.get_point(0).get_normal_coordinates().size() > 0){
		for (unsigned int vertex_index = 0;vertex_index < pc.size();++vertex_index) {
			arma::vec::fixed<3> p = dcm * pc.get_point_coordinates(vertex_index) + x;
			arma::vec::fixed<3> n = dcm * pc.get_point(vertex_index).get_normal_coordinates();
			shape_file << p(0) << " " << p(1) << " " << p(2) << " " << n(0) << " " << n(1) << " " << n(2) << std::endl;
		}
	}
	else{
		for (unsigned int vertex_index = 0;vertex_index < pc.size();++vertex_index) {
			arma::vec::fixed<3> p = dcm * pc.get_point_coordinates(vertex_index) + x;
			shape_file << p(0) << " " << p(1) << " " << p(2) << std::endl;
		}
	}


}

template<>
void PointCloudIO<PointDescriptor>::save_to_obj(const PointCloud<PointDescriptor> & pc, 
	std::string savepath,const arma::mat::fixed<3,3> & dcm, const arma::vec::fixed<3> & x){

	throw(std::runtime_error("Saving a point cloud of feature descriptors to obj makes no sense"));
}

template<>
void PointCloudIO<PointDescriptor>::save_to_txt(const PointCloud<PointDescriptor> & pc, 
	std::string savepath,const arma::mat::fixed<3,3> & dcm, const arma::vec::fixed<3> & x){

	std::ofstream feature_file;
	feature_file.open(savepath);

	for (unsigned int index = 0;index < pc.size();++index) {
		const arma::vec & hist = pc.get_point(index).get_histogram();
		int feature_size = hist.size();

		for (int i = 0; i < feature_size; ++i){
			if (i < feature_size - 1){
				feature_file << hist(i) << " ";
			}
			else{
				feature_file << hist(i) << "\n";
			}
		}
	}


}


template <class T>
void PointCloudIO<T>::save_active_features_positions(
	const PointCloud<PointNormal> & pc_points,
	const PointCloud<PointDescriptor> & pc_features, 
	std::string savepath,
	const arma::mat::fixed<3,3> & dcm, 
	const arma::vec::fixed<3> & x){

	assert(pc_points.size() == pc_features.size());
	PointCloud<PointNormal> active_points_pc;
	for (int i = 0; i < pc_features.size(); ++i){
		if (pc_features.get_point(i).get_is_valid_feature()){
			active_points_pc.push_back(pc_points.get_point(i));
		}
	}
	
	PointCloudIO<PointNormal>::save_to_obj(active_points_pc,savepath,dcm,x);

}



template class PointCloudIO<PointDescriptor>;
template class PointCloudIO<PointNormal>;