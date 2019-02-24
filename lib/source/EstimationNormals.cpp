#include <EstimationNormals.hpp>
#include <PointNormal.hpp>

template <class T,class U>
EstimationNormals<T,U>::EstimationNormals(const PointCloud<T> & input_pc,PointCloud<U> & output_pc) : EstimationFeature<T,U>(input_pc,output_pc){

}


template <>
void EstimationNormals<PointNormal,PointNormal>::estimate(int N_neighbors,bool force_use_previous){
	
	#pragma omp parallel for 
	for (unsigned int i = 0; i < this -> input_pc .  size(); ++i) {

		// Get the N nearest neighbors to this point
		auto closest_points = this -> input_pc.get_closest_N_points(this -> input_pc.get_point_coordinates(i), N_neighbors);

		arma::mat::fixed<3,3> covariance = arma::zeros<arma::mat>(3,3);
		arma::vec::fixed<3> centroid = {0,0,0};

		for (auto it = closest_points.begin(); it != closest_points.end(); ++it) {
			centroid += this -> input_pc.get_point_coordinates(it -> second)/closest_points.size();
		}

		for (auto it = closest_points.begin(); it != closest_points.end(); ++it) {
			const arma::vec & p = this -> input_pc.get_point_coordinates(it -> second);
			covariance += (p - centroid) * (p - centroid).t();
		}
		covariance *= 1./(closest_points.size() - 1) ;
		
		// The eigenvalue problem is solved
		arma::vec eigval;
		arma::mat eigvec;

		arma::eig_sym(eigval, eigvec, covariance);
		arma::vec n = arma::normalise(eigvec.col(arma::abs(eigval).index_min()).rows(0, 2));

		// The normal is flipped to make sure it is facing the los
		// or that it is consistently oriented with a previously computed normal
		if (force_use_previous){
			if (arma::dot(n, this -> output_pc.get_point(i).get_normal_coordinates()) < 0) {
				this -> output_pc.get_point(i).set_normal_coordinates(n);
			}
			else {
				this -> output_pc.get_point(i).set_normal_coordinates(-n);
			}
		}
		else{
			if (arma::dot(n, this -> los_dir) < 0) {
				this -> output_pc.get_point(i).set_normal_coordinates(n);
			}
			else {
				this -> output_pc.get_point(i).set_normal_coordinates(-n);
			}
		}


	}
	
}


template <>
void EstimationNormals<PointNormal,PointNormal>::estimate(double radius,bool force_use_previous){
	
	#pragma omp parallel for 
	for (unsigned int i = 0; i < this -> input_pc .  size(); ++i) {

		// Get the N nearest neighbors to this point

		auto closest_points = this -> input_pc.get_nearest_neighbors_radius(this -> input_pc.get_point_coordinates(i), radius);

		arma::mat::fixed<3,3> covariance = arma::zeros<arma::mat>(3,3);
		arma::vec::fixed<3> centroid = {0,0,0};

		int size = closest_points.size();
		for (auto it = closest_points.begin(); it != closest_points.end(); ++it) {
			centroid += this -> input_pc.get_point_coordinates(*it)/size;
		}

		for (auto it = closest_points.begin(); it != closest_points.end(); ++it) {
			const arma::vec & p = this -> input_pc.get_point_coordinates(*it);
			covariance += 1./(size - 1) * (p - centroid) * (p - centroid).t();
		}

		// The eigenvalue problem is solved
		arma::vec eigval;
		arma::mat eigvec;

		arma::eig_sym(eigval, eigvec, covariance);
		arma::vec n = arma::normalise(eigvec.col(arma::abs(eigval).index_min()).rows(0, 2));

		// The normal is flipped to make sure it is facing the los
		if (force_use_previous){

			if (arma::dot(n, this -> output_pc.get_point(i).get_normal_coordinates()) < 0) {
				this -> output_pc.get_point(i).set_normal_coordinates(n);
			}
			else {
				this -> output_pc.get_point(i).set_normal_coordinates(-n);
			}

		}
		else{
			if (arma::dot(n, this -> los_dir) < 0) {
				this -> output_pc.get_point(i).set_normal_coordinates(n);
			}
			else {
				this -> output_pc.get_point(i).set_normal_coordinates(-n);
			}
		}


	}
	
}


template <class T,class U>
void EstimationNormals<T,U>::set_los_dir(const arma::vec::fixed<3> & los_dir){
	this -> los_dir = los_dir;
}








template class EstimationNormals<PointNormal,PointNormal>;
