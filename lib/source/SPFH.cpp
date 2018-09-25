#include "SPFH.hpp"
#include "PointNormal.hpp"

#include <armadillo>


SPFH::SPFH() : PointDescriptor(){}

SPFH::SPFH(const int & query_point,
	const std::vector<int > & points,
	int N_bins,
	const PointCloud<PointNormal> & pc) : PointDescriptor(){

	arma::vec::fixed<3> u,v,w,p_i,p_j,n_j;

	int N_global_bins = 3 * N_bins;
	

	this -> histogram = arma::zeros<arma::vec>(N_global_bins);
	this -> distance_to_closest_neighbor = std::numeric_limits<double>::infinity();

	int non_trivial_neighbors_count = 0;

	
	for (int j = 0; j < points.size(); ++j){
		double distance = arma::norm(pc.get_point_coordinates(j) - pc.get_point_coordinates(query_point));

		if (distance > 0){

			int alpha_bin_index,phi_bin_index,theta_bin_index;

			this -> distance_to_closest_neighbor = std::min(this-> distance_to_closest_neighbor,distance);

			this -> compute_darboux_frames_local_hist( 
				alpha_bin_index,
				phi_bin_index,
				theta_bin_index, 
				N_bins,
				pc.get_point_coordinates(query_point),
				pc.get_point(query_point).get_normal_coordinates(),
				pc.get_point_coordinates(points.at(j)),
				pc.get_point(points.at(j)).get_normal_coordinates());

			this -> histogram(theta_bin_index) += 1.;
			this -> histogram(N_bins + alpha_bin_index) += 1.;
			this -> histogram(2 * N_bins + phi_bin_index) += 1.;
			



			++non_trivial_neighbors_count;

		}

	}

	assert(non_trivial_neighbors_count + 1 == static_cast<int>(points.size()));

	this -> histogram *= 100./non_trivial_neighbors_count;

}

double SPFH::get_distance_to_closest_neighbor() const{
	return this -> distance_to_closest_neighbor;
}
