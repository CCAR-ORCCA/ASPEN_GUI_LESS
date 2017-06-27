#include "PC.hpp"



PC::PC(arma::vec los_dir, std::vector<std::vector<std::shared_ptr<Ray> > > * focal_plane) {

	std::vector<Ray * > valid_measurements;

	// The valid measurements used to form the point cloud are extracted
	for (unsigned int y_index = 0; y_index < focal_plane -> size(); ++y_index) {


		// column by column
		for (unsigned int z_index = 0; z_index < focal_plane -> at(0).size(); ++z_index) {

			if (focal_plane -> at(y_index)[z_index] -> get_true_range() < std::numeric_limits<double>::infinity()) {

				valid_measurements.push_back(focal_plane -> at(y_index)[z_index].get());

			}

		}
	}

	// The points are created

	arma::

}

