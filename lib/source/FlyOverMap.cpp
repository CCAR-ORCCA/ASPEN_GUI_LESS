#include "FlyOverMap.hpp"


FlyOverMap::FlyOverMap(int n_bins_longitude,int n_bins_latitude){
	this -> n_bins_longitude = n_bins_longitude;
	this -> n_bins_latitude = n_bins_latitude;

	this -> d_bin_longitude = 360./ this -> n_bins_longitude;
	this -> d_bin_latitude = 180./ this -> n_bins_latitude;

	for (int i = 0; i < this -> n_bins_latitude; ++i){

		std::vector< std::vector < int > > row;

		for (int j = 0; j <this -> n_bins_longitude; ++j){
			std::vector < int > empty_vector;
			row.push_back(empty_vector);
		}
		this -> bins.push_back(row);

	}
}

void FlyOverMap::add_label(int label, double longitude, double latitude){

	int bin_longitude = int(longitude / this -> d_bin_longitude) + this -> n_bins_longitude/2;
	int bin_latitude = int(latitude / this -> d_bin_latitude) + this -> n_bins_latitude/2;
	this -> bins[this -> n_bins_latitude -  bin_latitude - 1][ bin_longitude].push_back(label);
	this -> labels[label] = std::make_pair(this -> n_bins_latitude -  bin_latitude - 1,bin_longitude);
}

void FlyOverMap::update_label(int label,double longitude, double latitude){

	// Cleaning up the old bin
	std::pair<int,int> old_bin_coordinates = this -> labels[label];
	std::vector<int> old_bin = this -> bins[old_bin_coordinates.first][old_bin_coordinates.second];
	
	old_bin.erase(std::find(old_bin.begin(), old_bin.end(), label));
	this -> bins[old_bin_coordinates.first][old_bin_coordinates.second] = old_bin;

	// Moving to the new bin
	int new_bin_longitude = int(longitude / this -> d_bin_longitude) + this -> n_bins_longitude/2;
	int new_bin_latitude = int(latitude / this -> d_bin_latitude) + this -> n_bins_latitude/2;

	this -> bins[this -> n_bins_latitude -  new_bin_latitude - 1][ new_bin_longitude].push_back(label);
	this -> labels[label] = std::make_pair(this -> n_bins_latitude -  new_bin_latitude - 1,new_bin_longitude);


	if ((old_bin_coordinates.first != this -> n_bins_latitude -  new_bin_latitude - 1) || (old_bin_coordinates.second != new_bin_longitude )){
		std::cout << "Updated label " << label << std::endl;
	}
}

std::vector<int> FlyOverMap::get_bin(int bin_longitude,int bin_latitude )const {
	return this -> bins[this -> n_bins_latitude - bin_latitude - 1][bin_longitude];
}


int FlyOverMap::get_bin_depth(int bin_longitude,int bin_latitude ) const {
	return this -> bins[this -> n_bins_latitude - bin_latitude - 1][bin_longitude].size();
}

void FlyOverMap::get_flyovers_in_bin(int bin_longitude,int bin_latitude,
	std::set<std::set<int> > & flyovers) const {

	std::vector < int > pc_in_bin = this -> get_bin(bin_longitude,bin_latitude );

	while(pc_in_bin.size() > 1){
		for (int pc_index = 0; pc_index < pc_in_bin.size() - 1; ++pc_index){

			if (std::abs(pc_in_bin.back() - pc_in_bin[pc_index]) != 1){
				std::set<int> new_pair;
				new_pair.insert(pc_in_bin.back());
				new_pair.insert(pc_in_bin[pc_index]);

				flyovers.insert(new_pair);
			}

		}
		pc_in_bin.pop_back();
	}

}

bool FlyOverMap::has_flyovers(double longitude, double latitude) const {

	int bin_longitude = int(longitude / this -> d_bin_longitude) + this -> n_bins_longitude/2;
	int bin_latitude = int(latitude / this -> d_bin_latitude) + this -> n_bins_latitude/2;
	
	std::set<std::set<int> > flyovers;
	this -> get_flyovers_in_bin(bin_longitude,bin_latitude,flyovers );

	return (flyovers.size() > 0) ;
}


std::set<std::set< int> > FlyOverMap::get_flyovers() const{

	std::set<std::set< int> > flyovers;
	for (int bin_longitude = 0; bin_longitude < this ->n_bins_longitude; ++bin_longitude){

		for (int bin_latitude = 0; bin_latitude < this -> n_bins_latitude; ++bin_latitude){

			this -> get_flyovers_in_bin(bin_longitude,bin_latitude,flyovers);
		}
	}

	return flyovers;
}










