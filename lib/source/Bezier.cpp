#include "Bezier.hpp"
#include "Psopt.hpp"
#include <cassert>
Bezier::Bezier(std::vector<std::shared_ptr<ControlPoint > > control_points) : Element(control_points){

	double n = (-3 + std::sqrt(9 - 8 * (1 - control_points.size() )))/2;
	double intpart;

	if (modf(n,&intpart) == 0){
		this -> n = (unsigned int)(n);
	}
	else{
		throw(std::runtime_error("The control points cardinal does not correspond to a bezier control net"));
	}

	this -> construct_index_tables();

	this -> update();


}

std::shared_ptr<ControlPoint> Bezier::get_control_point(unsigned int i, unsigned int j){

	return this -> control_points[this -> rev_table[std::make_tuple(i,j,this -> n - i - j)]];
}



std::tuple<unsigned int, unsigned int,unsigned int> Bezier::get_local_indices(std::shared_ptr<ControlPoint> point){
	
	for (unsigned int i = 0; i < this -> control_points.size(); ++i){
		if (this -> control_points[i] == point){
			return this -> forw_table[i];
		}
	}

	throw(std::runtime_error("The provided point should belong to this patch"));
	
}

void Bezier::construct_index_tables(){
	
	this -> rev_table = reverse_table(this -> n);
	this -> forw_table =  forward_table(this -> n);

}

void Bezier::elevate_degree(){



	// A vector storing the new coordinates is created
	std::vector<std::shared_ptr<ControlPoint > > new_control_points;

	// Number of control points after degree elevation by one
	unsigned int N = (this -> n + 2) * (this -> n + 3) / 2;

	
	// The forward table is constructed
	auto forw_post_el = forward_table(this -> n + 1);

	// The reverse table is constructed
	auto rev_post_el = reverse_table(this -> n + 1);

	// Not all new control points need to be computed. Some 
	// are invariants and others may be owned by neighboring patches
	std::map<unsigned int,std::shared_ptr<ControlPoint> > sorted_control_points;

	// The invariant control points are extracted
	int l_n00_before_elevation = 0;
	int l_0n0_before_elevation = (unsigned int)((this -> n) * (this -> n + 1) / 2);
	int l_00n_before_elevation = this -> control_points.size() - 1;

	int i_n00_after_elevation = 0;
	int i_0n0_after_elevation = (unsigned int)((this -> n + 1) * (this -> n + 1 + 1) / 2);
	int i_00n_after_elevation = (unsigned int)((this -> n + 1 + 1) * (this -> n + 2 + 1) / 2) - 1;

	sorted_control_points[i_n00_after_elevation] = this -> control_points[l_n00_before_elevation];
	sorted_control_points[i_0n0_after_elevation] = this -> control_points[l_0n0_before_elevation];
	sorted_control_points[i_00n_after_elevation] = this -> control_points[l_00n_before_elevation];

	// The edge control points that have already been created during the elevation of a neighboring patch are found
	std::set<unsigned int> new_points_to_add_index;
	for (unsigned int i =0; i < N;++i){
		new_points_to_add_index.insert(i);
	}
	new_points_to_add_index.erase(i_n00_after_elevation);
	new_points_to_add_index.erase(i_0n0_after_elevation);
	new_points_to_add_index.erase(i_00n_after_elevation);


	for (auto neighbor =  this -> new_points.begin(); neighbor != this -> new_points.end(); ++ neighbor){

		for (unsigned int k = 0; k < neighbor -> second.size(); ++ k){

			// The local index of the invariant control point is found
			NewPoint new_point = neighbor -> second[k];
			std::shared_ptr<ControlPoint> invariant = new_point.end_point;

			auto local_index_endpoint_neighbor = new_point.indices_endpoint;
			auto local_index_newpoint_neighbor = new_point.indices_newpoint;

			std::tuple<unsigned int, unsigned int, unsigned int> local_index_endpoint_current,local_index_newpoint_current;
			unsigned int global_index;

			if (invariant == this -> control_points[l_n00_before_elevation]){
				global_index = i_n00_after_elevation;
			}
			else if (invariant == this -> control_points[l_0n0_before_elevation]){
				global_index = i_0n0_after_elevation;
			}
			else if (invariant == this -> control_points[l_00n_before_elevation]){
				global_index = i_00n_after_elevation;
			}
			else {
				throw(std::runtime_error("Can't find invariant"));
			}

			local_index_endpoint_current = forw_post_el[global_index];


			// Now, the local indices of the inserted point are found

			arma::vec local_index_endpoint_neighbor_arma = {
				double(std::get<0>(local_index_endpoint_neighbor)),
				double(std::get<1>(local_index_endpoint_neighbor)),
				double(std::get<2>(local_index_endpoint_neighbor))
			};

			arma::vec local_index_newpoint_neighbor_arma = {
				double(std::get<0>(local_index_newpoint_neighbor)),
				double(std::get<1>(local_index_newpoint_neighbor)),
				double(std::get<2>(local_index_newpoint_neighbor))
			};

			int distance_to_endpoint = arma::max(local_index_endpoint_neighbor_arma - local_index_newpoint_neighbor_arma);

			if (distance_to_endpoint <= 0){
				throw(std::runtime_error("Negative or zero distance to endpoint for one new point: " + std::to_string(distance_to_endpoint)));
			}


			if (std::get<0>(local_index_endpoint_current) != 0){

				// On the xy0 edge
				local_index_newpoint_current = std::make_tuple(std::get<0>(local_index_endpoint_current) - distance_to_endpoint,distance_to_endpoint,0);

			}
			else if(std::get<1>(local_index_endpoint_current) != 0){
				// On the 0xy edge

				local_index_newpoint_current = std::make_tuple(0,std::get<1>(local_index_endpoint_current) - distance_to_endpoint,distance_to_endpoint);

			}

			else if(std::get<2>(local_index_endpoint_current) != 0){
				// On the y0x edge

				local_index_newpoint_current = std::make_tuple(distance_to_endpoint,0,std::get<2>(local_index_endpoint_current) - distance_to_endpoint);

			}

			else {
				throw(std::runtime_error("All indices are zero"));
			}

			sorted_control_points[rev_post_el[local_index_newpoint_current]] = new_point.point;
			new_points_to_add_index.erase(rev_post_el[local_index_newpoint_current]);
		}


	}


	// The new coordinates are calculated
	for (auto iter = new_points_to_add_index.begin(); iter != new_points_to_add_index.end(); ++iter){

		int i = (*iter);

		// The coordinates of the inner control points are computed
		arma::vec new_C = arma::zeros<arma::vec>(3);

		for (unsigned int l = 0; l < this -> forw_table.size(); ++l){

			double coef = double(
				combinations(std::get<0>(this -> forw_table[l]),std::get<0>(forw_post_el[i])) 
				* combinations(std::get<1>(this -> forw_table[l]),std::get<1>(forw_post_el[i])) 
				* combinations(std::get<2>(this -> forw_table[l]),std::get<2>(forw_post_el[i])));

			new_C += this -> control_points[l] -> get_coordinates() * coef/(this -> n + 1);

		}

		std::shared_ptr<ControlPoint> new_control_point = std::make_shared<ControlPoint>(ControlPoint());
		new_control_point -> set_coordinates(new_C);

		new_control_point -> add_ownership(this);



		sorted_control_points[i] = new_control_point;


		// I need to store this newly created control point in a structure that keeps track of the 
		// patch where it needs to be added ---- would be great to have the index as well!

		// Such a point lives along an edge	

		// The other facet can be determined by finding which side the 
		// point is on, and by using the fact that the endpoints are invariant!
		std::set<Element *> shared_elements;



		int i_end_point_after;

		if (std::get<0>(forw_post_el[i]) == 0){
			shared_elements = sorted_control_points[i_0n0_after_elevation] -> common_facets(sorted_control_points[i_00n_after_elevation]);
			shared_elements.erase(this);


			i_end_point_after = i_00n_after_elevation;


		}
		else if (std::get<1>(forw_post_el[i]) == 0){
			shared_elements = sorted_control_points[i_n00_after_elevation] -> common_facets(sorted_control_points[i_00n_after_elevation]);
			shared_elements.erase(this);
			i_end_point_after = i_n00_after_elevation;

		}

		else if(std::get<2>(forw_post_el[i]) == 0){
			shared_elements = sorted_control_points[i_0n0_after_elevation] -> common_facets(sorted_control_points[i_n00_after_elevation]);
			shared_elements.erase(this);
			i_end_point_after = i_0n0_after_elevation;

		}

		// This new point should be shared with a neighboring facet
		if (shared_elements.size() == 1){

			std::shared_ptr<ControlPoint> end_point = sorted_control_points[i_end_point_after];

			auto indices_endpoint = forw_post_el[i_end_point_after];
			auto indices_newpoint = forw_post_el[i];

			NewPoint new_point(new_control_point,end_point,indices_newpoint,indices_endpoint);

			new_control_point -> add_ownership(*shared_elements.begin());

			dynamic_cast<Bezier *>(*shared_elements.begin()) -> add_point_from_neighbor(this,new_point);
		}

	}

	// The control points are brought together in an array
	for (int i = 0; i < sorted_control_points.size(); ++i){
		new_control_points.push_back(sorted_control_points[i]);
	}


	this -> n = this -> n + 1;
	this -> control_points = new_control_points;
	this -> construct_index_tables();

	this -> new_points.clear();

	this -> info_mat_ptr = nullptr;
	this -> dX_bar_ptr = nullptr;


}


unsigned int Bezier::get_degree() const{
	return this -> n;
}


std::set < Element * > Bezier::get_neighbors(bool all_neighbors) const{
	std::set < Element * > neighbors;
	throw(std::runtime_error("Bezier::get_neighbors is not implemented"));
	return neighbors;
}


arma::vec Bezier::evaluate(const double u, const double v) const{

	arma::vec P = arma::zeros<arma::vec>(3);
	for (unsigned int l = 0; l < this -> control_points.size(); ++l){
		
		int i = std::get<0>(this -> forw_table[l]);
		int j = std::get<1>(this -> forw_table[l]);

		P += this -> bernstein(u,v,i,j,this -> n) * this -> control_points[l] -> get_coordinates();
	}
	return P;
}

arma::vec Bezier::get_normal(const double u, const double v) {
	arma::mat partials = this -> partial_bezier(u,v);
	return arma::normalise(arma::cross(partials.col(0),partials.col(1)));
}


double Bezier::bernstein(
	const double u, 
	const double v,
	const int i,
	const int j,
	const int n) {

	if (i < 0 || i > n || j < 0 || j > n || i + j > n){
		return 0;
	}

	if (n == 0){
		return 1;
	}

	double coef =  boost::math::factorial<double>(n) / (
		boost::math::factorial<double>(i)
		* boost::math::factorial<double>(j) 
		* boost::math::factorial<double>(n - i - j));

	return coef * (std::pow(u,i) *  std::pow(v,j)
		* std::pow(1 - u - v, n - i - j ));

}

arma::rowvec Bezier::partial_bernstein( 
	const double u, 
	const double v,
	const int i ,  
	const int j, 
	const int n) {

	arma::rowvec partials(2);


	partials(0) = n *( bernstein(u, v,i - 1,j,n - 1) - bernstein(u, v,i,j,n - 1));
	partials(1) = n *( bernstein(u, v,i,j - 1,n - 1) - bernstein(u, v,i,j,n - 1));

	return partials;
}


arma::mat Bezier::partial_bernstein_du( 
	const double u, 
	const double v,
	const int i ,  
	const int j, 
	const int n) {

	arma::mat partials = n * ( partial_bernstein(u, v,i - 1,j,n - 1) - partial_bernstein(u, v,i,j,n - 1));




	return partials;
}

arma::mat Bezier::partial_bernstein_dv( 
	const double u, 
	const double v,
	const int i ,  
	const int j, 
	const int n) {

	arma::mat partials = n * ( partial_bernstein(u, v,i,j-1,n - 1) - partial_bernstein(u, v,i,j,n - 1));

	return partials;
}

arma::mat Bezier::partial_bezier_du(
	const double u,
	const double v) {

	arma::mat partials = arma::zeros<arma::mat>(3,2);
	for (unsigned int l = 0; l < this -> control_points.size(); ++l){
		
		int i = std::get<0>(this -> forw_table[l]);
		
		int j = std::get<1>(this -> forw_table[l]);

		partials += this -> control_points[l] -> get_coordinates() * Bezier::partial_bernstein_du(u,v,i,j,this -> n) ;
	}
	return partials;

}

arma::mat Bezier::partial_bezier_dv(
	const double u,
	const double v) {

	arma::mat partials = arma::zeros<arma::mat>(3,2);
	for (unsigned int l = 0; l < this -> control_points.size(); ++l){
		
		int i = std::get<0>(this -> forw_table[l]);
		
		int j = std::get<1>(this -> forw_table[l]);

		partials += this -> control_points[l] -> get_coordinates() * Bezier::partial_bernstein_du(u,v,i,j,this -> n) ;
	}
	return partials;

}




arma::mat Bezier::partial_bezier(
	const double u,
	const double v) {

	arma::mat partials = arma::zeros<arma::mat>(3,2);
	for (unsigned int l = 0; l < this -> control_points.size(); ++l){	
		int i = std::get<0>(this -> forw_table[l]);
		int j = std::get<1>(this -> forw_table[l]);

		partials += this -> control_points[l] -> get_coordinates() * Bezier::partial_bernstein(u,v,i,j,this -> n) ;
	}
	return partials;

}

arma::mat Bezier::covariance_surface_point_deprecated(
	const double u,
	const double v,
	const arma::vec & dir,
	const arma::mat & P_X){


	arma::mat P = arma::zeros<arma::mat>(3,3);

	arma::mat A = RBK::tilde(dir) * partial_bezier(u,v);
	arma::mat AAA = A * arma::inv(A .t() * A);


	for (unsigned int i = 0; i < this -> control_points.size(); ++i){
		for (unsigned int k = 0; k < this -> control_points.size(); ++k){

			P += (
				bernstein(u, v,std::get<0>(this -> forw_table[i]),std::get<1>(this -> forw_table[i]),n) 
				* bernstein(u, v,std::get<0>(this -> forw_table[k]),std::get<1>(this -> forw_table[k]),n) 
				* P_X.submat( 3 * i, 3 * k, 3 * i + 2, 3 * k + 2 )
				);
		}
	}

	for (unsigned int i = 0; i < this -> control_points.size(); ++i){
		for (unsigned int k = 0; k < this -> control_points.size(); ++k){
			for (unsigned int l = 0; l < this -> control_points.size(); ++l){

				P += (
					bernstein(u, v,std::get<0>(this -> forw_table[i]),std::get<1>(this -> forw_table[i]),n) 
					* bernstein(u, v,std::get<0>(this -> forw_table[l]),std::get<1>(this -> forw_table[l]),n) 
					* P_X.submat( 3 * i, 3 * l, 3 * i + 2, 3 * l + 2 )
					* RBK::tilde(dir)
					* AAA
					* partial_bernstein(
						u, 
						v,
						std::get<0>(this -> forw_table[k]) ,  
						std::get<1>(this -> forw_table[k]), 
						n).t()
					* this -> control_points[k] -> get_coordinates().t() );
			}
		}
	}

	for (unsigned int i = 0; i < this -> control_points.size(); ++i){
		for (unsigned int j = 0; j < this -> control_points.size(); ++j){
			for (unsigned int k = 0; k < this -> control_points.size(); ++k){

				P -= (
					this -> control_points[i] -> get_coordinates()
					* partial_bernstein(
						u, 
						v,
						std::get<0>(this -> forw_table[i]) ,  
						std::get<1>(this -> forw_table[i]), 
						n)
					* AAA.t()
					* RBK::tilde(dir)

					* bernstein(u, v,std::get<0>(this -> forw_table[j]),std::get<1>(this -> forw_table[j]),n) 
					* bernstein(u, v,std::get<0>(this -> forw_table[k]),std::get<1>(this -> forw_table[k]),n) 
					* P_X.submat( 3 * j, 3 * k, 3 * j + 2, 3 * k + 2 ) );
			}
		}
	}

	for (unsigned int i = 0; i < this -> control_points.size(); ++i){
		for (unsigned int k = 0; k < this -> control_points.size(); ++k){
			for (unsigned int l = 0; l < this -> control_points.size(); ++l){
				for (unsigned int j = 0; j < this -> control_points.size(); ++j){

					P -= (
						this -> control_points[i] -> get_coordinates()
						* partial_bernstein(
							u, 
							v,
							std::get<0>(this -> forw_table[i]) ,  
							std::get<1>(this -> forw_table[i]), 
							n)
						* AAA.t()
						* RBK::tilde(dir)
						* bernstein(u, v,std::get<0>(this -> forw_table[l]),std::get<1>(this -> forw_table[l]),n) 
						* bernstein(u, v,std::get<0>(this -> forw_table[j]),std::get<1>(this -> forw_table[j]),n) 
						* P_X.submat( 3 * l, 3 * j, 3 * l + 2, 3 * j + 2 ) 
						* RBK::tilde(dir)
						* AAA
						* partial_bernstein(
							u, 
							v,
							std::get<0>(this -> forw_table[k]) ,  
							std::get<1>(this -> forw_table[k]), 
							n).t() * this -> control_points[k] -> get_coordinates().t() );
				}
			}
		}
	}

	return P;




}


arma::mat Bezier::partial_n_partial_Ck(const double u, const double v,const int i ,  const int j, const int n){



	auto P_chi = this -> partial_bezier(u,v);
	auto P_u = P_chi.col(0);
	auto P_v = P_chi.col(1);

	double norm = arma::norm(arma::cross(P_u,P_v));
	arma::mat P_u_tilde = RBK::tilde(P_u);
	arma::mat P_v_tilde = RBK::tilde(P_v);


	arma::rowvec dBernstein_chi = Bezier::partial_bernstein(u,v,i,j,n);


	return (1./norm * (arma::eye<arma::mat>(3,3) - P_u_tilde * P_v * arma::cross(P_u,P_v).t() / std::pow(norm,2))
		* (P_u_tilde * dBernstein_chi(1) - P_v_tilde * dBernstein_chi(0)));

}






double Bezier::initialize_covariance(const std::vector<Footpoint> & footpoints,
	std::vector<arma::vec> & v,
	std::vector<arma::vec> & W,
	std::vector<arma::vec> & v_i_norm,
	std::vector<double> & epsilon){

	unsigned int N_C = this -> control_points.size();
	unsigned int P = 3 * N_C * (3 * N_C + 1) / 2;

	arma::mat H_mat = arma::zeros<arma::mat>(3*N_C, 3*N_C);
	arma::mat N_mat = arma::zeros<arma::mat>(3*N_C, 3*N_C);

	for (unsigned int i = 0; i < footpoints.size(); ++i){

		Footpoint footpoint = footpoints[i];
		arma::vec dir = footpoint.n;

		arma::mat A = RBK::tilde(dir) * partial_bezier(footpoint.u,footpoint.v);
		arma::mat AAA = A * arma::inv(A .t() * A);
		arma::mat M = arma::zeros<arma::mat>(3 * N_C,3);
		arma::mat Ck_dBkdchi = arma::zeros<arma::mat>(3,2);
		arma::vec v_i_norm_vec = arma::zeros<arma::vec>(N_C);

		for (unsigned int k = 0; k < N_C; ++k){
			auto indices = this -> forw_table[k];
			Ck_dBkdchi += this -> control_points[k] -> get_coordinates()  * partial_bernstein(footpoint.u, footpoint.v,std::get<0>(indices) ,  std::get<1>(indices), this -> n);
			M.submat( 3 * k ,0, 3 * k + 2,2) = bernstein(footpoint.u, footpoint.v,std::get<0>(indices),std::get<1>(indices),n)  * arma::eye<arma::mat>(3,3);
		}

		arma::mat K =  RBK::tilde(dir) * AAA * Ck_dBkdchi.t();
		arma::mat J = M * (arma::eye<arma::mat>(3,3) + K);
		arma::vec v_i = J * dir;
		arma::vec W_i = arma::zeros<arma::vec>(P);
		arma::mat v_i_v_i = v_i * v_i.t();

		for (unsigned int k = 0; k < N_C; ++k){
			v_i_norm_vec(k) = arma::dot(v_i.rows(3 * k,3 * k + 2),v_i.rows(3 * k,3 * k + 2));
		}

		for (unsigned int p = 0; p < W_i.n_rows; ++p){
			int k = int ( (6 * N_C + 1 - std::sqrt(std::pow(6 * N_C + 1,2) - 8 * p))/2);
			int l = p - k * 3 * N_C + k * (k + 1) / 2;
			W_i(p) = v_i_v_i(k,l);
		}

		double epsilon_i = arma::dot(dir,footpoint.Ptilde - footpoint.Pbar);

		v.push_back(v_i);
		W.push_back(W_i);
		epsilon.push_back(epsilon_i);
		v_i_norm.push_back(v_i_norm_vec);

		N_mat += epsilon_i * epsilon_i * v_i * v_i.t() / std::pow(arma::dot(v_i,v_i),2);
		H_mat += v_i * v_i.t() / arma::dot(v_i,v_i);

	}

	// A-priori covariance
	arma::vec N_mat_vec = arma::vectorise(N_mat);
	arma::vec H_mat_vec = arma::vectorise(H_mat);

	double alpha = arma::dot(H_mat_vec,N_mat_vec) /  arma::dot(H_mat_vec,H_mat_vec);

	return alpha;

}


void Bezier::train_patch_covariance(){

	std::vector<arma::vec> v;
	std::vector<arma::vec> W;
	std::vector<arma::vec> v_i_norm;
	std::vector<double> epsilon;

	unsigned int N_C = this -> control_points.size();
	unsigned int P = 3 * N_C * (3 * N_C + 1) / 2;
	unsigned int N_iter = 10 ;
	this -> P_X = arma::mat(3 * N_C,3 * N_C);


	// The initial guess for the covariance is computed.
	double alpha = initialize_covariance(this -> footpoints,v,W,v_i_norm,epsilon);

	arma::vec L = arma::ones<arma::vec>(N_C) * std::log(alpha);	
	arma::vec lower_bounds =  L - 3;
	arma::vec upper_bounds = L + 3;	
	std::cout << "-- Initial guess: " << std::log(alpha) << std::endl;

	std::pair< const std::vector<Footpoint> * ,std::vector<arma::vec> * > args = std::make_pair(&footpoints,&v_i_norm);

	// The covariance is refined by a particle-in-swarm optimizer
	Psopt<std::pair< const std::vector<Footpoint> * ,std::vector<arma::vec> * > > psopt(Bezier::compute_log_likelihood_block_diagonal, 
		lower_bounds,
		upper_bounds, 
		1000,
		N_iter,
		args);

	psopt.run( true,true);

	L = psopt.get_result();
	std::cout << "-- Final parametrization: " << L.t() << std::endl;

	arma::vec L_correct_shape = arma::vectorise(arma::repmat(L,1,3),1).t();
	std::cout << L_correct_shape << std::endl;

	this -> P_X = arma::diagmat(arma::exp(arma::vectorise(arma::repmat(L,1,3),1)));

}

void Bezier::train_patch_covariance(const std::vector<Footpoint> & footpoints){

	std::vector<arma::vec> v;
	std::vector<arma::vec> W;
	std::vector<arma::vec> v_i_norm;
	std::vector<double> epsilon;

	unsigned int N_C = this -> control_points.size();
	unsigned int P = 3 * N_C * (3 * N_C + 1) / 2;
	unsigned int N_iter = 10 ;
	this -> P_X = arma::mat(3 * N_C,3 * N_C);


	// The initial guess for the covariance is computed.
	double alpha = initialize_covariance(footpoints,v,W,v_i_norm,epsilon);

	arma::vec L = arma::ones<arma::vec>(3 * N_C) * std::log(alpha);	
	arma::vec lower_bounds =  L - 3;
	arma::vec upper_bounds = L + 3;	

	std::pair< const std::vector<Footpoint> * ,Bezier * > args = std::make_pair(&footpoints,this);

	// The covariance is refined by a particle-in-swarm optimizer
	Psopt<std::pair< const std::vector<Footpoint> * ,Bezier * > > psopt(Bezier::compute_log_likelihood_full_diagonal, 
		lower_bounds,
		upper_bounds, 
		100,
		N_iter,
		args);

	psopt.run( true,true);
	L = psopt.get_result();

	this -> P_X = arma::diagmat(arma::exp(L));

}


void Bezier::train_patch_covariance(arma::mat & P_X,const std::vector<Footpoint> & footpoints,bool diag){


	std::vector<arma::vec> v;
	std::vector<arma::vec> W;
	std::vector<arma::vec> v_i_norm;
	std::vector<double> epsilon;

	unsigned int N_C = this -> control_points.size();
	unsigned int P = 3 * N_C * (3 * N_C + 1) / 2;
	unsigned int N_iter = 10 ;
	this -> P_X = arma::mat(3 * N_C,3 * N_C);


	// The initial guess for the covariance is computed.
	double alpha = initialize_covariance(footpoints,v,W,v_i_norm,epsilon);

	if (diag){
		P_X = alpha * arma::eye<arma::mat>(3 * N_C,3 * N_C) ;
		return;
	}


	arma::vec L = arma::ones<arma::vec>(3 * N_C) * std::log(alpha);	
	arma::vec lower_bounds =  L - 3;
	arma::vec upper_bounds = L + 3;	
	
	std::pair< const std::vector<Footpoint> * ,Bezier * > args = std::make_pair(&footpoints,
		this);

	Psopt<std::pair< const std::vector<Footpoint> * ,Bezier * > > psopt(Bezier::compute_log_likelihood_full_diagonal, 
		lower_bounds,
		upper_bounds, 
		100,
		N_iter,
		args);

	psopt.run( true,true);
	L = psopt.get_result();

	P_X = arma::diagmat(arma::exp(L));

}


void Bezier::add_footpoint(Footpoint footpoint){
	this -> footpoints.push_back(footpoint);
}

bool Bezier::has_footpoints() const{
	return (this -> footpoints.size()!= 0);
}

void Bezier::reset_footpoints(){
	this -> footpoints.clear();
}

double Bezier::compute_log_likelihood_full_diagonal(arma::vec L,
	std::pair<const std::vector<Footpoint> * ,Bezier * > args){

	// All the footpoints are processed
	double log_likelihood = 0;

	for (unsigned int i = 0; i <  args.first -> size(); ++i){
		auto footpoint = args.first -> at(i);

		arma::mat P = args.second -> covariance_surface_point(footpoint . u,
			footpoint . v,
			footpoint . n,
			arma::diagmat(arma::exp(L)));


		double sigma_i_2 = arma::dot(footpoint . n,P * footpoint . n);
		double epsilon_i = arma::dot(footpoint . n,
			footpoint . Ptilde - footpoint . Pbar);


		log_likelihood += - std::log(sigma_i_2) - std::pow(epsilon_i,2) / sigma_i_2;
	}


	return log_likelihood;

}



double Bezier::compute_log_likelihood_block_diagonal(arma::vec L,
	std::pair< const std::vector<Footpoint> * ,std::vector<arma::vec> * > args){

	// All the footpoints are processed
	double log_likelihood = 0;

	for (unsigned int i = 0; i <  args.first -> size(); ++i){
		auto footpoint = args.first -> at(i);

		auto v_i_norm = args.second -> at(i);
		arma::vec Z_i =  v_i_norm % arma::exp(L);
		double sigma_i_2 =  arma::sum(Z_i);

		double epsilon_i = arma::dot(footpoint . n,
			footpoint . Ptilde - footpoint . Pbar);


		log_likelihood += - std::log(sigma_i_2) - std::pow(epsilon_i,2) / sigma_i_2;
	}


	return log_likelihood;

}







arma::mat Bezier::covariance_surface_point(
	const double u,
	const double v,
	const arma::vec & dir,
	const arma::mat & P_X){

	arma::mat A = RBK::tilde(dir) * partial_bezier(u,v);
	arma::mat AAA = A * arma::inv(A .t() * A);
	arma::mat M = arma::zeros<arma::mat>(3 * this -> control_points.size(),3);
	arma::mat Ck_dBkdchi = arma::zeros<arma::mat>(3,2);
	for (unsigned int k = 0; k < this -> control_points.size(); ++k){

		auto indices = this -> forw_table[k];

		Ck_dBkdchi += this -> control_points[k] -> get_coordinates()  * partial_bernstein(u, v,std::get<0>(indices) ,  std::get<1>(indices), this -> n);
		M.submat( 3 * k ,0, 3 * k + 2,2) = bernstein(u, v,std::get<0>(indices),std::get<1>(indices),n)  * arma::eye<arma::mat>(3,3);
	}

	arma::mat K =  RBK::tilde(dir) * AAA * Ck_dBkdchi.t();
	arma::mat J = M * (arma::eye<arma::mat>(3,3) + K);

	return J.t() * P_X * J;


}



void Bezier::compute_normal(){

}

void Bezier::compute_area(){


	// The area is computed by quadrature
	arma::vec weights = {-27./96.,25./96,25./96,25./96};
	arma::vec u = {1./3.,1./5.,1./5,3./5};
	arma::vec v = {1./3.,1./5.,3./5,1./5};

	this -> area = 0;
	for (
		int i = 0; i < weights.n_rows; ++i){
		this -> area += weights(i) * g(u(i),v(i));
}



}

void Bezier::compute_center(){
	this -> center = this -> evaluate(1./3.,1./3.);
}


double Bezier::g(double u, double v) const{


	arma::vec V = arma::zeros<arma::vec>(3);

	for (unsigned int l = 0; l < this -> forw_table.size(); ++l){

		for (unsigned int k = 0; k < this -> forw_table.size(); ++k){


			double bl0 = bernstein(u,v,std::get<0>(this -> forw_table[l]) - 1,std::get<1>(this -> forw_table[l]), this -> n - 1);
			double bl1 = bernstein(u,v,std::get<0>(this -> forw_table[l]),std::get<1>(this -> forw_table[l]), this -> n - 1);
			double bk0 = bernstein(u,v,std::get<0>(this -> forw_table[k]),std::get<1>(this -> forw_table[k])-1, this -> n - 1);
			double bk1 = bernstein(u,v,std::get<0>(this -> forw_table[k]),std::get<1>(this -> forw_table[k]), this -> n - 1);

			V = V + (bl0 - bl1) * (bk0 - bk1) * arma::cross(this -> control_points[l] -> get_coordinates(),
				this -> control_points[k] -> get_coordinates());

		}

	}

	return arma::norm(V);

}

unsigned int Bezier::combinations(unsigned int k, unsigned int n){

	if (k < 0 || k > n){
		return 0;
	}

	// unsigned int v = n--;

	// for (
	// int i = 2; i < k + 1; ++i, --n){
	// 	v = v * n / i;
	// }

	

	return boost::math::factorial<double>(n) / (boost::math::factorial<double>(k)  * boost::math::factorial<double>(n - k));

}


void Bezier::add_point_from_neighbor(Element * element, NewPoint & new_point){
	this -> new_points[element].push_back(new_point);
}


std::map< std::tuple<unsigned int, unsigned int, unsigned int> ,unsigned int> Bezier::reverse_table(unsigned int n){

	std::map< std::tuple<unsigned int, unsigned int, unsigned int>,unsigned int> map;
	unsigned int l = 0;

	for (int i = n; i > -1 ; -- i){

		for (unsigned int k = 0 ; k < n + 1 - i; ++k){
			
			int j = n - i - k;
			auto indices = std::make_tuple(i,j,k);
			map[indices] = l;

			l = l + 1;
		}
	}

	return map;

}

std::vector<std::tuple<unsigned int, unsigned int, unsigned int> > Bezier::forward_table(unsigned int n){

	std::vector<std::tuple<unsigned int, unsigned int, unsigned int> > table;


	for (int i = n; i > -1 ; -- i){

		for (unsigned int k = 0 ; k < n + 1 - i; ++k){
			int j = n - i - k;
			auto indices = std::make_tuple(i,j,k);

			table.push_back(indices);
		}
	}

	return table;
}









