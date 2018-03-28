#include "ShapeModelBezier.hpp"
#include "ShapeModelTri.hpp"
#include "ShapeModelImporter.hpp"



ShapeModelBezier::ShapeModelBezier(ShapeModelTri * shape_model,
	std::string ref_frame_name,
	FrameGraph * frame_graph): ShapeModel(ref_frame_name,frame_graph){

	// All the facets of the original shape model are browsed
	// The shape starts as a uniform union of order-2 Bezier patches

	// The control point of this shape model are the same as that
	// of the provided shape
	this -> control_points = (*shape_model -> get_control_points());

	// The ownership relationships are reset
	for (unsigned int i = 0; i < shape_model -> get_NControlPoints(); ++i){
		this -> control_points[i] -> reset_ownership();
	}

	// The surface elements are almost the same, expect that they are 
	// Bezier patches and not facets
	for (unsigned int i = 0; i < shape_model -> get_NElements(); ++i){
		auto patch = std::make_shared<Bezier>(Bezier(*shape_model -> get_elements() -> at(i) -> get_control_points()));

		this -> elements.push_back(patch);

		patch -> get_control_points() -> at(0) -> add_ownership(patch.get());
		patch -> get_control_points() -> at(1) -> add_ownership(patch.get());
		patch -> get_control_points() -> at(2) -> add_ownership(patch.get());

	}

	this -> construct_kd_tree_control_points();
	this -> populate_mass_properties_coefs();
	this -> update_mass_properties();
	this -> inertia = shape_model -> get_inertia();

}

std::shared_ptr<arma::mat> ShapeModelBezier::get_info_mat_ptr() const{
	return this -> info_mat_ptr;
}

std::shared_ptr<arma::vec> ShapeModelBezier::get_dX_bar_ptr() const{
	return this -> dX_bar_ptr;
}


void ShapeModelBezier::initialize_info_mat(){
	unsigned int N = this -> control_points.size();
	this -> info_mat_ptr = std::make_shared<arma::mat>(arma::eye<arma::mat>(3 * N,3 * N));
}

void ShapeModelBezier::initialize_dX_bar(){
	unsigned int N = this -> control_points.size();
	this -> dX_bar_ptr = std::make_shared<arma::vec>(arma::zeros<arma::vec>(3 * N));
}

ShapeModelBezier::ShapeModelBezier(std::string ref_frame_name,
	FrameGraph * frame_graph): ShapeModel(ref_frame_name,frame_graph){

}


arma::mat ShapeModelBezier::random_sampling(unsigned int N,const arma::mat & R) const{

	arma::mat points = arma::zeros<arma::mat>(3,N);
	arma::mat S = arma::chol(R,"lower");

	// N points are randomly sampled from the surface of the shape model
	// #pragma omp parallel for
	for (unsigned int i = 0; i < N; ++i){

		unsigned int element_index = arma::randi<arma::vec>( 1, arma::distr_param(0,this -> elements.size() - 1) ) (0);

		Bezier * patch = dynamic_cast<Bezier * >(this -> elements[element_index].get());
		arma::vec random = arma::randu<arma::vec>(2);
		double u = random(0);
		double v = (1 - u) * random(1);

		points.col(i) = patch -> evaluate(u,v) + S * arma::randn<arma::vec>(3) ;

	}

	return points;



}


ShapeModelBezier::ShapeModelBezier(Bezier patch){

	this -> elements.push_back(std::make_shared<Bezier>(patch));
}

void ShapeModelBezier::compute_surface_area(){
	std::cout << "Warning: should only be used for post-processing\n";

}

void ShapeModelBezier::update_mass_properties() {
	
	std::chrono::time_point<std::chrono::system_clock> start, end;

	start = std::chrono::system_clock::now();

	this -> compute_volume();
	this -> compute_center_of_mass();
	this -> compute_inertia();

	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time in ShapeModelBezier::update_mass_properties: " << elapsed_seconds.count() << " s"<< std::endl;


}


void ShapeModelBezier::compute_volume(){
	
	double volume = 0;

	#pragma omp parallel for reduction(+:volume) if (USE_OMP_SHAPE_MODEL)
	for (unsigned int el_index = 0; el_index < this -> elements.size(); ++el_index) {
		
		Bezier * patch = static_cast<Bezier * >(this -> elements[el_index].get());		
		
		for (int index = 0 ; index <  this -> volume_indices_coefs_table.size(); ++index) {

			int i =  int(this -> volume_indices_coefs_table[index][0]);
			int j =  int(this -> volume_indices_coefs_table[index][1]);
			int k =  int(this -> volume_indices_coefs_table[index][2]);
			int l =  int(this -> volume_indices_coefs_table[index][3]);
			int m =  int(this -> volume_indices_coefs_table[index][4]);
			int p =  int(this -> volume_indices_coefs_table[index][5]);
			
			volume += this -> volume_indices_coefs_table[index][6] * patch -> triple_product(i,j,k,l,m,p);

		}

	}
	this -> volume = volume;
}


void ShapeModelBezier::compute_center_of_mass(){

	this -> cm = arma::zeros<arma::vec>(3);

	double cx = 0;
	double cy = 0;
	double cz = 0;

	int n = this -> get_degree();

	// #pragma omp parallel for reduction(+:cx,cy,cz)
	for (unsigned int el_index = 0; el_index < this -> elements.size(); ++el_index) {

		Bezier * patch = dynamic_cast<Bezier * >(this -> elements[el_index].get());		

		for (auto index = 0 ; index <  this -> cm_gamma_indices_coefs_table.size(); ++index) {

			int i =  int(this -> cm_gamma_indices_coefs_table[index][0]);
			int j =  int(this -> cm_gamma_indices_coefs_table[index][1]);
			int k =  int(this -> cm_gamma_indices_coefs_table[index][2]);
			int l =  int(this -> cm_gamma_indices_coefs_table[index][3]);
			int m =  int(this -> cm_gamma_indices_coefs_table[index][4]);
			int p =  int(this -> cm_gamma_indices_coefs_table[index][5]);
			int q =  int(this -> cm_gamma_indices_coefs_table[index][6]);
			int r =  int(this -> cm_gamma_indices_coefs_table[index][7]);


			double result[3];

			patch -> quadruple_product(result,i ,j ,k ,l ,m ,p, q, r );

			cx += this -> cm_gamma_indices_coefs_table[index][8] * result[0];
			cy += this -> cm_gamma_indices_coefs_table[index][8] * result[1];
			cz += this -> cm_gamma_indices_coefs_table[index][8] * result[2];


		}

	}

	this -> cm = {cx,cy,cz};
	this -> cm = this -> cm / this -> volume;


}

void ShapeModelBezier::compute_inertia(){

	arma::mat::fixed<3,3> inertia;
	inertia.fill(0);
	double norm_length = std::cbrt(this -> volume);
	double factor = 1./std::pow(norm_length,5);

	#pragma omp parallel for reduction(+:inertia) if (USE_OMP_SHAPE_MODEL)
	for (unsigned int el_index = 0; el_index < this -> elements.size(); ++el_index) {
		
		Bezier * patch = static_cast<Bezier * >(this -> elements[el_index].get());		
		
		for (int index = 0 ; index <  this -> inertia_indices_coefs_table.size(); ++index) {

			int i =  int(this -> inertia_indices_coefs_table[index][0]);
			int j =  int(this -> inertia_indices_coefs_table[index][1]);
			int k =  int(this -> inertia_indices_coefs_table[index][2]);
			int l =  int(this -> inertia_indices_coefs_table[index][3]);
			int m =  int(this -> inertia_indices_coefs_table[index][4]);
			int p =  int(this -> inertia_indices_coefs_table[index][5]);
			int q =  int(this -> inertia_indices_coefs_table[index][6]);
			int r =  int(this -> inertia_indices_coefs_table[index][7]);
			int s =  int(this -> inertia_indices_coefs_table[index][8]);
			int t =  int(this -> inertia_indices_coefs_table[index][9]);
			
			inertia += factor * (this -> inertia_indices_coefs_table[index][10]  
				* RBK::tilde(patch -> get_control_point_coordinates(i,j)) 
				* RBK::tilde(patch -> get_control_point_coordinates(k,l))
				* patch -> triple_product(m,p,q,r,s,t));

		}

	}
	this -> inertia = inertia;




}


double ShapeModelBezier::get_volume_sd() const{
	return this -> volume_sd;
}

arma::mat ShapeModelBezier::get_cm_cov() const{
	return this -> cm_cov;
}


void ShapeModelBezier::compute_volume_sd(){


	double vol_sd = 0;




	// The list of connected facets should be formed somewhere here

	// a vector storing sets of vector pointers? where the index of the vector
	// refers to the index of the surface element

	std::vector<std::set < Element * > > connected_elements;

	for (unsigned int e = 0; e < this -> elements.size(); ++e) {

		auto elements = this -> elements[e] -> get_neighbors(true);
		connected_elements.push_back(elements);
	}




	std::cout << "- Computing volume sd...\n";
	boost::progress_display progress(this -> elements.size()) ;

	#pragma omp parallel for reduction(+:vol_sd)

	for (unsigned int e = 0; e < this -> elements.size(); ++e) {
		
		Bezier * patch_e = static_cast<Bezier * >(this -> elements[e].get());

		auto neighbors = connected_elements[e];
		for (auto it_neighbors = neighbors.begin(); it_neighbors  != neighbors.end(); ++it_neighbors){

			Bezier * patch_f = static_cast<Bezier * >(*it_neighbors);

			for (int index = 0 ; index <  this -> volume_sd_indices_coefs_table.size(); ++index) {


				// i
				int i =  int(this -> volume_sd_indices_coefs_table[index][0]);
				int j =  int(this -> volume_sd_indices_coefs_table[index][1]);

				// j
				int k =  int(this -> volume_sd_indices_coefs_table[index][2]);
				int l =  int(this -> volume_sd_indices_coefs_table[index][3]);
				
				// k
				int m =  int(this -> volume_sd_indices_coefs_table[index][4]);
				int p =  int(this -> volume_sd_indices_coefs_table[index][5]);

				// l
				int q =  int(this -> volume_sd_indices_coefs_table[index][6]);
				int r =  int(this -> volume_sd_indices_coefs_table[index][7]);

				// m
				int s =  int(this -> volume_sd_indices_coefs_table[index][8]);
				int t =  int(this -> volume_sd_indices_coefs_table[index][9]);

				// p
				int u =  int(this -> volume_sd_indices_coefs_table[index][10]);
				int v =  int(this -> volume_sd_indices_coefs_table[index][11]);



				arma::vec left_vec = patch_e -> get_cross_products(i,j,k,l,m,p);
				arma::vec right_vec = patch_f -> get_cross_products(q,r,s,t,u,v);

				auto Ci = patch_e -> get_control_point(i,j);
				auto Cj = patch_e -> get_control_point(k,l);
				auto Ck = patch_e -> get_control_point(m,p);
				auto Cl = patch_f -> get_control_point(q,r);
				auto Cm = patch_f -> get_control_point(s,t);
				auto Cp = patch_f -> get_control_point(u,v);



				arma::mat P;


				ShapeModel::assemble_covariance(P,Ci,Cj,Ck,Cl,Cm,Cp);


				vol_sd += this -> volume_sd_indices_coefs_table[index][12] * arma::dot(left_vec,P * right_vec);


			}


		}
		++progress;


	}

	this -> volume_sd = std::sqrt(vol_sd);


}

void ShapeModelBezier::compute_cm_cov(){

	std::cout << "- Computing cm covariance ...\n";

	arma::mat::fixed<3,3> cm_cov_temp;

	
	cm_cov_temp = this -> cm * this -> cm.t() * std::pow(this -> volume_sd,2);

	std::vector<std::set < Element * > > connected_elements;

	for (unsigned int e = 0; e < this -> elements.size(); ++e) {

		auto elements = this -> elements[e] -> get_neighbors(true);
		connected_elements.push_back(elements);
	}

	boost::progress_display progress(this -> elements.size()) ;



	#pragma omp parallel for reduction (+:cm_cov_temp)
	
	for (unsigned int e = 0; e < this -> elements.size(); ++e) {
		Bezier * patch_e = static_cast<Bezier * >(this -> elements[e].get());

		auto neighbors = connected_elements[e];

		for (auto it_neighbors = neighbors.begin(); it_neighbors  != neighbors.end(); ++it_neighbors){

			Bezier * patch_f = static_cast<Bezier * >(*it_neighbors);


			for (int index = 0 ; index <  this -> cm_cov_1_indices_coefs_table.size(); ++index) {

				auto coefs_row = this -> cm_cov_1_indices_coefs_table[index];

				// i
				int i =  int(coefs_row[0]);
				int j =  int(coefs_row[1]);

				// j
				int k =  int(coefs_row[2]);
				int l =  int(coefs_row[3]);

				// k
				int m =  int(coefs_row[4]);
				int p =  int(coefs_row[5]);

				// l
				int q =  int(coefs_row[6]);
				int r =  int(coefs_row[7]);

				// m
				int s =  int(coefs_row[8]);
				int t =  int(coefs_row[9]);

				// p
				int u =  int(coefs_row[10]);
				int v =  int(coefs_row[11]);

				// q
				int w =  int(coefs_row[12]);
				int x =  int(coefs_row[13]);

				// r
				int y =  int(coefs_row[14]);
				int z =  int(coefs_row[15]);


				auto Ci = patch_e -> get_control_point(i,j);
				auto Cj = patch_e -> get_control_point(k,l);
				auto Ck = patch_e -> get_control_point(m,p);
				auto Cl = patch_e -> get_control_point(q,r);

				auto Cm = patch_f -> get_control_point(s,t);
				auto Cp = patch_f -> get_control_point(u,v);
				auto Cq = patch_f -> get_control_point(w,x);
				auto Cr = patch_f -> get_control_point(y,z);


				arma::mat P;

				ShapeModel::assemble_covariance(P,Ci,Cj,Ck,Cl,Cm,Cp,Cq,Cr);


				// std::cout << P << std::endl;

				arma::mat left_mat = patch_e -> get_augmented_cross_products(i,j,k,l,m,p,q,r);
				arma::mat right_mat = patch_f -> get_augmented_cross_products(s,t,u,v,w,x,y,z);

				cm_cov_temp += coefs_row[16] * left_mat.t() * P * right_mat;
			}
		}

		++progress;
	}




	// 		for (int index = 0 ; index <  this -> cm_cov_2_indices_coefs_table.size(); ++index) {

	// 			// i
	// 			int i =  int(this -> cm_cov_2_indices_coefs_table[index][0]);
	// 			int j =  int(this -> cm_cov_2_indices_coefs_table[index][1]);

	// 			// j
	// 			int k =  int(this -> cm_cov_2_indices_coefs_table[index][2]);
	// 			int l =  int(this -> cm_cov_2_indices_coefs_table[index][3]);

	// 			// k
	// 			int m =  int(this -> cm_cov_2_indices_coefs_table[index][4]);
	// 			int p =  int(this -> cm_cov_2_indices_coefs_table[index][5]);

	// 			// l
	// 			int q =  int(this -> cm_cov_2_indices_coefs_table[index][6]);
	// 			int r =  int(this -> cm_cov_2_indices_coefs_table[index][7]);

	// 			// m
	// 			int s =  int(this -> cm_cov_2_indices_coefs_table[index][8]);
	// 			int t =  int(this -> cm_cov_2_indices_coefs_table[index][9]);

	// 			// p
	// 			int u =  int(this -> cm_cov_2_indices_coefs_table[index][10]);
	// 			int v =  int(this -> cm_cov_2_indices_coefs_table[index][11]);

	// 			// q
	// 			int w =  int(this -> cm_cov_2_indices_coefs_table[index][12]);
	// 			int x =  int(this -> cm_cov_2_indices_coefs_table[index][13]);


	// 			auto Ci = patch_e -> get_control_point(i,j);
	// 			auto Cj = patch_e -> get_control_point(k,l);
	// 			auto Ck = patch_e -> get_control_point(m,p);
	// 			auto Cl = patch_e -> get_control_point(q,r);

	// 			auto Cm = patch_f -> get_control_point(s,t);
	// 			auto Cp = patch_f -> get_control_point(u,v);
	// 			auto Cq = patch_f -> get_control_point(w,x);


	// 			arma::mat P;

	// 			ShapeModel::assemble_covariance(P,Ci,Cj,Ck,Cl,Cm,Cp,Cq);

	// 			arma::mat left_mat = patch_e -> get_augmented_cross_products(i,j,k,l,m,p,q,r);
	// 			arma::vec right_vec = patch_f -> get_cross_products(s,t,u,v,w,x);

	// 			arma::mat P_inc = this -> cm_cov_2_indices_coefs_table[index][14] * left_mat.t() * P * right_vec * this -> cm.t();

	// 			cm_cov_temp -= (P_inc + P_inc.t());

	// 		}


	// 	}



	// }


	this -> cm_cov = cm_cov_temp / std::pow(this -> volume,2);




}



arma::vec ShapeModelBezier::run_monte_carlo_volume(int N){

	// Setting the means
	for (unsigned int i = 0; i < this -> get_NControlPoints(); ++i){
		this -> control_points[i] -> set_mean_coordinates();
	}

	boost::progress_display progress(N) ;
	arma::vec results_volume(N);

	for (int iter = 0; iter < N; ++iter){

		++progress;

		for (unsigned int i = 0; i < this -> get_NControlPoints(); ++i){


			if (arma::det(this -> control_points[i] -> get_covariance()) > 0){

				this -> control_points[i] -> set_coordinates(this -> control_points[i] -> get_mean_coordinates()
					+ arma::chol(this -> control_points[i] -> get_covariance(), "lower") * arma::randn<arma::vec>(3) );

			}
		}

		this -> compute_volume();
		results_volume(iter) = this -> get_volume();

	}



	// Cleaning up
	for (unsigned int i = 0; i < this -> get_NControlPoints(); ++i){
		this -> control_points[i] -> set_coordinates(this -> control_points[i] -> get_mean_coordinates());
	}
	



	return results_volume;


}





arma::mat ShapeModelBezier::run_monte_carlo_cm(int N){

	// Setting the means
	for (unsigned int i = 0; i < this -> get_NControlPoints(); ++i){
		this -> control_points[i] -> set_mean_coordinates();
	}

	boost::progress_display progress(N) ;
	arma::mat results_cm(3,N);

	for (int iter = 0; iter < N; ++iter){

		++progress;

		for (unsigned int i = 0; i < this -> get_NControlPoints(); ++i){

			if (arma::det(this -> control_points[i] -> get_covariance()) > 0){

				this -> control_points[i] -> set_coordinates(this -> control_points[i] -> get_mean_coordinates()
					+ arma::chol(this -> control_points[i] -> get_covariance(), "lower") * arma::randn<arma::vec>(3) );

			}
		}

		this -> compute_volume();
		this -> compute_center_of_mass();

		results_cm.col(iter) = this -> get_center_of_mass();

	}



	// Cleaning up
	for (unsigned int i = 0; i < this -> get_NControlPoints(); ++i){
		this -> control_points[i] -> set_coordinates(this -> control_points[i] -> get_mean_coordinates());
	}
	



	return results_cm;


}












bool ShapeModelBezier::ray_trace(Ray * ray){


	return this -> kdt_facet -> hit(this -> get_KDTreeShape().get(),ray,this);

}

void ShapeModelBezier::elevate_degree(bool update){

	// All patches are elevated
	for (unsigned int i = 0; i < this -> get_NElements(); ++i){
		dynamic_cast<Bezier *>(this -> get_elements() -> at(i).get()) -> elevate_degree();

	}

	std::vector<std::shared_ptr<ControlPoint> > new_control_points;
	std::set<std::shared_ptr<ControlPoint> > new_control_points_set;



	for (unsigned int i = 0; i < this -> get_NElements(); ++i){

		auto points = this -> get_elements() -> at(i) -> get_control_points();

		for (auto point = points -> begin(); point != points -> end(); ++point){
			if (new_control_points_set.find(*point) == new_control_points_set.end()){
				new_control_points_set.insert(*point) ;
				new_control_points.push_back(*point);
			}
		}
	}


	// The control point of this shape model are the same as that
	// of the provided shape
	this -> control_points = new_control_points;

	// The ownership relationships are reset
	for (unsigned int i = 0; i < this -> get_NControlPoints(); ++i){
		this -> control_points[i] -> reset_ownership();
	}

	// The surface elements are almost the same, expect that they are 
	// Bezier patches and not facets
	for (auto patch = this -> elements.begin(); patch != this -> elements.end(); ++patch){

		auto points = (*patch) -> get_control_points();

		for (auto point = points -> begin(); point != points -> end(); ++point){

			(*point) -> add_ownership(patch -> get());

		}


	}

	if (update){
		this -> construct_kd_tree_control_points();
		this -> populate_mass_properties_coefs();
		this -> update_mass_properties();
	}


}

void ShapeModelBezier::populate_mass_properties_coefs(){

	this -> cm_gamma_indices_coefs_table.clear();
	this -> volume_indices_coefs_table.clear();
	this -> inertia_indices_coefs_table.clear();
	this -> volume_sd_indices_coefs_table.clear();


	this ->	cm_cov_1_indices_coefs_table.clear();
	this -> cm_cov_2_indices_coefs_table.clear();

	double n = this -> get_degree();

	std::cout << "- Shape degree: " << n << std::endl;

	// Volume
	for (int i = 0; i < 1 + n; ++i){
		for (int j = 0; j < 1 + n - i; ++j){

			for (int k = 0; k < 1 + n ; ++k){
				for (int l = 0; l < 1 + n - k; ++l){

					for (int m = 0; m < 1 + n ; ++m){
						for (int p = 0; p < 1 + n - m; ++p){

							if ((i == k) && (j == l)){
										// the dot product in the quadruple product
										// will be 0
								continue;
							}
							if ((i == m) && (j == p)){
										// the dot product in the quadruple product
										// will be 0
								continue;
							}

							if ((k == m) && (l == p)){
										// the cross product in the quadruple product
										// will be 0
								continue;
							}

							double alpha = Bezier::alpha_ijk(i, j, k, l, m, p, n);
							if (std::abs(alpha) > 1e-13){
								std::vector<double> index_vector = {double(i),double(j),double(k),double(l),double(m),double(p),alpha};
								this -> volume_indices_coefs_table.push_back(index_vector);
							}
							
						}
					}
				}
			}
		}
	}

	std::cout << "- Volume coefficients: " << this -> volume_indices_coefs_table.size() << std::endl;

	// Volume sd
	for (int i = 0; i < 1 + n; ++i){
		for (int j = 0; j < 1 + n - i; ++j){

			for (int k = 0; k < 1 + n ; ++k){
				for (int l = 0; l < 1 + n - k; ++l){

					for (int m = 0; m < 1 + n ; ++m){
						for (int p = 0; p < 1 + n - m; ++p){

							for (int q = 0; q < 1 + n ; ++q){
								for (int r = 0; r < 1 + n - q; ++r){

									for (int s = 0; s < 1 + n ; ++s){
										for (int t = 0; t < 1 + n - s; ++t){

											for (int u = 0; u < 1 + n ; ++u){
												for (int v = 0; v < 1 + n - u; ++v){



													if ((i == k) && (k == m) && (j == l) && (l == p)){
														// the left vector of cross products will be zero
														continue;
													}

													if ((q == s) && (s == u) && (r== t) && (t == v)){
														// the right vector of cross products will be zero
														continue;
													}

													double alpha_1 = Bezier::alpha_ijk(i, j, k, l, m, p, n);
													double alpha_2 = Bezier::alpha_ijk(q, r, s, t, u, v, n);
													double aa = alpha_1 * alpha_2;
													
													if (std::abs(aa) > 1e-13){
														std::vector<double> index_vector = {
															double(i),double(j),
															double(k),double(l),
															double(m),double(p),
															double(q),double(r),
															double(s),double(t),
															double(u),double(v),
															aa
														};
														this -> volume_sd_indices_coefs_table.push_back(index_vector);
													}		




												}
											}
										}
									}
								}

							}
						}
					}
				}
			}
		}
	}


	std::cout << "- Volume SD coefficients: " << this -> volume_sd_indices_coefs_table.size() << std::endl;





	// CM
	// i
	for (int i = 0; i < 1 + n; ++i){
		for (int j = 0; j < 1 + n - i; ++j){
			// j
			for (int k = 0; k < 1 + n ; ++k){
				for (int l = 0; l < 1 + n - k; ++l){
					// k
					for (int m = 0; m < 1 + n ; ++m){
						for (int p = 0; p < 1 + n - m; ++p){
							// l
							for (int q = 0; q < 1 + n ; ++q){
								for (int r = 0; r < 1 + n - q; ++r){

									// if ((m == q) && (p == r)){
									// 	// the cross product in the quadruple product
									// 	// will be 0
									// 	continue;
									// }
									// if ((k == m) && (l == p)){
									// 	// the dot product in the quadruple product
									// 	// will be 0
									// 	continue;
									// }

									// if ((k == q) && (l == r)){
									// 	// the dot product in the quadruple product
									// 	// will be 0
									// 	continue;
									// }


									double gamma = Bezier::gamma_ijkl(i, j, k, l, m, p,q, r, n);
									
									if (std::abs(gamma) > 1e-13){
										std::vector<double> index_vector = {
											double(i),
											double(j),
											double(k),
											double(l),
											double(m),
											double(p),
											double(q),
											double(r),
											gamma
										};
										this -> cm_gamma_indices_coefs_table.push_back(index_vector);
									} 

								}
							}
						}
					}
				}
			}
		}
	}


	std::cout << "- CM coefficients: " << this -> cm_gamma_indices_coefs_table.size() << std::endl;




	// CM covar, 1

	// i
	for (int i = 0; i < 1 + n; ++i){
		for (int j = 0; j < 1 + n - i; ++j){

	// j

			for (int k = 0; k < 1 + n ; ++k){
				for (int l = 0; l < 1 + n - k; ++l){

	// k
					for (int m = 0; m < 1 + n ; ++m){
						for (int p = 0; p < 1 + n - m; ++p){
	// l

							for (int q = 0; q < 1 + n ; ++q){
								for (int r = 0; r < 1 + n - q; ++r){
	// m

									for (int s = 0; s < 1 + n ; ++s){
										for (int t = 0; t < 1 + n - s; ++t){
	// p

											for (int u = 0; u < 1 + n ; ++u){
												for (int v = 0; v < 1 + n - u; ++v){

	// q
													for (int w = 0; w < 1 + n ; ++w){
														for (int x = 0; x < 1 + n - w; ++x){
	// r
															for (int y = 0; y < 1 + n ; ++y){
																for (int z = 0; z < 1 + n - y; ++z){



																	double gamma_1 = Bezier::gamma_ijkl(i, j, k, l, m, p,q, r, n);
																	double gamma_2 = Bezier::gamma_ijkl(s, t, u, v, w, x,y, z, n);

																	double gamma = gamma_1 * gamma_2;

																	if (std::abs(gamma) > 1e-13){
																		std::vector<double> index_vector = {
																			double(i),double(j),
																			double(k),double(l),
																			double(m),double(p),
																			double(q),double(r),
																			double(s),double(t),
																			double(u),double(v),
																			double(w),double(x),
																			double(y),double(z),
																			gamma
																		};
																		this -> cm_cov_1_indices_coefs_table.push_back(index_vector);
																	}
																}
															}
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}

				}
			}
		}
	}





		// CM covar, 2

	// i
	for (int i = 0; i < 1 + n; ++i){
		for (int j = 0; j < 1 + n - i; ++j){

	// j

			for (int k = 0; k < 1 + n ; ++k){
				for (int l = 0; l < 1 + n - k; ++l){

	// k
					for (int m = 0; m < 1 + n ; ++m){
						for (int p = 0; p < 1 + n - m; ++p){
	// l

							for (int q = 0; q < 1 + n ; ++q){
								for (int r = 0; r < 1 + n - q; ++r){
	// m

									for (int s = 0; s < 1 + n ; ++s){
										for (int t = 0; t < 1 + n - s; ++t){
	// p

											for (int u = 0; u < 1 + n ; ++u){
												for (int v = 0; v < 1 + n - u; ++v){

	// q
													for (int w = 0; w < 1 + n ; ++w){
														for (int x = 0; x < 1 + n - w; ++x){



															double gamma= Bezier::gamma_ijkl(i, j, k, l, m, p,q, r, n);
															double alpha = Bezier::alpha_ijk(s, t, u, v, w, x, n);

															double coef = gamma * alpha;

															if (std::abs(coef) > 1e-13){
																std::vector<double> index_vector = {
																	double(i),double(j),
																	double(k),double(l),
																	double(m),double(p),
																	double(q),double(r),
																	double(s),double(t),
																	double(u),double(v),
																	double(w),double(x),
																	coef
																};
																this -> cm_cov_2_indices_coefs_table.push_back(index_vector);
															}
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}

				}
			}

		}
	}







	std::cout << "- CM cov coefficients : " << this -> cm_cov_1_indices_coefs_table.size() + this -> cm_cov_2_indices_coefs_table.size() << std::endl;











	// Inertia
	for (int i = 0; i < 1 + n; ++i){
		for (int j = 0; j < 1 + n - i; ++j){

			for (int k = 0; k < 1 + n ; ++k){
				for (int l = 0; l < 1 + n - k; ++l){

					for (int m = 0; m < 1 + n ; ++m){
						for (int p = 0; p < 1 + n - m; ++p){

							for (int q = 0; q < 1 + n ; ++q){
								for (int r = 0; r < 1 + n - q; ++r){

									for (int s = 0; s < 1 + n ; ++s){
										for (int t = 0; t < 1 + n - s; ++t){


											if ((m == q) && (p == r)){
										// the dot product in the quadruple product
										// will be 0
												continue;
											}
											if ((m == s) && (p == t)){
										// the dot product in the quadruple product
										// will be 0
												continue;
											}

											if ((q == s) && (r == t)){
										// the cross product in the quadruple product
										// will be 0
												continue;
											}






											double kappa = Bezier::kappa_ijklm(i, j, k, l, m, p,q, r,s,t, n);
											if (std::abs(kappa) > 1e-13){
												std::vector<double> index_vector = {double(i),double(j),double(k),double(l),double(m),double(p),double(q),double(r),
													double(s),double(t),kappa};
													this -> inertia_indices_coefs_table.push_back(index_vector);
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}

		std::cout << "- Inertia coefficients: " << this -> inertia_indices_coefs_table.size() + this -> cm_gamma_indices_coefs_table.size() << std::endl;


	}


	void ShapeModelBezier::save_both(std::string partial_path){


		this -> save(partial_path + ".b");

		ShapeModelImporter shape_bezier(partial_path + ".b", 1, true);
		ShapeModelBezier self("",nullptr);

		shape_bezier.load_bezier_shape_model(&self);
		self.elevate_degree(false);
		self.elevate_degree(false);
		self.elevate_degree(false);
		self.elevate_degree(false);
		self.elevate_degree(false);
		self.elevate_degree(false);
		self.elevate_degree(false);

		self.save_to_obj(partial_path + ".obj");


	}

	void ShapeModelBezier::save(std::string path) {
	// An inverse map going from vertex pointer to global indices is created

		std::map<std::shared_ptr<ControlPoint> , unsigned int> pointer_to_global_indices;
		std::map<unsigned int,std::shared_ptr<ControlPoint> > global_index_to_pointer;

		std::vector<arma::vec> vertices;
		std::vector< std::vector<unsigned int> > shape_patch_indices;

	// The global indices of the control points are found. 
		for (unsigned int i = 0; i < this -> get_NElements(); ++i){

			auto patch = this -> get_elements() -> at(i);

			std::vector<unsigned int> patch_indices;

			for (unsigned int index = 0; index < patch -> get_control_points() -> size(); ++index){

				if (pointer_to_global_indices.find(patch -> get_control_points() -> at(index)) == pointer_to_global_indices.end()){
					pointer_to_global_indices[patch -> get_control_points() -> at(index)] = pointer_to_global_indices.size();
					global_index_to_pointer[pointer_to_global_indices.size()] = patch -> get_control_points() -> at(index);
				}

				patch_indices.push_back(pointer_to_global_indices[patch -> get_control_points() -> at(index)]);

			}



			shape_patch_indices.push_back(patch_indices);

		}

	// The coordinates are written to a file
		std::ofstream shape_file;
		shape_file.open(path);
		shape_file << this -> get_degree() << "\n";

		for (auto iter = global_index_to_pointer.begin(); iter != global_index_to_pointer.end(); ++iter){
			shape_file << "v " << iter -> second -> get_coordinates()(0) << " " << iter -> second -> get_coordinates()(1) << " " << iter -> second -> get_coordinates()(2) << "\n";
		}

		for (auto iter = shape_patch_indices.begin(); iter != shape_patch_indices.end(); ++iter){
			shape_file << "f ";

			for (unsigned int index = 0; index < iter -> size(); ++index){

				if (index != iter -> size() - 1){
					shape_file << iter -> at(index) << " ";
				}
				else if (index == iter -> size() - 1 && iter != shape_patch_indices.end() - 1 ){
					shape_file << iter -> at(index) << "\n";
				}
				else{
					shape_file << iter -> at(index);
				}
			}

		}

	}




	void ShapeModelBezier::construct_kd_tree_shape(){

		std::chrono::time_point<std::chrono::system_clock> start, end;
		start = std::chrono::system_clock::now();


	// The KD tree is constructed by building an "enclosing" (not strictly-speaking) KD tree from the bezier shape


	// An inverse map going from vertex pointer to global indices is created
	// Note that the actual vertices on the shape model will not be be 
	// the control points, but the points lying on the bezier patch
 	// they support

		std::vector<std::shared_ptr<Element > > facets;


		for (unsigned int i = 0; i < this -> get_NElements(); ++i){

			Bezier * patch = dynamic_cast<Bezier * >(this -> get_elements() -> at(i).get());


	// The facets are created

			for (unsigned int l = 0; l < patch -> get_degree(); ++l){

				for (unsigned int t = 0; t < l + 1; ++t){

					if (t <= l){

						std::shared_ptr<ControlPoint> v0 = patch -> get_control_point(patch -> get_degree() - l,l - t);
						std::shared_ptr<ControlPoint> v1 = patch -> get_control_point(patch -> get_degree() - l - 1,l - t + 1);
						std::shared_ptr<ControlPoint> v2 = patch -> get_control_point(patch -> get_degree() - l - 1,l-t);


						std::vector<std::shared_ptr<ControlPoint>> vertices;
						vertices.push_back(v0);
						vertices.push_back(v1);
						vertices.push_back(v2);

						std::shared_ptr<Element> facet = std::make_shared<Facet>(Facet(vertices));
						facet -> set_super_element(this -> get_elements() -> at(i).get());
						facets.push_back(facet);
					}

					if (t > 0 ){

						std::shared_ptr<ControlPoint> v0 = patch -> get_control_point(patch -> get_degree() - l,l-t);
						std::shared_ptr<ControlPoint> v1 = patch -> get_control_point(patch -> get_degree() - l,l - t + 1 );
						std::shared_ptr<ControlPoint> v2 = patch -> get_control_point(patch -> get_degree() - l -1,l - t + 1);


						std::vector<std::shared_ptr<ControlPoint>> vertices;

						vertices.push_back(v0);
						vertices.push_back(v1);
						vertices.push_back(v2);


						std::shared_ptr<Element> facet = std::make_shared<Facet>(Facet(vertices));
						facet -> set_super_element(this -> get_elements() -> at(i).get());

						facets.push_back(facet);

					}

				}

			}
		}




		this -> kdt_facet = std::make_shared<KDTreeShape>(KDTreeShape());
		this -> kdt_facet = this -> kdt_facet -> build(facets, 0);


		end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;


		std::cout << "\n Elapsed time during Bezier KDTree construction : " << elapsed_seconds.count() << "s\n\n";










	}


	unsigned int ShapeModelBezier::get_degree(){
		if (this -> get_elements() -> size() == 0){
			throw(std::runtime_error("This bezier shape model has no elements"));
		}
		return dynamic_cast<Bezier * >(this -> elements. begin() -> get()) -> get_degree();
	}


	void ShapeModelBezier::save_to_obj(std::string path) {

	// An inverse map going from vertex pointer to global indices is created

	// Note that the actual vertices on the shape model will not be be 
	// the control points, but the points lying on the bezier patch
 	// they support

		std::map<std::shared_ptr<ControlPoint> , unsigned int> pointer_to_global_indices;
		std::vector<arma::vec> vertices;
		std::vector<std::tuple<std::shared_ptr<ControlPoint>,std::shared_ptr<ControlPoint>,std::shared_ptr<ControlPoint> > > facets;


	// The global indices of the control points are found. 
		for (unsigned int i = 0; i < this -> get_NElements(); ++i){

			Bezier * patch = dynamic_cast<Bezier * >(this -> get_elements() -> at(i).get());

			for (unsigned int index = 0; index < patch -> get_control_points() -> size(); ++index){

				if (pointer_to_global_indices.find(patch -> get_control_points() -> at(index))== pointer_to_global_indices.end()){
					pointer_to_global_indices[patch -> get_control_points() -> at(index)] = pointer_to_global_indices.size();

					auto local_indices = patch -> get_local_indices(patch -> get_control_points() -> at(index));
					double u =  double(std::get<0>(local_indices)) / patch -> get_degree();
					double v =  double(std::get<1>(local_indices)) / patch -> get_degree();

					arma::vec surface_point = patch -> evaluate(u,v);
					vertices.push_back(surface_point);
				}

			}


	// The facets are created

			for (unsigned int l = 0; l < patch -> get_degree(); ++l){

				for (unsigned int t = 0; t < l + 1; ++t){

					if (t <= l){

						std::shared_ptr<ControlPoint> v0 = patch -> get_control_point(patch -> get_degree() - l,l - t);
						std::shared_ptr<ControlPoint> v1 = patch -> get_control_point(patch -> get_degree() - l - 1,l - t + 1);
						std::shared_ptr<ControlPoint> v2 = patch -> get_control_point(patch -> get_degree() - l - 1,l-t);

						facets.push_back(std::make_tuple(v0,v1,v2));
					}

					if (t > 0 ){

						std::shared_ptr<ControlPoint> v0 = patch -> get_control_point(patch -> get_degree() - l,l-t);
						std::shared_ptr<ControlPoint> v1 = patch -> get_control_point(patch -> get_degree() - l,l - t + 1 );
						std::shared_ptr<ControlPoint> v2 = patch -> get_control_point(patch -> get_degree() - l -1,l - t + 1);

						facets.push_back(std::make_tuple(v0,v1,v2));
					}

				}

			}
		}

	// The coordinates are written to a file

		std::ofstream shape_file;
		shape_file.open(path);

		for (unsigned int i = 0; i < vertices.size(); ++i){
			shape_file << "v " << vertices[i](0) << " " << vertices[i](1) << " " << vertices[i](2) << "\n";
		}

		for (unsigned int i = 0; i < facets.size(); ++i){
			unsigned int indices[3];
			indices[0] = pointer_to_global_indices[std::get<0>(facets[i])] + 1;
			indices[1] = pointer_to_global_indices[std::get<1>(facets[i])] + 1;
			indices[2] = pointer_to_global_indices[std::get<2>(facets[i])] + 1;


			shape_file << "f " << indices[0] << " " << indices[1] << " " << indices[2] << "\n";

		}


	}






