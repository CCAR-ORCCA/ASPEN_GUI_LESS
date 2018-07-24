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
		patch -> set_global_index(i);

		this -> elements.push_back(patch);

		patch -> get_control_points() -> at(0) -> add_ownership(patch.get());
		patch -> get_control_points() -> at(1) -> add_ownership(patch.get());
		patch -> get_control_points() -> at(2) -> add_ownership(patch.get());

	}
	this -> build_structure();

	this -> construct_kd_tree_control_points();
	this -> populate_mass_properties_coefs();
	this -> update_mass_properties();

}


ShapeModelBezier::ShapeModelBezier(ShapeModelTri * shape_model,
	std::string ref_frame_name,
	FrameGraph * frame_graph,
	double surface_noise): ShapeModel(ref_frame_name,frame_graph){

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
		patch -> set_global_index(i);
		this -> elements.push_back(patch);

		patch -> get_control_points() -> at(0) -> add_ownership(patch.get());
		patch -> get_control_points() -> at(1) -> add_ownership(patch.get());
		patch -> get_control_points() -> at(2) -> add_ownership(patch.get());
		patch -> set_P_X(surface_noise * surface_noise * arma::eye<arma::mat>(9,9));

	}
	this -> build_structure();

	this -> construct_kd_tree_control_points();
	this -> populate_mass_properties_coefs();
	this -> update_mass_properties();

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
	this -> control_points = (*this -> elements[0] -> get_control_points());

	// The ownership relationships are reset
	for (unsigned int i = 0; i < this -> get_NControlPoints(); ++i){
		this -> control_points[i] -> reset_ownership();
		this -> control_points[i] -> add_ownership(this -> elements[0].get());
	}
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






double ShapeModelBezier::compute_volume_omp(const arma::vec & deviation) const{
	
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
			
			volume += this -> volume_indices_coefs_table[index][6] * patch -> triple_product(i,j,k,l,m,p,deviation);

		}

	}
	return volume;
}








void ShapeModelBezier::compute_center_of_mass(){

	this -> cm = arma::zeros<arma::vec>(3);

	double cx = 0;
	double cy = 0;
	double cz = 0;

	int n = this -> get_degree();

	#pragma omp parallel for reduction(+:cx,cy,cz)
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


arma::vec::fixed<3> ShapeModelBezier::compute_center_of_mass_omp(const double & volume, const arma::vec & deviation) const{

	double cx = 0;
	double cy = 0;
	double cz = 0;

	int n = this -> get_degree();

	#pragma omp parallel for reduction(+:cx,cy,cz)
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

			auto Ci = patch -> get_control_point(i,j);
			
			int i_g = Ci -> get_global_index();
			
			arma::vec result = (Ci -> get_coordinates() 
				+ deviation.rows(3 * i_g,3 * i_g + 2)) * patch -> triple_product(k,l,m,p,q,r,deviation);

			cx += this -> cm_gamma_indices_coefs_table[index][8] * result(0);
			cy += this -> cm_gamma_indices_coefs_table[index][8] * result(1);
			cz += this -> cm_gamma_indices_coefs_table[index][8] * result(2);


		}

	}



	arma::vec cm = {cx,cy,cz};

	return ( cm / volume);
}









void ShapeModelBezier::compute_inertia(){

	arma::mat::fixed<3,3> inertia = arma::zeros<arma::mat>(3,3);

	double norm_length = std::cbrt(this -> volume);
	// double factor = 1./std::pow(norm_length,5);
	double factor = 1.;

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



arma::mat::fixed<3,3> ShapeModelBezier::compute_inertia_omp(const arma::vec & deviation) const{

	arma::mat::fixed<3,3> inertia = arma::zeros<arma::mat>(3,3);

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


			auto Ci = patch -> get_control_point(i,j);
			auto Cj = patch -> get_control_point(k,l);

			int i_g = Ci -> get_global_index();
			int j_g = Cj -> get_global_index();

			inertia += (this -> inertia_indices_coefs_table[index][10]  
				* RBK::tilde(Ci -> get_coordinates() + deviation.rows(3 * i_g, 3 * i_g + 2)) 
				* RBK::tilde(Cj -> get_coordinates() + deviation.rows(3 * j_g, 3 * j_g + 2))
				* patch -> triple_product(m,p,q,r,s,t,deviation));

		}

	}
	return inertia;

}


void ShapeModelBezier::find_correlated_elements(){


	this -> correlated_elements.clear();

	for (unsigned int e = 0; e < this -> get_NElements(); ++e){

		Bezier * patch_e = static_cast<Bezier *>(this -> get_element(e).get());


		std::vector < int > elements_correlated_with_e ;

		for (unsigned int f = 0; f < this -> get_NElements(); ++f){

			Bezier * patch_f = static_cast<Bezier *>(this -> get_element(f).get());

			for (unsigned int i = 0; i < patch_e -> get_control_points() -> size(); ++i){
				auto Ci = patch_e -> get_control_points() -> at(i);
				int i_g = Ci -> get_global_index();
				
				for (unsigned int j = 0; j < patch_f -> get_control_points() -> size(); ++j){
					auto Cj = patch_e -> get_control_points() -> at(j);
					int j_g = Cj -> get_global_index();

					// If true, these two patches are correlated
					if (arma::abs(this -> get_point_covariance(i_g,j_g)).max() > 0 
						&& (std::find(elements_correlated_with_e.begin(),
							elements_correlated_with_e.end(),
							f) == elements_correlated_with_e.end() )){

						elements_correlated_with_e.push_back(f);

					}

				}

			}



		}

		this -> correlated_elements.push_back(elements_correlated_with_e);

	}

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

	std::cout << "\n- Computing volume sd...\n";
	boost::progress_display progress(this -> elements.size()) ;

	#pragma omp parallel for reduction(+:vol_sd)

	for (unsigned int e = 0; e < this -> elements.size(); ++e) {
		
		Bezier * patch_e = static_cast<Bezier * >(this -> elements[e].get());

		auto neighbors = this -> correlated_elements[e];

		for (auto f : neighbors) {

			Bezier * patch_f = static_cast<Bezier * >(this -> elements[f].get());

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


				int i_g = patch_e -> get_control_point_global_index(i,j);
				int j_g = patch_e -> get_control_point_global_index(k,l);
				int k_g = patch_e -> get_control_point_global_index(m,p);

				int l_g = patch_f -> get_control_point_global_index(q,r);
				int m_g = patch_f -> get_control_point_global_index(s,t);
				int p_g = patch_f -> get_control_point_global_index(u,v);


				arma::vec::fixed<9> left_vec = patch_e -> get_cross_products(i,j,k,l,m,p);
				arma::vec::fixed<9> right_vec = patch_f -> get_cross_products(q,r,s,t,u,v);

				vol_sd += this -> volume_sd_indices_coefs_table[index][12] * this -> increment_volume_variance(left_vec,
					right_vec, 
					i_g, j_g, k_g, 
					l_g,  m_g, p_g);

			}


		}
		++progress;


	}

	this -> volume_sd = std::sqrt(vol_sd);

}



double ShapeModelBezier::increment_volume_variance(const arma::vec::fixed<9> & left_vec,
	const arma::vec::fixed<9> & right_vec, 
	int i,int j,int k, 
	int l, int m, int p){

	arma::mat::fixed<9,9> P_CC = arma::zeros<arma::mat>(9,9);

	P_CC.submat(0,0,2,2) = this -> get_point_covariance(i, l);
	P_CC.submat(0,3,2,5) = this -> get_point_covariance(i, m);
	P_CC.submat(0,6,2,8) = this -> get_point_covariance(i, p);

	P_CC.submat(3,0,5,2) = this -> get_point_covariance(j, l);
	P_CC.submat(3,3,5,5) = this -> get_point_covariance(j, m);
	P_CC.submat(3,6,5,8) = this -> get_point_covariance(j, p);

	P_CC.submat(6,0,8,2) = this -> get_point_covariance(k, l);
	P_CC.submat(6,3,8,5) = this -> get_point_covariance(k, m);
	P_CC.submat(6,6,8,8) = this -> get_point_covariance(k, p);

	return arma::dot(left_vec, P_CC * right_vec);
}














void ShapeModelBezier::compute_cm_cov(){

	std::cout << "\n- Computing cm covariance ...\n";

	arma::mat::fixed<3,3> cm_cov_temp;

	
	cm_cov_temp = this -> cm * this -> cm.t() * std::pow(this -> volume_sd,2);

	std::vector<std::set < Element * > > connected_elements;

	for (unsigned int e = 0; e < this -> elements.size(); ++e) {

		auto elements = this -> elements[e] -> get_neighbors(true);
		connected_elements.push_back(elements);
	}

	boost::progress_display progress(this -> cm_cov_1_indices_coefs_table.size()) ;


	#pragma omp parallel for reduction (+:cm_cov_temp)

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

		arma::mat::fixed<12,3> left_mat;
		arma::mat::fixed<12,3> right_mat;
		
		for (unsigned int e = 0; e < this -> elements.size(); ++e) {
			
			Bezier * patch_e = static_cast<Bezier * >(this -> elements[e].get());
			int i_g,j_g,k_g,l_g;

			i_g = patch_e -> get_control_point_global_index(i,j);
			j_g = patch_e -> get_control_point_global_index(k,l);
			k_g = patch_e -> get_control_point_global_index(m,p);
			l_g = patch_e -> get_control_point_global_index(q,r);
			
			auto neighbors = connected_elements[e];

			this -> construct_cm_mapping_mat(left_mat,i_g,j_g,k_g,l_g);
			
			for (unsigned int f = 0; f < this -> elements.size(); ++f) {

				Bezier * patch_f = static_cast<Bezier * >(this -> elements[f].get());

				int m_g,p_g,q_g,r_g;

				m_g = patch_f -> get_control_point_global_index(s,t);
				p_g = patch_f -> get_control_point_global_index(u,v);
				q_g = patch_f -> get_control_point_global_index(w,x);
				r_g = patch_f -> get_control_point_global_index(y,z);	

				this -> construct_cm_mapping_mat(right_mat,m_g,p_g,q_g,r_g);

				cm_cov_temp += coefs_row[16] * this -> increment_cm_cov(left_mat,
					right_mat, 
					i_g,j_g,k_g,l_g, 
					m_g,p_g,q_g,r_g);
			}
		}

		++progress;
	}

	this -> cm_cov = cm_cov_temp / std::pow(this -> volume,2);

}


void ShapeModelBezier::construct_cm_mapping_mat(arma::mat::fixed<12,3> & mat,
	int i,int j,int k,int l) {

	arma::vec * Ci = this -> control_points.at(i) -> get_coordinates_pointer_arma();
	arma::vec * Cj = this -> control_points.at(j) -> get_coordinates_pointer_arma();
	arma::vec * Ck = this -> control_points.at(k) -> get_coordinates_pointer_arma();
	arma::vec * Cl = this -> control_points.at(l) -> get_coordinates_pointer_arma();

	mat.submat(0,0,2,2) = arma::eye<arma::mat>(3,3) * arma::dot(*Cj,arma::cross(*Ck,*Cl));
	mat.submat(3,0,5,2) = arma::cross(*Ck,*Cl) * Ci -> t();
	mat.submat(6,0,8,2) = arma::cross(*Cl,*Cj) * Ci -> t();
	mat.submat(9,0,11,2) = arma::cross(*Cj,*Ck) * Ci -> t();



}



void ShapeModelBezier::construct_inertia_mapping_mat(arma::mat::fixed<6,15> & mat,
	int i,int j,int k,int l,int m) const{

	arma::vec * Ci = this -> control_points.at(i) -> get_coordinates_pointer_arma();
	arma::vec * Cj = this -> control_points.at(j) -> get_coordinates_pointer_arma();
	arma::vec * Ck = this -> control_points.at(k) -> get_coordinates_pointer_arma();
	arma::vec * Cl = this -> control_points.at(l) -> get_coordinates_pointer_arma();
	arma::vec * Cm = this -> control_points.at(m) -> get_coordinates_pointer_arma();

	mat.row(0) = ShapeModelBezier::L_row(0,0,Ci,Cj,Ck,Cl,Cm);
	mat.row(1) = ShapeModelBezier::L_row(1,1,Ci,Cj,Ck,Cl,Cm);
	mat.row(2) = ShapeModelBezier::L_row(2,2,Ci,Cj,Ck,Cl,Cm);
	mat.row(3) = ShapeModelBezier::L_row(0,1,Ci,Cj,Ck,Cl,Cm);
	mat.row(4) = ShapeModelBezier::L_row(0,2,Ci,Cj,Ck,Cl,Cm);
	mat.row(5) = ShapeModelBezier::L_row(1,2,Ci,Cj,Ck,Cl,Cm);

	// mat.fill(1);
}

arma::rowvec::fixed<15> ShapeModelBezier::L_row(int q, int r, const arma::vec * Ci,const arma::vec * Cj,const arma::vec * Ck,const arma::vec * Cl,const arma::vec * Cm){


	arma::vec L_col(15);
	arma::vec e_q = arma::zeros<arma::vec>(3);
	e_q(q) = 1;
	arma::vec e_r = arma::zeros<arma::vec>(3);
	e_r(r) = 1;

	L_col.rows(0,2) = - arma::dot(*Ck,arma::cross(*Cl,*Cm)) * RBK::tilde(e_r) * RBK::tilde(*Cj) * e_q;
	L_col.rows(3,5) = - arma::dot(*Ck,arma::cross(*Cl,*Cm)) * RBK::tilde(e_q) * RBK::tilde(*Ci) * e_r;
	L_col.rows(6,8) = arma::dot(e_r, RBK::tilde(*Ci) * RBK::tilde(*Cj) * e_q ) * arma::cross(*Cl,*Cm);
	L_col.rows(9,11) = arma::dot(e_r, RBK::tilde(*Ci) * RBK::tilde(*Cj) * e_q ) * arma::cross(*Cm,*Ck);
	L_col.rows(12,14) = arma::dot(e_r, RBK::tilde(*Ci) * RBK::tilde(*Cj) * e_q ) * arma::cross(*Ck,*Cl);

	return L_col.t();
}




void ShapeModelBezier::apply_deviation(){

	// Setting the means
	for (unsigned int i = 0; i < this -> get_NControlPoints(); ++i){
		this -> control_points[i] -> set_mean_coordinates();
		auto coords = this -> control_points[i] -> get_mean_coordinates();
		this -> control_points[i] -> set_coordinates(coords + this -> control_points[i] -> get_deviation());
	}

}






void ShapeModelBezier::run_monte_carlo(int N,
	arma::vec & results_volume,
	arma::mat & results_cm,
	arma::mat & results_inertia,
	arma::mat & results_moments,
	arma::mat & results_mrp,
	arma::mat & results_lambda_I,
	arma::mat & results_eigenvectors,
	arma::mat & results_Evectors,
	arma::mat & results_Y,
	arma::mat & results_MI,
	arma::mat & results_dims){

	arma::arma_rng::set_seed(0);

	// Setting the means
	for (unsigned int i = 0; i < this -> get_NControlPoints(); ++i){
		this -> control_points[i] -> set_mean_coordinates();
	}

	boost::progress_display progress(N) ;
	results_volume = arma::zeros<arma::vec>(N);
	results_cm = arma::zeros<arma::mat>(3,N);
	results_inertia = arma::zeros<arma::mat>(6,N);
	results_moments = arma::zeros<arma::mat>(4,N);
	results_mrp = arma::zeros<arma::mat>(3,N);
	results_lambda_I = arma::zeros<arma::mat>(7,N);
	results_eigenvectors = arma::zeros<arma::mat>(9,N);
	results_Evectors = arma::zeros<arma::mat>(9,N);
	results_Y = arma::zeros<arma::mat>(4,N);
	results_MI = arma::zeros<arma::mat>(7,N);
	results_dims = arma::zeros<arma::mat>(3,N);

	this -> take_and_save_zslice("slice_baseline.txt",0);
	this -> save_to_obj("iter_baseline.obj");


	for (int iter = 0; iter < N; ++iter){

		++progress;

		arma::vec deviation = this -> shape_covariance_sqrt * arma::randn<arma::vec>(3 * this -> get_NControlPoints());

		for (unsigned int i = 0; i < this -> get_NControlPoints(); ++i){
			
			this -> control_points[i] -> set_coordinates(this -> control_points[i] -> get_mean_coordinates()
				+ deviation.rows(3 * i, 3 * i + 2) );

		}

		this -> compute_volume();
		this -> compute_center_of_mass();
		this -> compute_inertia();

		arma::mat I_C = this -> inertia - this -> get_volume() * RBK::tilde(this -> get_center_of_mass()) * RBK::tilde(this -> get_center_of_mass()).t() ;
		arma::vec I = {I_C(0,0),I_C(1,1),I_C(2,2),I_C(0,1),I_C(0,2),I_C(1,2)};

		arma::vec moments_col(4);
		arma::vec eig_val = arma::eig_sym(I_C);
		arma::mat eig_vec =  ShapeModelBezier::get_principal_axes_stable(I_C);
		moments_col.rows(0,2) = eig_val;
		moments_col(3) = this -> get_volume();

		results_volume(iter) = this -> get_volume();
		results_cm.col(iter) = this -> get_center_of_mass();
		results_inertia.col(iter) = I;
		results_moments.col(iter) = moments_col;
		results_mrp.col(iter) = RBK::dcm_to_mrp(eig_vec);

		results_lambda_I.col(iter).rows(0,5) = I;
		results_lambda_I.col(iter)(6) = eig_val(2);

		results_eigenvectors.col(iter).rows(0,2) = eig_vec.col(0);
		results_eigenvectors.col(iter).rows(3,5) = eig_vec.col(1);
		results_eigenvectors.col(iter).rows(6,8) = eig_vec.col(2);
		results_Evectors.col(iter) = ShapeModelBezier::get_E_vectors(I_C);
		results_Y.col(iter) = ShapeModelBezier::get_Y(this -> get_volume(),I_C);
		results_MI.col(iter).rows(0,5) = I;
		results_MI.col(iter)(6) = this -> volume;

		results_dims.col(iter) = ShapeModelBezier::get_dims(this -> volume,I_C);

		// saving shape model

		if (iter < 20){
			this -> take_and_save_zslice("slice_" + std::to_string(iter) + ".txt",0);
			this -> save_to_obj("iter_" + std::to_string(iter) + ".obj");
		}


	}

	// Cleaning up
	for (unsigned int i = 0; i < this -> get_NControlPoints(); ++i){
		this -> control_points[i] -> set_coordinates(this -> control_points[i] -> get_mean_coordinates());
	}


}




void ShapeModelBezier::run_monte_carlo_omp(int N,
	arma::vec & results_volume,
	arma::mat & results_cm,
	arma::mat & results_inertia,
	arma::mat & results_moments,
	arma::mat & results_mrp,
	arma::mat & results_lambda_I,
	arma::mat & results_eigenvectors,
	arma::mat & results_Evectors,
	arma::mat & results_Y,
	arma::mat & results_MI,
	arma::mat & results_dims) const{

	arma::arma_rng::set_seed(0);

	// Setting the means
	for (unsigned int i = 0; i < this -> get_NControlPoints(); ++i){
		this -> control_points[i] -> set_mean_coordinates();
	}

	boost::progress_display progress(N) ;
	results_volume = arma::zeros<arma::vec>(N);
	results_cm = arma::zeros<arma::mat>(3,N);
	results_inertia = arma::zeros<arma::mat>(6,N);
	results_moments = arma::zeros<arma::mat>(4,N);
	results_mrp = arma::zeros<arma::mat>(3,N);
	results_lambda_I = arma::zeros<arma::mat>(7,N);
	results_eigenvectors = arma::zeros<arma::mat>(9,N);
	results_Evectors = arma::zeros<arma::mat>(9,N);
	results_Y = arma::zeros<arma::mat>(4,N);
	results_MI = arma::zeros<arma::mat>(7,N);
	results_dims = arma::zeros<arma::mat>(3,N);

	this -> take_and_save_zslice("slice_baseline.txt",0);
	this -> save_to_obj("iter_baseline.obj");

	#pragma omp parallel for
	for (int iter = 0; iter < N; ++iter){

		arma::vec deviation = this -> shape_covariance_sqrt * arma::randn<arma::vec>(3 * this -> get_NControlPoints());

		double volume = this -> compute_volume_omp(deviation);
		arma::vec::fixed<3> cm = this -> compute_center_of_mass_omp(volume,deviation);
		arma::mat::fixed<3,3> inertia = this -> compute_inertia_omp(deviation);

		arma::mat I_C = inertia - volume * RBK::tilde(cm) * RBK::tilde(cm).t() ;
		arma::vec I = {I_C(0,0),I_C(1,1),I_C(2,2),I_C(0,1),I_C(0,2),I_C(1,2)};

		arma::vec moments_col(4);
		arma::vec eig_val = arma::eig_sym(I_C);
		arma::mat eig_vec =  ShapeModelBezier::get_principal_axes_stable(I_C);
		moments_col.rows(0,2) = eig_val;
		moments_col(3) = volume;

		results_volume(iter) = volume;
		results_cm.col(iter) = cm;
		results_inertia.col(iter) = I;
		results_moments.col(iter) = moments_col;
		results_mrp.col(iter) = RBK::dcm_to_mrp(eig_vec);

		results_lambda_I.col(iter).rows(0,5) = I;
		results_lambda_I.col(iter)(6) = eig_val(2);

		results_eigenvectors.col(iter).rows(0,2) = eig_vec.col(0);
		results_eigenvectors.col(iter).rows(3,5) = eig_vec.col(1);
		results_eigenvectors.col(iter).rows(6,8) = eig_vec.col(2);
		results_Evectors.col(iter) = ShapeModelBezier::get_E_vectors(I_C);
		results_Y.col(iter) = ShapeModelBezier::get_Y(volume,I_C);
		results_MI.col(iter).rows(0,5) = I;
		results_MI.col(iter)(6) = volume;

		results_dims.col(iter) = ShapeModelBezier::get_dims(volume,I_C);

		// saving shape model

		if (iter < 20){
			this -> take_and_save_zslice_omp("slice_" + std::to_string(iter) + ".txt",0,deviation);
			this -> save_to_obj_omp("iter_" + std::to_string(iter) + ".obj",deviation);
		}

		++progress;

	}



}

































arma::vec::fixed<3> ShapeModelBezier::get_dims(const double & volume,
	const arma::mat::fixed<3,3> & I_C){


	arma::vec moments = arma::eig_sym(I_C);

	double A = moments(0);
	double B = moments(1);
	double C = moments(2);

	arma::vec::fixed<3> dims = {
		std::sqrt(B + C - A),
		std::sqrt(A + C - B),
		std::sqrt(A + B - C)
	};

	return std::sqrt(5./(2 * volume)) * dims;



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




	this -> build_structure();




	if (update){
		this -> construct_kd_tree_control_points();
		this -> populate_mass_properties_coefs();
		this -> update_mass_properties();
	}


}


void ShapeModelBezier::build_structure() {

// The ownership relationships are reset
	for (unsigned int i = 0; i < this -> get_NControlPoints(); ++i){
		this -> control_points[i] -> reset_ownership();
		this -> control_points[i] -> set_global_index(i);

	}

	// The surface elements are almost the same, expect that they are 
	// Bezier patches and not facets
	for (auto patch = this -> elements.begin(); patch != this -> elements.end(); ++patch){

		auto points = (*patch) -> get_control_points();

		for (auto point = points -> begin(); point != points -> end(); ++point){

			(*point) -> add_ownership(patch -> get());

		}


	}



}





void ShapeModelBezier::populate_mass_properties_coefs(){

	this -> cm_gamma_indices_coefs_table.clear();
	this -> volume_indices_coefs_table.clear();
	this -> inertia_indices_coefs_table.clear();
	this -> volume_sd_indices_coefs_table.clear();

	this ->	cm_cov_1_indices_coefs_table.clear();
	this -> cm_cov_2_indices_coefs_table.clear();
	this -> inertia_stats_1_indices_coefs_table.clear();
	this -> inertia_stats_2_indices_coefs_table.clear();

	double n = this -> get_degree();

	std::cout << "- Shape degree: " << n << std::endl;


	std::vector<std::vector<int> >  base_vector;
	ShapeModelBezier::build_bezier_base_index_vector(n,base_vector);
	std::vector<std::vector<std::vector<int> > > index_vectors;
	std::vector < std::vector<int> > temp_vector;

	// Volume
	ShapeModelBezier::build_bezier_index_vectors(3,
		base_vector,
		index_vectors);


	for (auto vector : index_vectors){

		int i = vector[0][0];
		int j = vector[0][1];
		int k = vector[1][0];
		int l = vector[1][1];
		int m = vector[2][0];
		int p = vector[2][1];

		double alpha = Bezier::alpha_ijk(i, j, k, l, m, p, n);
		std::vector<double> index_vector = {double(i),double(j),double(k),double(l),double(m),double(p),alpha};
		this -> volume_indices_coefs_table.push_back(index_vector);

	}




	std::cout << "- Volume coefficients: " << this -> volume_indices_coefs_table.size() << std::endl;

	// Volume sd
	index_vectors.clear();
	ShapeModelBezier::build_bezier_index_vectors(6,
		base_vector,
		index_vectors);

	for (auto vector : index_vectors){

		int i = vector[0][0];
		int j = vector[0][1];
		int k = vector[1][0];
		int l = vector[1][1];
		int m = vector[2][0];
		int p = vector[2][1];

		int q = vector[3][0];
		int r = vector[3][1];
		int s = vector[4][0];
		int t = vector[4][1];
		int u = vector[5][0];
		int v = vector[5][1];

		double alpha_1 = Bezier::alpha_ijk(i, j, k, l, m, p, n);
		double alpha_2 = Bezier::alpha_ijk(q, r, s, t, u, v, n);
		double aa = alpha_1 * alpha_2;

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


	std::cout << "- Volume SD coefficients: " << this -> volume_sd_indices_coefs_table.size() << std::endl;

	// CM
	// i


	index_vectors.clear();
	ShapeModelBezier::build_bezier_index_vectors(4,
		base_vector,
		index_vectors);

	for (auto vector : index_vectors){

		int i = vector[0][0];
		int j = vector[0][1];
		int k = vector[1][0];
		int l = vector[1][1];
		int m = vector[2][0];
		int p = vector[2][1];
		int q = vector[3][0];
		int r = vector[3][1];

		double gamma = Bezier::gamma_ijkl(i, j, k, l, m, p,q, r, n);

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
		} ;
		this -> cm_gamma_indices_coefs_table.push_back(index_vector);

	}


	std::cout << "- CM coefficients: " << this -> cm_gamma_indices_coefs_table.size() << std::endl;




	// CM covar, 1


	index_vectors.clear();
	ShapeModelBezier::build_bezier_index_vectors(8,
		base_vector,
		index_vectors);



	for (auto vector : index_vectors){

		int i = vector[0][0];
		int j = vector[0][1];
		int k = vector[1][0];
		int l = vector[1][1];
		int m = vector[2][0];
		int p = vector[2][1];
		int q = vector[3][0];
		int r = vector[3][1];

		int s = vector[4][0];
		int t = vector[4][1];
		int u = vector[5][0];
		int v = vector[5][1];
		int w = vector[6][0];
		int x = vector[6][1];
		int y = vector[7][0];
		int z = vector[7][1];


		double gamma_1 = Bezier::gamma_ijkl(i, j, k, l, m, p,q, r, n);
		double gamma_2 = Bezier::gamma_ijkl(s, t, u, v, w, x,y, z, n);

		double gamma = gamma_1 * gamma_2;

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






		// CM covar, 2

	index_vectors.clear();
	ShapeModelBezier::build_bezier_index_vectors(7,
		base_vector,
		index_vectors);



	for (auto vector : index_vectors){

		int i = vector[0][0];
		int j = vector[0][1];
		int k = vector[1][0];
		int l = vector[1][1];
		int m = vector[2][0];
		int p = vector[2][1];
		int q = vector[3][0];
		int r = vector[3][1];

		int s = vector[4][0];
		int t = vector[4][1];
		int u = vector[5][0];
		int v = vector[5][1];
		int w = vector[6][0];
		int x = vector[6][1];


		double gamma = Bezier::gamma_ijkl(i, j, k, l, m, p,q, r, n);
		double alpha = Bezier::alpha_ijk(s, t, u, v, w, x, n);

		double coef = gamma * alpha;

		if (std::abs(coef) > 0){
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
	std::cout << "- CM cov coefficients : " << this -> cm_cov_1_indices_coefs_table.size() + this -> cm_cov_2_indices_coefs_table.size() << std::endl;


	// Inertia

	index_vectors.clear();
	ShapeModelBezier::build_bezier_index_vectors(5,
		base_vector,
		index_vectors);

	for (auto vector : index_vectors){

		int i = vector[0][0];
		int j = vector[0][1];
		int k = vector[1][0];
		int l = vector[1][1];
		int m = vector[2][0];
		int p = vector[2][1];
		int q = vector[3][0];
		int r = vector[3][1];
		int s = vector[4][0];
		int t = vector[4][1];

		double kappa = Bezier::kappa_ijklm(i, j, k, l, m, p,q, r,s,t, n);

		if (std::abs(kappa) > 0){

			std::vector<double> index_vector = {double(i),double(j),double(k),double(l),double(m),double(p),double(q),double(r),
				double(s),double(t),kappa
			};

			this -> inertia_indices_coefs_table.push_back(index_vector);
		}
	}



	std::cout << "- Inertia coefficients: " << this -> inertia_indices_coefs_table.size() << std::endl;






	// Inertia statistics 1

	index_vectors.clear();
	ShapeModelBezier::build_bezier_index_vectors(10,
		base_vector,
		index_vectors);

	for (auto vector : index_vectors){

		int i = vector[0][0];
		int j = vector[0][1];
		int k = vector[1][0];
		int l = vector[1][1];
		int m = vector[2][0];
		int p = vector[2][1];
		int q = vector[3][0];
		int r = vector[3][1];
		int s = vector[4][0];
		int t = vector[4][1];

		int u = vector[5][0];
		int v = vector[5][1];
		int w = vector[6][0];
		int x = vector[6][1];
		int y = vector[7][0];
		int z = vector[7][1];
		int a = vector[8][0];
		int b = vector[8][1];
		int c = vector[9][0];
		int d = vector[9][1];

		double kappa_kappa = Bezier::kappa_ijklm(i, j, k, l, m, p,q, r,s,t, n) * Bezier::kappa_ijklm(u, v, w, x, y, z,a, b,c,d, n);
		if (std::abs(kappa_kappa) > 0){
			std::vector<double> index_vector = {
				double(i),double(j),
				double(k),double(l),
				double(m),double(p),
				double(q),double(r),
				double(s),double(t),
				double(u),double(v),
				double(w),double(x),
				double(y),double(z),
				double(a),double(b),
				double(c),double(d),
				kappa_kappa
			};
			this -> inertia_stats_1_indices_coefs_table.push_back(index_vector);
		}
	}


	// Inertia statistics 2

	index_vectors.clear();
	ShapeModelBezier::build_bezier_index_vectors(8,
		base_vector,
		index_vectors);

	for (auto vector : index_vectors){

		int i = vector[0][0];
		int j = vector[0][1];
		int k = vector[1][0];
		int l = vector[1][1];
		int m = vector[2][0];
		int p = vector[2][1];
		int q = vector[3][0];
		int r = vector[3][1];
		int s = vector[4][0];
		int t = vector[4][1];

		int u = vector[5][0];
		int v = vector[5][1];
		int w = vector[6][0];
		int x = vector[6][1];
		int y = vector[7][0];
		int z = vector[7][1];


		double alpha_kappa = Bezier::kappa_ijklm(i, j, k, l, m, p,q, r,s,t, n) * Bezier::alpha_ijk(u, v, w, x, y, z, n);
		
		if (std::abs(alpha_kappa) > 0){
			std::vector<double> index_vector = {
				double(i),double(j),
				double(k),double(l),
				double(m),double(p),
				double(q),double(r),
				double(s),double(t),
				double(u),double(v),
				double(w),double(x),
				double(y),double(z),
				alpha_kappa
			};
			this -> inertia_stats_2_indices_coefs_table.push_back(index_vector);
		}
	}


	std::cout << "- Inertia stats coefficients: " << (this -> inertia_stats_1_indices_coefs_table.size() + this -> inertia_stats_2_indices_coefs_table.size()) << std::endl;

}

void ShapeModelBezier::build_bezier_index_vectors(const int & n_indices,
	const std::vector<std::vector<int> > & base_vector,
	std::vector<std::vector<std::vector<int> > > & index_vectors,
	std::vector < std::vector<int> > temp_vector,
	const int depth){

	if (temp_vector.size() == 0){
		for ( int i = 0; i < n_indices; ++i ){
			temp_vector.push_back(std::vector<int>());
		}
	}

	for (unsigned int i = 0; i < base_vector.size(); ++i ){

		temp_vector[depth] = base_vector[i];

		if (depth == n_indices - 1 || n_indices == 1){
			index_vectors.push_back(temp_vector);
		}
		else{
			build_bezier_index_vectors(n_indices,base_vector,index_vectors,temp_vector,
				depth + 1);
		}

	}

}





void ShapeModelBezier::compute_point_covariances(double sigma_sq,double correl_distance) {

	double epsilon = 1e-2;

	this -> point_covariances_indices.clear();
	this -> point_covariances.clear();

	for (unsigned int i = 0; i < this -> get_NControlPoints(); ++i){

		arma::vec ni = this -> get_control_points() -> at(i) -> get_normal(true);

		arma::vec u_2 = arma::randn<arma::vec>(3);
		u_2 = arma::normalise(arma::cross(ni,u_2));

		arma::vec u_1 = arma::cross(u_2,ni);
		arma::mat::fixed<3,3> P = sigma_sq * (ni * ni.t() + epsilon * (u_1 * u_1.t() + u_2 * u_2.t()));

		std::vector<int> Pi_cor_indices = {i};
		std::vector<arma::mat::fixed<3,3>> Pi_cor = {P};


		for (unsigned int j = i + 1; j < this -> get_NControlPoints(); ++j){

			arma::vec nj = this -> get_control_points() -> at(j) -> get_normal(true);

			double distance = arma::norm(this -> get_control_points() -> at(i) -> get_coordinates()
				- this -> get_control_points() -> at(j) -> get_coordinates());
			
			if ( distance < 3 * correl_distance){


				double decay = std::exp(- std::pow(distance / correl_distance,2)) ;

				Pi_cor_indices.push_back(j);
				Pi_cor.push_back( sigma_sq * decay * ni * nj.t());

			}

		}


		this -> point_covariances_indices.push_back(Pi_cor_indices);
		this -> point_covariances.push_back(Pi_cor);

	}


	this -> find_correlated_elements();

}


arma::mat::fixed<3,3> ShapeModelBezier::get_point_covariance(int i, int j) const {


	std::vector<int> major_indices;
	std::vector<int>::iterator major_indices_it;

	if (i <= j){
		major_indices = this -> point_covariances_indices[i];
		major_indices_it = std::find(major_indices.begin(), major_indices.end(), j);
	}
	else{
		major_indices = this -> point_covariances_indices[j];
		major_indices_it = std::find(major_indices.begin(), major_indices.end(), i);
	}

	if (major_indices_it == major_indices.end()){
		
		return arma::zeros<arma::mat>(3,3);
	}

	int minor_index = std::distance(major_indices.begin(),major_indices_it);

	if (i == 2 && j == 0){

	}
	if (i <= j){
		return this -> point_covariances[i][minor_index];
	}
	else{
		return this -> point_covariances[j][minor_index].t();
	}

}


void ShapeModelBezier::compute_shape_covariance_sqrt(){

	
	this -> shape_covariance_sqrt = arma::zeros<arma::mat>(3 * this -> get_NControlPoints(),
		3 * this -> get_NControlPoints());

	arma::mat shape_covariance_arma(3 * this -> get_NControlPoints(),
		3 * this -> get_NControlPoints());


	for (int i = 0; i < this -> get_NControlPoints(); ++i){
		for (int j = 0; j < this -> get_NControlPoints(); ++j){
			shape_covariance_arma.submat(3 * i,3 * j,3 * i + 2, 3 * j + 2)  = this -> get_point_covariance(i,j);
		}
	}

	arma::vec eig_val;
	arma::mat eig_vec;

	arma::eig_sym(eig_val,eig_vec,shape_covariance_arma);


	// Regularizing the eigenvalue decomposition
	double min_val = arma::abs(eig_val).min();
	for (int i = 0; i < eig_val.n_rows; ++i){
		if (eig_val(i) < 0){
			eig_val(i) = min_val;
		}
	}

	this -> shape_covariance_sqrt = eig_vec * arma::diagmat(arma::sqrt(eig_val)) * eig_vec.t();

	std::ofstream file("cov_mat.txt");
	std::ofstream file_sqrt("sqrt.txt");

	file.precision(15);
	file_sqrt.precision(15);

	shape_covariance_arma.raw_print(file);
	this -> shape_covariance_sqrt.raw_print(file_sqrt);

}


void ShapeModelBezier::build_bezier_base_index_vector(const int n,std::vector<std::vector<int> > & base_vector){

	for (int i = 0; i < 1 + n; ++i){
		for (int j = 0; j < 1 + n - i; ++j){

			std::vector<int> pair = {i,j};
			base_vector.push_back(pair);
		}
	}

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
				unsigned int size = pointer_to_global_indices.size();
				pointer_to_global_indices[patch -> get_control_points() -> at(index)] = size;
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
				shape_file << iter -> at(index)  + 1<< " ";
			}
			else if (index == iter -> size() - 1 && iter != shape_patch_indices.end() - 1 ){
				shape_file << iter -> at(index)  + 1<< "\n";
			}
			else{
				shape_file << iter -> at(index) + 1;
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


unsigned int ShapeModelBezier::get_degree() const{
	return static_cast<Bezier * >(this -> elements. begin() -> get()) -> get_degree();
}


void ShapeModelBezier::save_to_obj(std::string path) const{

	// An inverse map going from vertex pointer to global indices is created

	// Note that the actual vertices on the shape model will not be be 
	// the control points, but the points lying on the bezier patch
 	// they support

	std::map<std::shared_ptr<ControlPoint> , unsigned int> pointer_to_global_indices;
	std::vector<arma::vec> vertices;
	std::vector<std::tuple<std::shared_ptr<ControlPoint>,std::shared_ptr<ControlPoint>,std::shared_ptr<ControlPoint> > > facets;


	// The global indices of the control points are found. 
	for (unsigned int i = 0; i < this -> get_NElements(); ++i){

		Bezier * patch = dynamic_cast<Bezier * >(this -> get_element(i).get());

		for (unsigned int index = 0; index < patch -> get_control_points() -> size(); ++index){

			if (pointer_to_global_indices.find(patch -> get_control_points() -> at(index))== pointer_to_global_indices.end()){

				unsigned int size =  pointer_to_global_indices.size();

				pointer_to_global_indices[patch -> get_control_points() -> at(index)] = size;

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





void ShapeModelBezier::save_to_obj_omp(std::string path, const arma::vec & deviation) const{

	// An inverse map going from vertex pointer to global indices is created

	// Note that the actual vertices on the shape model will not be be 
	// the control points, but the points lying on the bezier patch
 	// they support

	std::map<std::shared_ptr<ControlPoint> , unsigned int> pointer_to_global_indices;
	std::vector<arma::vec> vertices;
	std::vector<std::tuple<std::shared_ptr<ControlPoint>,std::shared_ptr<ControlPoint>,std::shared_ptr<ControlPoint> > > facets;


	// The global indices of the control points are found. 
	for (unsigned int i = 0; i < this -> get_NElements(); ++i){

		Bezier * patch = static_cast<Bezier * >(this -> get_element(i).get());

		for (unsigned int index = 0; index < patch -> get_control_points() -> size(); ++index){

			if (pointer_to_global_indices.find(patch -> get_control_points() -> at(index))== pointer_to_global_indices.end()){

				unsigned int size =  pointer_to_global_indices.size();

				pointer_to_global_indices[patch -> get_control_points() -> at(index)] = size;

				auto local_indices = patch -> get_local_indices(patch -> get_control_points() -> at(index));
				double u =  double(std::get<0>(local_indices)) / patch -> get_degree();
				double v =  double(std::get<1>(local_indices)) / patch -> get_degree();

				arma::vec surface_point = patch -> evaluate_omp(u,v,deviation);
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



arma::vec::fixed<9> ShapeModelBezier::get_E_vectors(const arma::mat::fixed<3,3> & inertia){

	arma::vec::fixed<9> E_vectors;
	arma::vec moments = arma::eig_sym(inertia);
	for (int i = 0; i < 3; ++i){

		arma::mat L = inertia - moments(i) * arma::eye<arma::mat>(3,3);


		arma::vec norm_vec = {
			arma::norm(arma::cross(L.row(0).t(),L.row(1).t())),
			arma::norm(arma::cross(L.row(0).t(),L.row(2).t())),
			arma::norm(arma::cross(L.row(1).t(),L.row(2).t()))
		};

		int j = norm_vec.index_max();

		if (j == 0){
			E_vectors.rows(3 * i, 3 * i + 2) = arma::cross(L.row(0).t(),L.row(1).t());
		}
		else if (j == 1){
			E_vectors.rows(3 * i, 3 * i + 2) = arma::cross(L.row(0).t(),L.row(2).t());
		}
		else if (j == 2){
			E_vectors.rows(3 * i, 3 * i + 2) = arma::cross(L.row(1).t(),L.row(2).t());
		}
	}

	return E_vectors;

}


arma::mat::fixed<3,3> ShapeModelBezier::get_principal_axes_stable(const arma::mat::fixed<3,3> & inertia){


	auto E_vectors = ShapeModelBezier::get_E_vectors(inertia);

	arma::mat::fixed<3,3> pa;
	pa.col(0) = E_vectors.rows(0,2) / arma::norm( E_vectors.rows(0,2));
	pa.col(1) = E_vectors.rows(3,5) / arma::norm( E_vectors.rows(3,5));
	pa.col(2) = E_vectors.rows(6,8) / arma::norm( E_vectors.rows(6,8));

	return pa;

}



void ShapeModelBezier::compute_inertia_statistics() {
	std::cout << "\n- Computing inertia statistics ...\n";

	this -> compute_P_I();
	this -> compute_P_MI();
	this -> compute_P_MX();
	this -> compute_P_Y();
	this -> compute_P_moments();
	this -> compute_P_dims();


	this -> compute_P_Evectors();
	this -> compute_P_eigenvectors();
	this -> compute_P_sigma();


}


void ShapeModelBezier::compute_P_I(){


	arma::mat::fixed<6,6> P_I = arma::zeros<arma::mat>(6,6);

	std::vector<std::set < Element * > > connected_elements;

	for (unsigned int e = 0; e < this -> elements.size(); ++e) {
		auto elements = this -> elements[e] -> get_neighbors(true);
		connected_elements.push_back(elements);
	}

	boost::progress_display progress(this -> inertia_stats_1_indices_coefs_table.size()) ;


	#pragma omp parallel for reduction (+:P_I)

	for (int index = 0 ; index <  this -> inertia_stats_1_indices_coefs_table.size(); ++index) {

		auto coefs_row = this -> inertia_stats_1_indices_coefs_table[index];

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

				// s
		int a =  int(coefs_row[16]);
		int b =  int(coefs_row[17]);

				// t
		int c =  int(coefs_row[18]);
		int d =  int(coefs_row[19]);


		arma::mat::fixed<6,15> left_mat;
		arma::mat::fixed<6,15> right_mat;

		for (unsigned int e = 0; e < this -> elements.size(); ++e) {

			Bezier * patch_e = static_cast<Bezier * >(this -> elements[e].get());


			int i_g,j_g,k_g,l_g,m_g;

			i_g = patch_e -> get_control_point_global_index(i,j);
			j_g = patch_e -> get_control_point_global_index(k,l);
			k_g = patch_e -> get_control_point_global_index(m,p);
			l_g = patch_e -> get_control_point_global_index(q,r);
			m_g = patch_e -> get_control_point_global_index(s,t);

			auto neighbors = connected_elements[e];

			this -> construct_inertia_mapping_mat(left_mat,i_g,j_g,k_g,l_g,m_g);

			for (unsigned int f = 0; f < this -> elements.size(); ++f) {

				Bezier * patch_f = static_cast<Bezier * >(this -> elements[f].get());

				int p_g,q_g,r_g,s_g,t_g;

				p_g = patch_f -> get_control_point_global_index(u,v);
				q_g = patch_f -> get_control_point_global_index(w,x);
				r_g = patch_f -> get_control_point_global_index(y,z);
				s_g = patch_f -> get_control_point_global_index(a,b);
				t_g = patch_f -> get_control_point_global_index(c,d);	

				this -> construct_inertia_mapping_mat(right_mat,p_g,q_g,r_g,s_g,t_g);


				P_I += coefs_row[20] * this -> increment_P_I(left_mat,
					right_mat, 
					i_g,j_g,k_g,l_g,m_g, 
					p_g,q_g,r_g,s_g,t_g);


			}
		}

		++progress;
	}

	this -> P_I = P_I;

}








arma::vec ShapeModelBezier::d_I() const{


	std::vector<std::set < Element * > > connected_elements;

	for (unsigned int e = 0; e < this -> elements.size(); ++e) {

		auto elements = this -> elements[e] -> get_neighbors(true);
		connected_elements.push_back(elements);
	}

	arma::vec::fixed<6> dI = arma::zeros<arma::vec>(6);
	boost::progress_display progress(this -> inertia_indices_coefs_table.size()) ;


	for (int index = 0 ; index <  this -> inertia_indices_coefs_table.size(); ++index) {

		auto coefs_row = this -> inertia_indices_coefs_table[index];

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

		arma::mat::fixed<6,15> left_mat;

		for (unsigned int e = 0; e < this -> elements.size(); ++e) {

			Bezier * patch_e = static_cast<Bezier * >(this -> elements[e].get());
			int i_g,j_g,k_g,l_g,m_g;

			arma::vec dev(15);

			i_g = patch_e -> get_control_point_global_index(i,j);
			j_g = patch_e -> get_control_point_global_index(k,l);
			k_g = patch_e -> get_control_point_global_index(m,p);
			l_g = patch_e -> get_control_point_global_index(q,r);
			m_g = patch_e -> get_control_point_global_index(s,t);

			dev.rows(0,2) = this -> control_points[i_g] -> get_deviation();
			dev.rows(3,5) = this -> control_points[j_g] -> get_deviation();
			dev.rows(6,8) = this -> control_points[k_g] -> get_deviation();
			dev.rows(9,11) = this -> control_points[l_g] -> get_deviation();
			dev.rows(12,14) = this -> control_points[m_g] -> get_deviation();

			this -> construct_inertia_mapping_mat(left_mat,i_g,j_g,k_g,l_g,m_g);


			dI += coefs_row[10] * left_mat * dev;
		}

		++progress;
	}

	return dI;

}


void ShapeModelBezier::compute_P_MI(){

	arma::vec::fixed<6> P_MI = arma::zeros<arma::vec>(6);

	std::vector<std::set < Element * > > connected_elements;

	for (unsigned int e = 0; e < this -> elements.size(); ++e) {

		auto elements = this -> elements[e] -> get_neighbors(true);
		connected_elements.push_back(elements);
	}

	boost::progress_display progress(this -> inertia_stats_2_indices_coefs_table.size()) ;


	#pragma omp parallel for reduction (+:P_MI)

	for (int index = 0 ; index <  this -> inertia_stats_2_indices_coefs_table.size(); ++index) {

		auto coefs_row = this -> inertia_stats_2_indices_coefs_table[index];

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

		arma::mat::fixed<6,15> left_mat;
		arma::vec::fixed<9> right_vec;

		for (unsigned int e = 0; e < this -> elements.size(); ++e) {

			Bezier * patch_e = static_cast<Bezier * >(this -> elements[e].get());
			int i_g,j_g,k_g,l_g,m_g;


			i_g = patch_e -> get_control_point_global_index(i,j);
			j_g = patch_e -> get_control_point_global_index(k,l);
			k_g = patch_e -> get_control_point_global_index(m,p);
			l_g = patch_e -> get_control_point_global_index(q,r);
			m_g = patch_e -> get_control_point_global_index(s,t);	


			auto neighbors = connected_elements[e];

			this -> construct_inertia_mapping_mat(left_mat,i_g,j_g,k_g,l_g,m_g);

			
			for (unsigned int f = 0; f < this -> elements.size(); ++f) {

				Bezier * patch_f = static_cast<Bezier * >(this -> elements[f].get());

				int p_g,q_g,r_g;

				p_g = patch_f -> get_control_point_global_index(u,v);
				q_g = patch_f -> get_control_point_global_index(w,x);
				r_g = patch_f -> get_control_point_global_index(y,z);

				right_vec = patch_f -> get_cross_products(u,v,w,x,y,z);

				P_MI += coefs_row[16] *  this -> increment_P_MI(left_mat,
					right_vec, 
					i_g,j_g,k_g,l_g,m_g, 
					p_g,q_g,r_g);
			}
		}

		++progress;
	}

	this -> P_MI = P_MI;

}


arma::mat::fixed<6,6> ShapeModelBezier::increment_P_I(const arma::mat::fixed<6,15> & left_mat,
	const arma::mat::fixed<6,15>  & right_mat, 
	int i,int j,int k,int l,int m,
	int p, int q, int r, int s, int t) const{

	arma::mat::fixed<15,15> P_CC = arma::zeros<arma::mat>(15,15);


	P_CC.submat(0,0,2,2) = this -> get_point_covariance(i, p);
	P_CC.submat(0,3,2,5) = this -> get_point_covariance(i, q);
	P_CC.submat(0,6,2,8) = this -> get_point_covariance(i, r);
	P_CC.submat(0,9,2,11) = this -> get_point_covariance(i, s);
	P_CC.submat(0,12,2,14) = this -> get_point_covariance(i, t);

	P_CC.submat(3,0,5,2) = this -> get_point_covariance(j, p);
	P_CC.submat(3,3,5,5) = this -> get_point_covariance(j, q);
	P_CC.submat(3,6,5,8) = this -> get_point_covariance(j, r);
	P_CC.submat(3,9,5,11) = this -> get_point_covariance(j, s);
	P_CC.submat(3,12,5,14) = this -> get_point_covariance(j, t);

	P_CC.submat(6,0,8,2) = this -> get_point_covariance(k, p);
	P_CC.submat(6,3,8,5) = this -> get_point_covariance(k, q);
	P_CC.submat(6,6,8,8) = this -> get_point_covariance(k, r);
	P_CC.submat(6,9,8,11) = this -> get_point_covariance(k, s);
	P_CC.submat(6,12,8,14) = this -> get_point_covariance(k, t);


	P_CC.submat(9,0,11,2) = this -> get_point_covariance(l, p);
	P_CC.submat(9,3,11,5) = this -> get_point_covariance(l, q);
	P_CC.submat(9,6,11,8) = this -> get_point_covariance(l, r);
	P_CC.submat(9,9,11,11) = this -> get_point_covariance(l, s);
	P_CC.submat(9,12,11,14) = this -> get_point_covariance(l, t);


	P_CC.submat(12,0,14,2) = this -> get_point_covariance(m, p);
	P_CC.submat(12,3,14,5) = this -> get_point_covariance(m, q);
	P_CC.submat(12,6,14,8) = this -> get_point_covariance(m, r);
	P_CC.submat(12,9,14,11) = this -> get_point_covariance(m, s);
	P_CC.submat(12,12,14,14) = this -> get_point_covariance(m,t);


	return left_mat * P_CC * right_mat.t();



}


arma::vec::fixed<6> ShapeModelBezier::increment_P_MI(const arma::mat::fixed<6,15> & left_mat,
	const arma::vec::fixed<9>  & right_vec, 
	int i,int j,int k,int l,int m,
	int p, int q, int r){

	arma::mat::fixed<15,9> P_CC = arma::zeros<arma::mat>(15,9);

	P_CC.submat(0,0,2,2) = this -> get_point_covariance(i, p);
	P_CC.submat(0,3,2,5) = this -> get_point_covariance(i, q);
	P_CC.submat(0,6,2,8) = this -> get_point_covariance(i, r);
	
	P_CC.submat(3,0,5,2) = this -> get_point_covariance(j, p);
	P_CC.submat(3,3,5,5) = this -> get_point_covariance(j, q);
	P_CC.submat(3,6,5,8) = this -> get_point_covariance(j, r);
	
	P_CC.submat(6,0,8,2) = this -> get_point_covariance(k, p);
	P_CC.submat(6,3,8,5) = this -> get_point_covariance(k, q);
	P_CC.submat(6,6,8,8) = this -> get_point_covariance(k, r);
	

	P_CC.submat(9,0,11,2) = this -> get_point_covariance(l, p);
	P_CC.submat(9,3,11,5) = this -> get_point_covariance(l, q);
	P_CC.submat(9,6,11,8) = this -> get_point_covariance(l, r);
	
	P_CC.submat(12,0,14,2) = this -> get_point_covariance(m, p);
	P_CC.submat(12,3,14,5) = this -> get_point_covariance(m, q);
	P_CC.submat(12,6,14,8) = this -> get_point_covariance(m, r);

	return left_mat * P_CC * right_vec;

}


arma::mat::fixed<3,3> ShapeModelBezier::increment_cm_cov(const arma::mat::fixed<12,3> & left_mat,
	const arma::mat::fixed<12,3>  & right_mat, 
	int i,int j,int k,int l, 
	int m, int p, int q, int r){

	arma::mat::fixed<12,12> P_CC = arma::zeros<arma::mat>(12,12);


	P_CC.submat(0,0,2,2) = this -> get_point_covariance(i, m);
	P_CC.submat(0,3,2,5) = this -> get_point_covariance(i, p);
	P_CC.submat(0,6,2,8) = this -> get_point_covariance(i, q);
	P_CC.submat(0,9,2,11) = this -> get_point_covariance(i, r);

	P_CC.submat(3,0,5,2) = this -> get_point_covariance(j, m);
	P_CC.submat(3,3,5,5) = this -> get_point_covariance(j, p);
	P_CC.submat(3,6,5,8) = this -> get_point_covariance(j, q);
	P_CC.submat(3,9,5,11) = this -> get_point_covariance(j, r);


	P_CC.submat(6,0,8,2) = this -> get_point_covariance(k, m);
	P_CC.submat(6,3,8,5) = this -> get_point_covariance(k, p);
	P_CC.submat(6,6,8,8) = this -> get_point_covariance(k, q);
	P_CC.submat(6,9,8,11) = this -> get_point_covariance(k, r);


	P_CC.submat(9,0,11,2) = this -> get_point_covariance(l, m);
	P_CC.submat(9,3,11,5) = this -> get_point_covariance(l, p);
	P_CC.submat(9,6,11,8) = this -> get_point_covariance(l, q);
	P_CC.submat(9,9,11,11) = this -> get_point_covariance(l, r);

	return left_mat.t() * P_CC * right_mat;
}


void ShapeModelBezier::compute_P_Y(){
	arma::mat::fixed<4,4> mat = arma::zeros<arma::mat>(4,4);

	mat.submat(0,0,2,2) = ShapeModelBezier::P_XX();
	mat.submat(0,3,2,3) = this -> P_MX;
	mat.submat(3,0,3,2) = this -> P_MX.t();
	mat(3,3) = std::pow(this -> volume_sd,2);


	this -> P_Y = mat;
}


arma::mat::fixed<3,3> ShapeModelBezier::P_XX() const { 


	return ShapeModelBezier::partial_X_partial_I() * this -> P_I * ShapeModelBezier::partial_X_partial_I().t();

}


void ShapeModelBezier::compute_P_MX() {


	this -> P_MX = ShapeModelBezier::partial_X_partial_I() * this -> P_MI;

}

arma::mat::fixed<3,6> ShapeModelBezier::partial_X_partial_I() const{

	arma::mat::fixed<3,6> mat = arma::zeros<arma::mat>(3,6);

	mat.row(0) = ShapeModelBezier::partial_T_partial_I();
	mat.row(1) = ShapeModelBezier::partial_U_partial_I();
	mat.row(2) = ShapeModelBezier::partial_theta_partial_I();
	return mat;
}


arma::rowvec::fixed<6> ShapeModelBezier::partial_T_partial_I() {
	arma::rowvec dTdI = {1,1,1,0,0,0};
	return dTdI;
}



arma::rowvec::fixed<6> ShapeModelBezier::partial_theta_partial_I() const {

	arma::mat I_C = this -> inertia - this -> get_volume() * RBK::tilde(this -> get_center_of_mass()) * RBK::tilde(this -> get_center_of_mass()).t() ;

	double T = arma::trace(I_C);
	double d = arma::det (I_C);

	double I_xx = I_C(0,0);
	double I_yy = I_C(1,1);
	double I_zz = I_C(2,2);
	double I_xy = I_C(0,1);
	double I_xz = I_C(0,2);
	double I_yz = I_C(1,2);


	arma::vec I = {I_xx,I_yy,I_zz,I_xy,I_xz,I_yz};

	arma::mat::fixed<6,6> Q = {
		{0,1./2.,1./2,0,0,0},
		{1./2.,0,1./2,0,0,0},
		{1./2,1./2.,0,0,0,0},
		{0,0,0,-1,0,0},
		{0,0,0,0,-1,0},
		{0,0,0,0,0,-1}
	};



	double Pi = arma::dot(I, Q * I);

	double U = std::sqrt(T * T - 3 * Pi)/3;

	double Theta = (-2 * std::pow(T,3) + 9 * T * Pi - 27 * d)/(54 * std::pow(U,3));


	arma::rowvec dthetadI = (ShapeModelBezier::partial_theta_partial_Theta(Theta) 
		* ShapeModelBezier::partial_Theta_partial_W(T,Pi,U,d) 
		* ShapeModelBezier::partial_W_partial_I());

	return dthetadI;
}



arma::vec::fixed<4> ShapeModelBezier::get_Y(const double & volume, 
	const arma::mat::fixed<3,3> & I_C
	) {


	double T = arma::trace(I_C);
	double d = arma::det (I_C);

	double I_xx = I_C(0,0);
	double I_yy = I_C(1,1);
	double I_zz = I_C(2,2);
	double I_xy = I_C(0,1);
	double I_xz = I_C(0,2);
	double I_yz = I_C(1,2);

	arma::vec I = {I_xx,I_yy,I_zz,I_xy,I_xz,I_yz};

	arma::mat::fixed<6,6> Q = {
		{0,1./2.,1./2,0,0,0},
		{1./2.,0,1./2,0,0,0},
		{1./2,1./2.,0,0,0,0},
		{0,0,0,-1,0,0},
		{0,0,0,0,-1,0},
		{0,0,0,0,0,-1}
	};



	double Pi = arma::dot(I, Q * I);

	double U = std::sqrt(T * T - 3 * Pi)/3;

	double Theta = (-2 * std::pow(T,3) + 9 * T * Pi - 27 * d)/(54 * std::pow(U,3));
	double theta = std::acos(Theta);


	arma::vec::fixed<4> vec;
	vec(0) = T;
	vec(1) = U;
	vec(2) = theta;
	vec(3) = volume;

	return vec;
}


arma::rowvec::fixed<4> ShapeModelBezier::partial_Theta_partial_W(const double & T,const double & Pi,const double & U,const double & d){

	arma::rowvec::fixed<4> dThetadW =  {
		(-6 * T * T + 9 * Pi) * U,
		9 * T * U,
		-27 * U,
		-3 * (-2 * std::pow(T,3) + 9 * T * Pi - 27 * d)
	};
	return 1./(54 * std::pow(U,4)) * dThetadW;


}


double ShapeModelBezier::partial_theta_partial_Theta(const double & Theta){
	return - 1. / std::sqrt(1 - Theta * Theta);
}


arma::mat::fixed<4,6> ShapeModelBezier::partial_W_partial_I() const{

	arma::mat::fixed<4,6> dWdI;
	dWdI.row(0) = ShapeModelBezier::partial_T_partial_I();
	dWdI.row(1) = ShapeModelBezier::partial_Pi_partial_I();
	dWdI.row(2) = ShapeModelBezier::partial_d_partial_I();
	dWdI.row(3) = ShapeModelBezier::partial_U_partial_I();

	return dWdI;

}

arma::rowvec::fixed<6> ShapeModelBezier::partial_d_partial_I() const {

	arma::mat I_C = this -> inertia - this -> get_volume() * RBK::tilde(this -> get_center_of_mass()) * RBK::tilde(this -> get_center_of_mass()).t() ;

	double T = arma::trace(I_C);
	double d = arma::det (I_C);

	double I_xx = I_C(0,0);
	double I_yy = I_C(1,1);
	double I_zz = I_C(2,2);
	double I_xy = I_C(0,1);
	double I_xz = I_C(0,2);
	double I_yz = I_C(1,2);

	arma::rowvec::fixed<6> dddI = {
		I_yy * I_zz - std::pow(I_yz,2),
		I_xx * I_zz - std::pow(I_xz,2),
		I_xx * I_yy - std::pow(I_xy,2),
		2 * (I_xz * I_yz - I_zz * I_xy),
		2 * (I_xy * I_yz - I_yy * I_xz),
		2 * (I_xy * I_xz - I_xx * I_yz)
	};

	return dddI;

}


arma::rowvec::fixed<6> ShapeModelBezier::partial_U_partial_I() const{

	arma::rowvec::fixed<6> dUdI = ShapeModelBezier::partial_U_partial_Z() * ShapeModelBezier::partial_Z_partial_I() ;


	return dUdI;

}


arma::rowvec::fixed<2> ShapeModelBezier::partial_U_partial_Z() const{

	arma::mat I_C = this -> inertia - this -> get_volume() * RBK::tilde(this -> get_center_of_mass()) * RBK::tilde(this -> get_center_of_mass()).t() ;

	double T = arma::trace(I_C);
	double d = arma::det (I_C);

	double I_xx = I_C(0,0);
	double I_yy = I_C(1,1);
	double I_zz = I_C(2,2);
	double I_xy = I_C(0,1);
	double I_xz = I_C(0,2);
	double I_yz = I_C(1,2);

	arma::vec I = {I_xx,I_yy,I_zz,I_xy,I_xz,I_yz};

	arma::mat::fixed<6,6> Q = {
		{0,1./2.,1./2,0,0,0},
		{1./2.,0,1./2,0,0,0},
		{1./2,1./2.,0,0,0,0},
		{0,0,0,-1,0,0},
		{0,0,0,0,-1,0},
		{0,0,0,0,0,-1}
	};

	double Pi = arma::dot(I, Q * I);
	double U = std::sqrt(T * T - 3 * Pi)/3;


	arma::rowvec::fixed<2> dUdZ = {2 * T,-3}	;
	return 1./(18 * U) * dUdZ;

}

arma::mat::fixed<2,6> ShapeModelBezier::partial_Z_partial_I() const {

	arma::mat::fixed<2,6> dZdI;
	dZdI.row(0) = ShapeModelBezier::partial_T_partial_I();
	dZdI.row(1) = ShapeModelBezier::partial_Pi_partial_I();

	return dZdI;

}

arma::rowvec::fixed<6> ShapeModelBezier::partial_Pi_partial_I() const {

	arma::mat I_C = this -> inertia - this -> get_volume() * RBK::tilde(this -> get_center_of_mass()) * RBK::tilde(this -> get_center_of_mass()).t() ;

	double T = arma::trace(I_C);
	double d = arma::det (I_C);

	double I_xx = I_C(0,0);
	double I_yy = I_C(1,1);
	double I_zz = I_C(2,2);
	double I_xy = I_C(0,1);
	double I_xz = I_C(0,2);
	double I_yz = I_C(1,2);

	arma::vec I = {I_xx,I_yy,I_zz,I_xy,I_xz,I_yz};

	arma::mat::fixed<6,6> Q = {
		{0,1./2.,1./2,0,0,0},
		{1./2.,0,1./2,0,0,0},
		{1./2,1./2.,0,0,0,0},
		{0,0,0,-1,0,0},
		{0,0,0,0,-1,0},
		{0,0,0,0,0,-1}
	};

	return 2 * I.t() * Q;

}


arma::rowvec::fixed<3> ShapeModelBezier::partial_A_partial_Y(const double & theta,const double & U){

	arma::rowvec::fixed<3> dAdY = {1./3,- 2 * std::cos(theta / 3),2./3 * U * std::sin(theta/3)};

	return dAdY;

}


arma::rowvec::fixed<3> ShapeModelBezier::partial_B_partial_Y(const double & theta,const double & U){

	arma::rowvec::fixed<3> dBdY = {1./3,- 2 * std::cos(theta / 3 - 2 * arma::datum::pi /3),2./3 * U * std::sin(theta/3 - 2 * arma::datum::pi /3)};

	return dBdY;

}

arma::rowvec::fixed<3> ShapeModelBezier::partial_C_partial_Y(const double & theta,const double & U){

	arma::rowvec::fixed<3> dCdY = {1./3,- 2 * std::cos(theta / 3 + 2 * arma::datum::pi /3),2./3 * U * std::sin(theta/3 + 2 * arma::datum::pi /3)};

	return dCdY;

}


void ShapeModelBezier::compute_P_moments(){

	auto dMdY = ShapeModelBezier::partial_M_partial_Y();

	this -> P_moments = dMdY * this -> P_Y * dMdY.t();

}

arma::mat::fixed<4,4>  ShapeModelBezier::partial_M_partial_Y() const{


	arma::mat I_C = this -> inertia - this -> get_volume() * RBK::tilde(this -> get_center_of_mass()) * RBK::tilde(this -> get_center_of_mass()).t() ;

	double T = arma::trace(I_C);
	double d = arma::det (I_C);

	double I_xx = I_C(0,0);
	double I_yy = I_C(1,1);
	double I_zz = I_C(2,2);
	double I_xy = I_C(0,1);
	double I_xz = I_C(0,2);
	double I_yz = I_C(1,2);


	arma::vec I = {I_xx,I_yy,I_zz,I_xy,I_xz,I_yz};

	arma::mat::fixed<6,6> Q = {
		{0,1./2.,1./2,0,0,0},
		{1./2.,0,1./2,0,0,0},
		{1./2,1./2.,0,0,0,0},
		{0,0,0,-1,0,0},
		{0,0,0,0,-1,0},
		{0,0,0,0,0,-1}
	};


	double Pi = arma::dot(I, Q * I);
	double U = std::sqrt(T * T - 3 * Pi)/3;
	double Theta = (-2 * std::pow(T,3) + 9 * T * Pi - 27 * d)/(54 * std::pow(U,3));
	double theta = std::acos(Theta);


	arma::mat::fixed<4,4> mat = arma::zeros<arma::mat>(4,4);

	mat.submat(0,0,0,2) = ShapeModelBezier::partial_A_partial_Y(theta,U);
	mat.submat(1,0,1,2) = ShapeModelBezier::partial_B_partial_Y(theta,U);
	mat.submat(2,0,2,2) = ShapeModelBezier::partial_C_partial_Y(theta,U);
	mat(3,3) = 1;
	return mat;


}

void ShapeModelBezier::take_and_save_zslice(std::string path, const double & c) const {

	std::vector<std::vector<arma::vec> > lines;
	this -> take_zslice(lines,c);
	this -> save_zslice(path,lines);

}


void ShapeModelBezier::take_and_save_zslice_omp(std::string path, const double & c,const arma::vec & deviation) const {

	std::vector<std::vector<arma::vec> > lines;
	this -> take_zslice_omp(lines,c,deviation);
	this -> save_zslice(path,lines);

}





void ShapeModelBezier::save_zslice(std::string path, const std::vector<std::vector<arma::vec> > & lines) const{

	arma::mat lines_arma;

	if (lines.size() > 0){
		lines_arma = arma::mat(lines.size(),4);

		for (int i = 0; i < lines.size() ; ++i){

			arma::rowvec rowvec = {lines[i][0](0),lines[i][0](1),lines[i][1](0),lines[i][1](1)};
			lines_arma.row(i) = rowvec;
		}

		lines_arma.save(path, arma::raw_ascii);


	}



}






void ShapeModelBezier::take_zslice(std::vector<std::vector<arma::vec> > & lines,
	const double & c) const{


	arma::vec n_plane = {0,0,1};
	if (this -> get_degree() != 1){
		throw(std::runtime_error("Only works with bezier shapes of degree one"));
	}

	arma::mat::fixed<3,2> T = {
		{1,0},
		{0,1},
		{-1,-1}
	};

	arma::vec::fixed<3> e3 = {0,0,1};
	arma::mat::fixed<3,3> C;

	// Each surface element is "sliced"
	for (auto el = this -> elements.begin(); el != this -> elements.end(); ++el){

		C.col(0) = static_cast<Bezier * >((*el).get()) -> get_control_point(1,0) -> get_coordinates();
		C.col(1) = static_cast<Bezier * >((*el).get()) -> get_control_point(0,1) -> get_coordinates();
		C.col(2) = static_cast<Bezier * >((*el).get()) -> get_control_point(0,0) -> get_coordinates();

		arma::rowvec M = n_plane.t() * C * T;
		double e = c - arma::dot(n_plane,C * e3);
		arma::vec intersect;

		std::vector <arma::vec> intersects;

		// Looking for an intersect along the u = 0 edge
		if (std::abs(M(1)) > 1e-6){
			double v_intersect = e / M(1);
			if (v_intersect >= 0 && v_intersect <= 1 ){
				arma::vec Y = {0,v_intersect};

				intersect = C * (T * Y + e3);
				intersects.push_back(intersect.rows(0,1));
			}
		}

		// Looking for an intersect along the v = 0 edge
		if (std::abs(M(0)) > 1e-6){
			double u_intersect = e / M(0);
			if (u_intersect >= 0 && u_intersect <= 1 ){
				arma::vec Y = {u_intersect,0};

				intersect = C * (T * Y + e3);
				intersects.push_back(intersect.rows(0,1));
			}
		}

		// Looking for an intersect along the w = 0 edge
		// using u as the parameter

		if (std::abs(M(0) - M(1)) > 1e-6){
			double u_intersect = (e - M(1)) / (M(0) - M(1));
			if (u_intersect >= 0 && u_intersect <= 1 ){
				arma::vec Y = {u_intersect,1 - u_intersect};

				intersect = C * (T * Y + e3);
				intersects.push_back(intersect.rows(0,1));
			}
		}
		if (intersects.size() == 2){
			lines.push_back(intersects);
		}

	}

}



void ShapeModelBezier::take_zslice_omp(std::vector<std::vector<arma::vec> > & lines,
	const double & c,const arma::vec & deviation) const{


	arma::vec n_plane = {0,0,1};
	if (this -> get_degree() != 1){
		throw(std::runtime_error("Only works with bezier shapes of degree one"));
	}

	arma::mat::fixed<3,2> T = {
		{1,0},
		{0,1},
		{-1,-1}
	};

	arma::vec::fixed<3> e3 = {0,0,1};
	arma::mat::fixed<3,3> C;

	// Each surface element is "sliced"
	for (auto el = this -> elements.begin(); el != this -> elements.end(); ++el){

		auto Ci = static_cast<Bezier * >((*el).get()) -> get_control_point(1,0);
		auto Cj = static_cast<Bezier * >((*el).get()) -> get_control_point(0,1);
		auto Ck = static_cast<Bezier * >((*el).get()) -> get_control_point(0,0);

		int i_g = Ci -> get_global_index();
		int j_g = Cj -> get_global_index();
		int k_g = Ck -> get_global_index();


		C.col(0) = Ci -> get_coordinates() + deviation.rows(3 * i_g, 3 * i_g + 2);
		C.col(1) = Cj -> get_coordinates() + deviation.rows(3 * j_g, 3 * j_g + 2);
		C.col(2) = Ck -> get_coordinates() + deviation.rows(3 * k_g, 3 * k_g + 2);




		arma::rowvec M = n_plane.t() * C * T;
		double e = c - arma::dot(n_plane,C * e3);
		arma::vec intersect;

		std::vector <arma::vec> intersects;

		// Looking for an intersect along the u = 0 edge
		if (std::abs(M(1)) > 1e-6){
			double v_intersect = e / M(1);
			if (v_intersect >= 0 && v_intersect <= 1 ){
				arma::vec Y = {0,v_intersect};

				intersect = C * (T * Y + e3);
				intersects.push_back(intersect.rows(0,1));
			}
		}

		// Looking for an intersect along the v = 0 edge
		if (std::abs(M(0)) > 1e-6){
			double u_intersect = e / M(0);
			if (u_intersect >= 0 && u_intersect <= 1 ){
				arma::vec Y = {u_intersect,0};

				intersect = C * (T * Y + e3);
				intersects.push_back(intersect.rows(0,1));
			}
		}

		// Looking for an intersect along the w = 0 edge
		// using u as the parameter

		if (std::abs(M(0) - M(1)) > 1e-6){
			double u_intersect = (e - M(1)) / (M(0) - M(1));
			if (u_intersect >= 0 && u_intersect <= 1 ){
				arma::vec Y = {u_intersect,1 - u_intersect};

				intersect = C * (T * Y + e3);
				intersects.push_back(intersect.rows(0,1));
			}
		}
		if (intersects.size() == 2){
			lines.push_back(intersects);
		}

	}

}








void ShapeModelBezier::compute_P_sigma(){

	arma::mat I_C = this -> inertia - this -> get_volume() * RBK::tilde(this -> get_center_of_mass()) * RBK::tilde(this -> get_center_of_mass()).t() ;

	arma::mat eigvec = ShapeModelBezier::get_principal_axes_stable(I_C);

	arma::mat mapping_mat = arma::zeros<arma::mat>(3,9);
	mapping_mat.submat(0,3,0,5) = eigvec.col(2).t();
	mapping_mat.submat(1,6,1,8) = eigvec.col(0).t();
	mapping_mat.submat(2,0,2,2) = eigvec.col(1).t();

	this -> P_sigma = 1./16 * mapping_mat * this -> P_eigenvectors * mapping_mat.t();

}


arma::mat::fixed<3,4> ShapeModelBezier::partial_dim_partial_M() const{


	arma::vec moments = arma::eig_sym(this -> inertia);
	double A = moments(0);
	double B = moments(1);
	double C = moments(2);


	arma::mat::fixed<3,4> mat=  {
		{-1,1,1, -( B + C - A) / this -> volume}, 
		{1,-1,1, -( A + C - B) / this -> volume}, 
		{1,1,-1, -( A + B - C) / this -> volume}
	} ;

	return 5./(4 * this -> volume) * arma::diagmat(1./this -> get_dims(this -> volume,
		this -> inertia)) * mat;


}


void ShapeModelBezier::compute_P_dims() {


	arma::mat::fixed<3,4> partial = this -> partial_dim_partial_M();

	this -> P_dims = partial * this -> P_moments * partial.t();

}




void ShapeModelBezier::compute_P_eigenvectors() {

	arma::mat::fixed<9,9> partial_mat = arma::zeros<arma::mat>(9,9);

	arma::vec moments = arma::eig_sym(this -> inertia);


	partial_mat.submat(0,0,2,2) = ShapeModelBezier::partial_elambda_Elambda(moments(0));
	partial_mat.submat(3,3,5,5) = ShapeModelBezier::partial_elambda_Elambda(moments(1));
	partial_mat.submat(6,6,8,8) = ShapeModelBezier::partial_elambda_Elambda(moments(2));



	this -> P_eigenvectors = partial_mat * this -> P_Evectors * partial_mat.t();

}

void ShapeModelBezier::compute_P_Evectors() {

	arma::mat::fixed<9,9> P_Evectors;


	arma::mat I_C = this -> inertia - this -> get_volume() * RBK::tilde(this -> get_center_of_mass()) * RBK::tilde(this -> get_center_of_mass()).t() ;


	double T = arma::trace(I_C);
	double d = arma::det (I_C);

	double I_xx = I_C(0,0);
	double I_yy = I_C(1,1);
	double I_zz = I_C(2,2);
	double I_xy = I_C(0,1);
	double I_xz = I_C(0,2);
	double I_yz = I_C(1,2);

	arma::vec moments = arma::eig_sym(I_C);


	arma::vec I = {I_xx,I_yy,I_zz,I_xy,I_xz,I_yz};

	arma::mat::fixed<6,6> Q = {
		{0,1./2.,1./2,0,0,0},
		{1./2.,0,1./2,0,0,0},
		{1./2,1./2.,0,0,0,0},
		{0,0,0,-1,0,0},
		{0,0,0,0,-1,0},
		{0,0,0,0,0,-1}
	};


	double Pi = arma::dot(I, Q * I);
	double U = std::sqrt(T * T - 3 * Pi)/3;
	double Theta = (-2 * std::pow(T,3) + 9 * T * Pi - 27 * d)/(54 * std::pow(U,3));
	double theta = std::acos(Theta);

	for (int a = 0; a < 3; ++a ){
		for (int b = 0; b < 3; ++b){

			P_Evectors.submat(3 * a, 3 * b, 3 * a + 2, 3 * b + 2) = (
				ShapeModelBezier::partial_E_partial_R(moments(a)) 
				* ShapeModelBezier::P_R_lambda_R_mu(moments(a), moments(b),a,b,theta,U) 
				* ShapeModelBezier::partial_E_partial_R(moments(b)).t()
				);
		}
	}


	this -> P_Evectors = P_Evectors;

}


arma::mat::fixed<3,3> ShapeModelBezier::ShapeModelBezier::P_ril_rjm(
	const double lambda, 
	const double mu,
	const int i,
	const int j,
	const int lambda_index,
	const int mu_index,
	const double theta,
	const double U) const{

	arma::mat::fixed<7,7> mat = arma::zeros<arma::mat>(7,7);

	mat.submat(0,0,5,5) = this -> P_I;
	mat.submat(6,0,6,5) = ShapeModelBezier::P_lambda_I(lambda_index,theta,U);
	mat.submat(0,6,5,6) = ShapeModelBezier::P_lambda_I(mu_index,theta,U).t();
	mat(6,6) = this -> P_moments(lambda_index,mu_index);



	arma::mat::fixed<3,3> P = ShapeModelBezier::partial_r_i_partial_I_lambda(i) * mat * ShapeModelBezier::partial_r_i_partial_I_lambda(j).t();


	return P;


}	



arma::rowvec::fixed<6> ShapeModelBezier::get_P_lambda_I(const int lambda_index) const {

	arma::mat I_C = this -> inertia - this -> get_volume() * RBK::tilde(this -> get_center_of_mass()) * RBK::tilde(this -> get_center_of_mass()).t() ;

	double T = arma::trace(I_C);
	double d = arma::det (I_C);

	double I_xx = I_C(0,0);
	double I_yy = I_C(1,1);
	double I_zz = I_C(2,2);
	double I_xy = I_C(0,1);
	double I_xz = I_C(0,2);
	double I_yz = I_C(1,2);

	arma::vec moments = arma::eig_sym(I_C);


	arma::vec I = {I_xx,I_yy,I_zz,I_xy,I_xz,I_yz};

	arma::mat::fixed<6,6> Q = {
		{0,1./2.,1./2,0,0,0},
		{1./2.,0,1./2,0,0,0},
		{1./2,1./2.,0,0,0,0},
		{0,0,0,-1,0,0},
		{0,0,0,0,-1,0},
		{0,0,0,0,0,-1}
	};


	double Pi = arma::dot(I, Q * I);
	double U = std::sqrt(T * T - 3 * Pi)/3;
	double Theta = (-2 * std::pow(T,3) + 9 * T * Pi - 27 * d)/(54 * std::pow(U,3));
	double theta = std::acos(Theta);





	return this -> P_lambda_I(lambda_index,theta,U);




}



arma::rowvec::fixed<6> ShapeModelBezier::P_lambda_I(const int lambda_index,
	const double theta, const double U) const{

	arma::rowvec::fixed<3> d_lambda_d_Y;
	if (lambda_index == 0){
		d_lambda_d_Y = ShapeModelBezier::partial_A_partial_Y(theta,U);
	}
	else if (lambda_index == 1){
		d_lambda_d_Y = ShapeModelBezier::partial_B_partial_Y(theta,U);
	}
	else if (lambda_index == 2){
		d_lambda_d_Y = ShapeModelBezier::partial_C_partial_Y(theta,U);

	}
	else{
		throw (std::runtime_error("unsupported case"));
	}

	return d_lambda_d_Y * ShapeModelBezier::partial_Y_partial_I() * this -> P_I;

}


arma::mat::fixed<3,6> ShapeModelBezier::partial_Y_partial_I() const{ 

	arma::mat::fixed<3,6> mat;

	mat.row(0) = ShapeModelBezier::partial_T_partial_I();
	mat.row(1) = ShapeModelBezier::partial_U_partial_I();
	mat.row(2) = ShapeModelBezier::partial_theta_partial_I();

	return mat;

}



arma::mat::fixed<3,7> ShapeModelBezier::ShapeModelBezier::partial_r_i_partial_I_lambda(const int i){

	arma::mat::fixed<3,7> mat;

	if (i == 0){

		mat = {
			{1,0,0,0,0,0,-1},
			{0,0,0,1,0,0,0},
			{0,0,0,0,1,0,0}
		};

	}
	else if (i == 1){

		mat = {
			{0,0,0,1,0,0,0},
			{0,1,0,0,0,0,-1},
			{0,0,0,0,0,1,0}
		};


	}
	else if (i == 2){
		mat = {
			{0,0,0,0,1,0,0},
			{0,0,0,0,0,1,0},
			{0,0,1,0,0,0,-1}
		};
	}
	else{
		throw (std::runtime_error("unsupported case"));
	}
	return mat;

}


arma::mat::fixed<3,3> ShapeModelBezier::partial_elambda_Elambda(const double & lambda) const{


	arma::mat I_C = this -> inertia - this -> get_volume() * RBK::tilde(this -> get_center_of_mass()) * RBK::tilde(this -> get_center_of_mass()).t() ;

	arma::mat L = I_C - lambda * arma::eye<arma::mat>(3,3);

	arma::vec Elambda;

	arma::vec norm_vec = {
		arma::norm(arma::cross(L.row(0).t(),L.row(1).t()))/(arma::norm(L.row(0)) * arma::norm(L.row(1))),
		arma::norm(arma::cross(L.row(0).t(),L.row(2).t()))/(arma::norm(L.row(0)) * arma::norm(L.row(2))),
		arma::norm(arma::cross(L.row(1).t(),L.row(2).t()))/(arma::norm(L.row(1)) * arma::norm(L.row(2)))
	};

	int i = norm_vec.index_max();

	if (i == 0){
		Elambda = arma::cross(L.row(0).t(),L.row(1).t());
	}
	else if (i == 1){
		Elambda = arma::cross(L.row(0).t(),L.row(2).t());
	}
	else if (i == 2){
		Elambda = arma::cross(L.row(1).t(),L.row(2).t());
	}
	else{
		throw (std::runtime_error("case not supported"));
	}

	arma::mat::fixed<3,3> partial = 1./arma::norm(Elambda) * (arma::eye<arma::mat>(3,3) - Elambda * Elambda.t() / arma::dot(Elambda,Elambda));



	return partial;
}

arma::mat::fixed<9,9> ShapeModelBezier::P_R_lambda_R_mu(const double lambda, 
	const double mu,
	const int lambda_index,
	const int mu_index,
	const double theta,
	const double U) const {

	arma::mat::fixed<9,9> mat = arma::zeros<arma::mat>(9,9); 

	for (int i = 0; i < 3 ; ++ i){

		for (int j = 0; j < 3 ; ++ j){

			mat.submat(3 * i, 3 * j, 3 * i + 2, 3 * j + 2) = ShapeModelBezier::P_ril_rjm(lambda, mu,
				i,j,lambda_index,mu_index,theta,U);

		}

	}

	return mat;


}


arma::mat::fixed<9,9> ShapeModelBezier::P_E_lambda_E_mu() const {

	arma::mat::fixed<9,9> mat;

	arma::mat I_C = this -> inertia - this -> get_volume() * RBK::tilde(this -> get_center_of_mass()) * RBK::tilde(this -> get_center_of_mass()).t() ;

	double T = arma::trace(I_C);
	double d = arma::det (I_C);

	double I_xx = I_C(0,0);
	double I_yy = I_C(1,1);
	double I_zz = I_C(2,2);
	double I_xy = I_C(0,1);
	double I_xz = I_C(0,2);
	double I_yz = I_C(1,2);

	arma::vec moments = arma::eig_sym(I_C);


	arma::vec I = {I_xx,I_yy,I_zz,I_xy,I_xz,I_yz};

	arma::mat::fixed<6,6> Q = {
		{0,1./2.,1./2,0,0,0},
		{1./2.,0,1./2,0,0,0},
		{1./2,1./2.,0,0,0,0},
		{0,0,0,-1,0,0},
		{0,0,0,0,-1,0},
		{0,0,0,0,0,-1}
	};

	double Pi = arma::dot(I, Q * I);
	double U = std::sqrt(T * T - 3 * Pi)/3;
	double Theta = (-2 * std::pow(T,3) + 9 * T * Pi - 27 * d)/(54 * std::pow(U,3));
	double theta = std::acos(Theta);


	for (int a = 0; a < 3 ; ++ a){

		double lambda = moments(a);

		for (int b = 0; b < 3 ; ++ b){

			double mu = moments(b);

			mat.submat(3 * a, 3 * b, 3 * a + 2, 3 * b + 2) = (
				ShapeModelBezier::partial_E_partial_R(lambda)
				* P_R_lambda_R_mu(lambda, mu,a,b,theta,U) 
				* ShapeModelBezier::partial_E_partial_R(mu).t());

		}
	}


}


arma::mat::fixed<3,9> ShapeModelBezier::partial_E_partial_R(const double lambda) const{

	arma::mat I_C = this -> inertia - this -> get_volume() * RBK::tilde(this -> get_center_of_mass()) * RBK::tilde(this -> get_center_of_mass()).t() ;

	double T = arma::trace(I_C);
	double d = arma::det (I_C);

	double I_xx = I_C(0,0);
	double I_yy = I_C(1,1);
	double I_zz = I_C(2,2);
	double I_xy = I_C(0,1);
	double I_xz = I_C(0,2);
	double I_yz = I_C(1,2);

	arma::vec moments = arma::eig_sym(I_C);

	arma::mat L = I_C - lambda * arma::eye<arma::mat>(3,3);

	arma::mat::fixed<3,9> dEdR = arma::zeros<arma::mat>(3,9);

	arma::vec norm_vec = {
		arma::norm(arma::cross(L.row(0).t(),L.row(1).t()))/(arma::norm(L.row(0)) * arma::norm(L.row(1))),
		arma::norm(arma::cross(L.row(0).t(),L.row(2).t()))/(arma::norm(L.row(0)) * arma::norm(L.row(2))),
		arma::norm(arma::cross(L.row(1).t(),L.row(2).t()))/(arma::norm(L.row(1)) * arma::norm(L.row(2)))
	};

	int i = norm_vec.index_max();



	if (i == 0){
		dEdR.submat(0,0,2,2) = - RBK::tilde(L.row(1).t());
		dEdR.submat(0,3,2,5) = RBK::tilde(L.row(0).t());

	}
	else if (i == 1){
		dEdR.submat(0,0,2,2) = - RBK::tilde(L.row(2).t());
		dEdR.submat(0,6,2,8) = RBK::tilde(L.row(0).t());

	}
	else{
		dEdR.submat(0,3,2,5) = - RBK::tilde(L.row(2).t());
		dEdR.submat(0,6,2,8) = RBK::tilde(L.row(1).t());

	}

	return dEdR;

}
