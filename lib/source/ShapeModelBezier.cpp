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

void ShapeModelBezier::compute_inertia(){

	arma::mat::fixed<3,3> inertia;
	inertia.fill(0);
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
			for (auto it_neighbors = neighbors.begin(); it_neighbors  != neighbors.end(); ++it_neighbors){

				Bezier * patch_f = static_cast<Bezier * >(*it_neighbors);

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


}

arma::rowvec ShapeModelBezier::L_row(int q, int r, const arma::vec * Ci,const arma::vec * Cj,const arma::vec * Ck,const arma::vec * Cl,const arma::vec * Cm){


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











void ShapeModelBezier::run_monte_carlo(int N,
	arma::vec & results_volume,
	arma::mat & results_cm,
	arma::mat & results_inertia,
	arma::mat & results_moments){

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
		this -> compute_inertia();

		arma::mat I_C = this -> inertia - this -> get_volume() * RBK::tilde(this -> get_center_of_mass()) * RBK::tilde(this -> get_center_of_mass()).t() ;
		arma::vec I = {I_C(0,0),I_C(1,1),I_C(2,2),I_C(0,1),I_C(0,2),I_C(1,2)};

		results_volume(iter) = this -> get_volume();
		results_cm.col(iter) = this -> get_center_of_mass();
		results_inertia.col(iter) = I;
		arma::vec col(4);
		col.rows(0,2) = arma::eig_sym(I_C);
		col(3) = this -> get_volume();
		results_moments.col(iter) = col;

	}



	// Cleaning up
	for (unsigned int i = 0; i < this -> get_NControlPoints(); ++i){
		this -> control_points[i] -> set_coordinates(this -> control_points[i] -> get_mean_coordinates());
	}


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
		std::vector<double> index_vector = {double(i),double(j),double(k),double(l),double(m),double(p),double(q),double(r),
			double(s),double(t),kappa
		};
		this -> inertia_indices_coefs_table.push_back(index_vector);
	}



	std::cout << "- Inertia coefficients: " << this -> inertia_indices_coefs_table.size() + this -> cm_gamma_indices_coefs_table.size() << std::endl;






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
		// if (std::abs(kappa_kappa) > 1e-13){
		std::vector<double> index_vector = {double(i),double(j),double(k),double(l),double(m),double(p),double(q),double(r),
			double(s),double(t),
			double(u),double(v),double(w),double(x),double(y),double(z),double(a),double(b),
			double(c),double(d),
			kappa_kappa
		};
		this -> inertia_stats_1_indices_coefs_table.push_back(index_vector);
		// }
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
		std::vector<double> index_vector = {double(i),double(j),double(k),double(l),double(m),double(p),double(q),double(r),
			double(s),double(t),
			double(u),double(v),double(w),double(x),double(y),double(z),
			alpha_kappa
		};
		this -> inertia_stats_2_indices_coefs_table.push_back(index_vector);

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



void ShapeModelBezier::compute_inertia_statistics() {
	std::cout << "\n- Computing inertia statistics ...\n";

	this -> compute_P_I();
	this -> compute_P_IV();
	this -> compute_P_Y();
	this -> compute_P_moments();


}






void ShapeModelBezier::compute_P_I(){


	arma::mat::fixed<6,6> P_I,increment;
	P_I.fill(0);


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

			for (auto it_neighbors = neighbors.begin(); it_neighbors  != neighbors.end(); ++it_neighbors){


				Bezier * patch_f = static_cast<Bezier * >(*it_neighbors);

				int p_g,q_g,r_g,s_g,t_g;

				p_g = patch_f -> get_control_point_global_index(u,v);
				q_g = patch_f -> get_control_point_global_index(w,x);
				r_g = patch_f -> get_control_point_global_index(y,z);
				s_g = patch_f -> get_control_point_global_index(a,b);
				t_g = patch_f -> get_control_point_global_index(c,d);	

				this -> construct_inertia_mapping_mat(right_mat,p_g,q_g,r_g,s_g,t_g);

				increment = this -> increment_P_I(left_mat,
					right_mat, 
					i_g,j_g,k_g,l_g,m_g, 
					p_g,q_g,r_g,s_g,t_g);

				P_I += coefs_row[20] * increment;
			}
		}

		++progress;
	}

	this -> P_I = P_I;

}


void ShapeModelBezier::compute_P_IV(){

	arma::vec::fixed<6> P_IV,increment;
	P_IV.fill(0);

	std::vector<std::set < Element * > > connected_elements;

	for (unsigned int e = 0; e < this -> elements.size(); ++e) {

		auto elements = this -> elements[e] -> get_neighbors(true);
		connected_elements.push_back(elements);
	}

	boost::progress_display progress(this -> inertia_stats_2_indices_coefs_table.size()) ;


	#pragma omp parallel for reduction (+:P_IV)

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

			for (auto it_neighbors = neighbors.begin(); it_neighbors  != neighbors.end(); ++it_neighbors){


				Bezier * patch_f = static_cast<Bezier * >(*it_neighbors);

				int p_g,q_g,r_g;

				p_g = patch_f -> get_control_point_global_index(u,v);
				q_g = patch_f -> get_control_point_global_index(w,x);
				r_g = patch_f -> get_control_point_global_index(y,z);



				right_vec = patch_f -> get_cross_products(u,v,w,x,y,z);

				increment = this -> increment_P_IV(left_mat,
					right_vec, 
					i_g,j_g,k_g,l_g,m_g, 
					p_g,q_g,r_g);


				P_IV += coefs_row[16] * increment;
			}
		}

		++progress;
	}

	this -> P_IV = P_IV;

}











arma::mat::fixed<6,6> ShapeModelBezier::increment_P_I(arma::mat::fixed<6,15> & left_mat,
	arma::mat::fixed<6,15>  & right_mat, 
	int i,int j,int k,int l,int m,
	int p, int q, int r, int s, int t){

	arma::mat::fixed<15,15> P_CC = arma::zeros<arma::mat>(15,15);

	// First row
	if (i == p){
		P_CC.submat(0,0,2,2) = this -> control_points.at(i) -> get_covariance() ;
	}
	if (i == q){
		P_CC.submat(0,3,2,5) = this -> control_points.at(i) -> get_covariance() ;
	}
	if (i == r){
		P_CC.submat(0,6,2,8) = this -> control_points.at(i) -> get_covariance() ;
	}
	if (i == s){
		P_CC.submat(0,9,2,11) = this -> control_points.at(i) -> get_covariance() ;
	}
	if (i == t){
		P_CC.submat(0,12,2,14) = this -> control_points.at(i) -> get_covariance() ;
	}

	// Second row

	if (j == p){
		P_CC.submat(3,0,5,2) = this -> control_points.at(j) -> get_covariance() ;
	}
	if (j == q){
		P_CC.submat(3,3,5,5) = this -> control_points.at(j) -> get_covariance() ;
	}
	if (j == r){
		P_CC.submat(3,6,5,8) = this -> control_points.at(j) -> get_covariance() ;
	}
	if (j == s){
		P_CC.submat(3,9,5,11) = this -> control_points.at(j) -> get_covariance() ;
	}
	if (j == t){
		P_CC.submat(3,12,5,14) = this -> control_points.at(j) -> get_covariance() ;
	}

	// Third row

	if (k == p){
		P_CC.submat(6,0,8,2) = this -> control_points.at(k) -> get_covariance() ;
	}
	if (k == q){
		P_CC.submat(6,3,8,5) = this -> control_points.at(k) -> get_covariance() ;
	}
	if (k == r){
		P_CC.submat(6,6,8,8) = this -> control_points.at(k) -> get_covariance() ;
	}
	if (k == s){
		P_CC.submat(6,9,8,11) = this -> control_points.at(k) -> get_covariance() ;
	}
	if (k == t){
		P_CC.submat(6,12,8,14) = this -> control_points.at(k) -> get_covariance() ;
	}


	// Fourth row
	if (l == p){
		P_CC.submat(9,0,11,2) = this -> control_points.at(l) -> get_covariance() ;
	}
	if (l == q){
		P_CC.submat(9,3,11,5) = this -> control_points.at(l) -> get_covariance() ;
	}
	if (l == r){
		P_CC.submat(9,6,11,8) = this -> control_points.at(l) -> get_covariance() ;
	}
	if (l == s){
		P_CC.submat(9,9,11,11) = this -> control_points.at(l) -> get_covariance() ;
	}
	if (l == t){
		P_CC.submat(9,12,11,14) = this -> control_points.at(l) -> get_covariance() ;
	}


	// Fifth row
	if (m == p){
		P_CC.submat(12,0,14,2) = this -> control_points.at(m) -> get_covariance() ;
	}
	if (m == q){
		P_CC.submat(12,3,14,5) = this -> control_points.at(m) -> get_covariance() ;
	}
	if (m == r){
		P_CC.submat(12,6,14,8) = this -> control_points.at(m) -> get_covariance() ;
	}
	if (m == s){
		P_CC.submat(12,9,14,11) = this -> control_points.at(m) -> get_covariance() ;
	}
	if (m == t){
		P_CC.submat(12,12,14,14) = this -> control_points.at(m) -> get_covariance() ;
	}


	return left_mat * P_CC * right_mat.t();

}



arma::vec::fixed<6> ShapeModelBezier::increment_P_IV(arma::mat::fixed<6,15> & left_mat,
	arma::vec::fixed<9>  & right_vec, 
	int i,int j,int k,int l,int m,
	int p, int q, int r){

	arma::mat::fixed<15,9> P_CC = arma::zeros<arma::mat>(15,9);

	// First row
	if (i == p){
		P_CC.submat(0,0,2,2) = this -> control_points.at(i) -> get_covariance() ;
	}
	if (i == q){
		P_CC.submat(0,3,2,5) = this -> control_points.at(i) -> get_covariance() ;
	}
	if (i == r){
		P_CC.submat(0,6,2,8) = this -> control_points.at(i) -> get_covariance() ;
	}


	// Second row

	if (j == p){
		P_CC.submat(3,0,5,2) = this -> control_points.at(j) -> get_covariance() ;
	}
	if (j == q){
		P_CC.submat(3,3,5,5) = this -> control_points.at(j) -> get_covariance() ;
	}
	if (j == r){
		P_CC.submat(3,6,5,8) = this -> control_points.at(j) -> get_covariance() ;
	}


	// Third row

	if (k == p){
		P_CC.submat(6,0,8,2) = this -> control_points.at(k) -> get_covariance() ;
	}
	if (k == q){
		P_CC.submat(6,3,8,5) = this -> control_points.at(k) -> get_covariance() ;
	}
	if (k == r){
		P_CC.submat(6,6,8,8) = this -> control_points.at(k) -> get_covariance() ;
	}



	// Fourth row
	if (l == p){
		P_CC.submat(9,0,11,2) = this -> control_points.at(l) -> get_covariance() ;
	}
	if (l == q){
		P_CC.submat(9,3,11,5) = this -> control_points.at(l) -> get_covariance() ;
	}
	if (l == r){
		P_CC.submat(9,6,11,8) = this -> control_points.at(l) -> get_covariance() ;
	}



	// Fifth row
	if (m == p){
		P_CC.submat(12,0,14,2) = this -> control_points.at(m) -> get_covariance() ;
	}
	if (m == q){
		P_CC.submat(12,3,14,5) = this -> control_points.at(m) -> get_covariance() ;
	}
	if (m == r){
		P_CC.submat(12,6,14,8) = this -> control_points.at(m) -> get_covariance() ;
	}

	return left_mat * P_CC * right_vec;

}






arma::mat::fixed<3,3> ShapeModelBezier::increment_cm_cov(arma::mat::fixed<12,3> & left_mat,
	arma::mat::fixed<12,3>  & right_mat, 
	int i,int j,int k,int l, 
	int m, int p, int q, int r){

	arma::mat::fixed<12,12> P_CC = arma::zeros<arma::mat>(12,12);

		// First row

	if (i == m){
		P_CC.submat(0,0,2,2) = this -> control_points.at(i) -> get_covariance() ;
	}
	if (i == p){
		P_CC.submat(0,3,2,5) = this -> control_points.at(i) -> get_covariance() ;
	}
	if (i == q){
		P_CC.submat(0,6,2,8) = this -> control_points.at(i) -> get_covariance() ;
	}
	if (i == r){
		P_CC.submat(0,9,2,11) = this -> control_points.at(i) -> get_covariance() ;
	}


	// Second row

	if (j == m){
		P_CC.submat(3,0,5,2) = this -> control_points.at(j) -> get_covariance() ;
	}
	if (j == p){
		P_CC.submat(3,3,5,5) = this -> control_points.at(j) -> get_covariance() ;
	}
	if (j == q){
		P_CC.submat(3,6,5,8) = this -> control_points.at(j) -> get_covariance() ;
	}
	if (j == r){
		P_CC.submat(3,9,5,11) = this -> control_points.at(j) -> get_covariance() ;
	}


	// Third row

	if (k == m){
		P_CC.submat(6,0,8,2) = this -> control_points.at(k) -> get_covariance() ;
	}
	if (k == p){
		P_CC.submat(6,3,8,5) = this -> control_points.at(k) -> get_covariance() ;
	}
	if (k == q){
		P_CC.submat(6,6,8,8) = this -> control_points.at(k) -> get_covariance() ;
	}
	if (k == r){
		P_CC.submat(6,9,8,11) = this -> control_points.at(k) -> get_covariance() ;
	}


	// Fourth row
	if (l == m){
		P_CC.submat(9,0,11,2) = this -> control_points.at(l) -> get_covariance() ;
	}
	if (l == p){
		P_CC.submat(9,3,11,5) = this -> control_points.at(l) -> get_covariance() ;
	}
	if (l == q){
		P_CC.submat(9,6,11,8) = this -> control_points.at(l) -> get_covariance() ;
	}
	if (l == r){
		P_CC.submat(9,9,11,11) = this -> control_points.at(l) -> get_covariance() ;
	}


	return left_mat.t() * P_CC * right_mat;
}



void ShapeModelBezier::compute_P_Y(){

	double sigma_TT = arma::dot(ShapeModelBezier::partial_T_partial_I() .t(), 
		this -> P_I * ShapeModelBezier::partial_T_partial_I().t());
	double sigma_TU = arma::dot(ShapeModelBezier::partial_T_partial_I() .t(), 
		this -> P_I * ShapeModelBezier::partial_U_partial_I().t());
	double sigma_Tt = arma::dot(ShapeModelBezier::partial_T_partial_I() .t(), 
		this -> P_I * ShapeModelBezier::partial_theta_partial_I().t());
	double sigma_UU = arma::dot(ShapeModelBezier::partial_U_partial_I() .t(), 
		this -> P_I * ShapeModelBezier::partial_U_partial_I().t());
	double sigma_Ut = arma::dot(ShapeModelBezier::partial_U_partial_I() .t(), 
		this -> P_I * ShapeModelBezier::partial_theta_partial_I().t());
	double sigma_tt = arma::dot(ShapeModelBezier::partial_theta_partial_I() .t(), 
		this -> P_I * ShapeModelBezier::partial_U_partial_I().t());

	this -> P_Y = {
		{sigma_TT,sigma_TU,sigma_Tt},
		{sigma_TU,sigma_UU,sigma_Ut},
		{sigma_Tt,sigma_Ut,sigma_tt}
	};
}


arma::rowvec::fixed<6> ShapeModelBezier::partial_T_partial_I() {
	arma::rowvec dTdI = {1,1,1,0,0,0};
	return dTdI;
}




arma::rowvec::fixed<6> ShapeModelBezier::partial_theta_partial_I() const {

	double T = arma::trace(this -> inertia);
	double d = arma::det (this -> inertia);

	double I_xx = this -> inertia(0,0);
	double I_yy = this -> inertia(1,1);
	double I_zz = this -> inertia(2,2);
	double I_xy = this -> inertia(0,1);
	double I_xz = this -> inertia(0,2);
	double I_yz = this -> inertia(1,2);

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

	double I_xx = this -> inertia(0,0);
	double I_yy = this -> inertia(1,1);
	double I_zz = this -> inertia(2,2);
	double I_xy = this -> inertia(0,1);
	double I_xz = this -> inertia(0,2);
	double I_yz = this -> inertia(1,2);

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

	double T = arma::trace(this -> inertia);
	double d = arma::det (this -> inertia);

	double I_xx = this -> inertia(0,0);
	double I_yy = this -> inertia(1,1);
	double I_zz = this -> inertia(2,2);
	double I_xy = this -> inertia(0,1);
	double I_xz = this -> inertia(0,2);
	double I_yz = this -> inertia(1,2);

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

	double I_xx = this -> inertia(0,0);
	double I_yy = this -> inertia(1,1);
	double I_zz = this -> inertia(2,2);
	double I_xy = this -> inertia(0,1);
	double I_xz = this -> inertia(0,2);
	double I_yz = this -> inertia(1,2);

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

	arma::mat::fixed<4,4> P_moments;

	double T = arma::trace(this -> inertia);
	double d = arma::det (this -> inertia);

	double I_xx = this -> inertia(0,0);
	double I_yy = this -> inertia(1,1);
	double I_zz = this -> inertia(2,2);
	double I_xy = this -> inertia(0,1);
	double I_xz = this -> inertia(0,2);
	double I_yz = this -> inertia(1,2);

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

	auto dAdY = ShapeModelBezier::partial_A_partial_Y(theta,U);
	auto dBdY = ShapeModelBezier::partial_B_partial_Y(theta,U);
	auto dCdY = ShapeModelBezier::partial_C_partial_Y(theta,U);

	arma::vec::fixed<3> sigma_sq_Y =  {
		arma::dot(ShapeModelBezier::partial_T_partial_I().t() , this -> P_IV),
		arma::dot(ShapeModelBezier::partial_U_partial_I().t() , this -> P_IV),
		arma::dot(ShapeModelBezier::partial_theta_partial_I().t() , this -> P_IV)
	};



	P_moments(0,0) = arma::dot(dAdY.t(), this -> P_Y * dAdY.t());
	P_moments(1,1) = arma::dot(dBdY.t(), this -> P_Y * dBdY.t());
	P_moments(2,2) = arma::dot(dCdY.t(), this -> P_Y * dCdY.t());

	P_moments(0,1) = arma::dot(dAdY.t(), this -> P_Y * dBdY.t());
	P_moments(1,0) = P_moments(0,1);

	P_moments(1,2) = arma::dot(dBdY.t(), this -> P_Y * dCdY.t());
	P_moments(2,1) = P_moments(1,2);

	P_moments(0,2) = arma::dot(dAdY.t(), this -> P_Y * dCdY.t());
	P_moments(2,0) = P_moments(0,2);

	P_moments(0,3) = arma::dot(dAdY.t(),sigma_sq_Y);
	P_moments(3,0) = P_moments(0,3);

	P_moments(1,3) = arma::dot(dBdY.t(),sigma_sq_Y);
	P_moments(3,1) = P_moments(1,3);

	P_moments(2,3) = arma::dot(dCdY.t(),sigma_sq_Y);
	P_moments(3,2) = P_moments(2,3);

	P_moments(3,3) = std::pow(this -> volume_sd,2);

	this -> P_moments = P_moments;

}




