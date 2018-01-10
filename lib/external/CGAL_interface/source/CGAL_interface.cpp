#include "CGAL_interface.hpp"



void CGALINTERFACE::CGAL_interface(std::string input_path, std::string savepath) {

    // Poisson options
    FT sm_angle = 30.0; // Min triangle angle in degrees.
    FT sm_radius = 10; // Max triangle size w.r.t. point set average spacing.
    FT sm_distance = 3; // Surface Approximation error w.r.t. point set average spacing.

    // Reads the point set file in points[].
    // Note: read_xyz_points_and_normals() requires an iterator over points
    // + property maps to access each point's position and normal.
    // The position property map can be omitted here as we use iterators over Point_3 elements.
    PointList points;
    std::ifstream stream(input_path);
    if (!stream ||
        !CGAL::read_xyz_points_and_normals(
            stream,
            std::back_inserter(points),
            CGAL::make_normal_of_point_with_normal_pmap(PointList::value_type())))
    {
        throw (std::runtime_error("Error: cannot read file " + input_path));

    }


    // Creates implicit function from the read points using the default solver.

    // Note: this method requires an iterator over points
    // + property maps to access each point's position and normal.
    // The position property map can be omitted here as we use iterators over Point_3 elements.
    Poisson_reconstruction_function function(points.begin(), points.end(),
        CGAL::make_normal_of_point_with_normal_pmap(PointList::value_type()) );

    // Computes the Poisson indicator function f()
    // at each vertex of the triangulation.
    if ( ! function.compute_implicit_function() )
        throw (std::runtime_error("Error in computation of implicit function"));

    // Computes average spacing
    FT average_spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>(points.begin(), points.end(),
                         6 /* knn = 1 ring */);

    // Gets one point inside the implicit surface
    // and computes implicit function bounding sphere radius.
        Point inner_point = function.get_inner_point();
        Sphere bsphere = function.bounding_sphere();
        FT radius = std::sqrt(bsphere.squared_radius());

    // Defines the implicit surface: requires defining a
    // conservative bounding sphere centered at inner point.
        FT sm_sphere_radius = 5.0 * radius;
    FT sm_dichotomy_error = sm_distance * average_spacing / 1000.0; // Dichotomy error must be << sm_distance
    Surface_3 surface(function,
      Sphere(inner_point, sm_sphere_radius * sm_sphere_radius),
      sm_dichotomy_error / sm_sphere_radius);

    // Defines surface mesh generation criteria
    CGAL::Surface_mesh_default_criteria_3<STr> criteria(sm_angle,  // Min triangle angle (degrees)
            sm_radius * average_spacing, // Max triangle size
            sm_distance * average_spacing); // Approximation error

    // Generates surface mesh with manifold option
    STr tr; // 3D Delaunay triangulation for surface mesh generation
    C2t3 c2t3(tr); // 2D complex in 3D Delaunay triangulation
    CGAL::make_surface_mesh(c2t3,                                 // reconstructed mesh
                            surface,                              // implicit surface
                            criteria,                             // meshing criteria
                            CGAL::Manifold_with_boundary_tag());  // require manifold mesh

    if (tr.number_of_vertices() == 0)
        throw (std::runtime_error("Number of vertices equated 0"));

    // saves reconstructed surface mesh
    std::ofstream ofs(savepath);
    Polyhedron output_mesh;
    
    CGAL::output_surface_facets_to_polyhedron(c2t3, output_mesh);


    if ( ! CGAL::Polygon_mesh_processing::is_outward_oriented(output_mesh)) {
        throw (std::runtime_error("Spurious normal orientations in CGAL"));
    }


    // The Polyhedron is decimated

    if (!CGAL::is_triangle_mesh(output_mesh)){
     throw (std::runtime_error("Input geometry is not triangulated."));
 }

  // This is a stop predicate (defines when the algorithm terminates).
  // In this example, the simplification stops when the number of undirected edges
  // left in the surface mesh drops below the specified number (1000)
 SMS::Count_stop_predicate<Polyhedron> stop(600);

  // This the actual call to the simplification algorithm.
  // The surface mesh and stop conditions are mandatory arguments.
  // The index maps are needed because the vertices and edges
  // of this surface mesh lack an "id()" field.
 int r = SMS::edge_collapse
 (output_mesh
    ,stop
    ,CGAL::parameters::vertex_index_map(get(CGAL::vertex_external_index,output_mesh)) 
    .halfedge_index_map  (get(CGAL::halfedge_external_index  ,output_mesh)) 
    .get_cost (SMS::Edge_length_cost <Polyhedron>())
    .get_placement(SMS::Midpoint_placement<Polyhedron>())
    );

 std::cout << "\nFinished...\n" << r << " edges removed.\n" 
 << (output_mesh.size_of_halfedges()/2) << " final edges.\n" ;














    // out << output_mesh;
 CGAL::print_polyhedron_wavefront(ofs, output_mesh);


    /// [PMP_distance_snippet]
    // computes the approximation error of the reconstruction
 double max_dist =
 CGAL::Polygon_mesh_processing::approximate_max_distance_to_point_set(output_mesh,
    points,
    4000);
 std::cout << "Max distance to point_set: " << max_dist << std::endl;
    /// [PMP_distance_snippet]

}
