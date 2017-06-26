#ifndef HEADER_KDTree_pc
#define HEADER_KDTree_pc

class KDTree_pc {

public:
	std::shared_ptr<KDTree_pc> left;
	std::shared_ptr<KDTree_pc> right;
	std::vector<PointNormals * > points_normals;

	KDTree_pc();

	std::shared_ptr<KDNode> build(std::vector<PointNormals * > & points_normals, int depth, bool verbose = false);


	int get_depth() const;
	void set_depth(int depth);

protected:

	int depth;
	int max_depth = 1000;

};


#endif