#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <opencv2/core/core.hpp>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/registration/icp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <thread>
#include <pcl/point_cloud.h>
#include <pcl/registration/ndt.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/boost.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/surface/convex_hull.h>


//遍历后缀.xyz
#include <regex>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
namespace fs = boost::filesystem;

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

//https://github.com/leaveitout/pcl_point_cloud_merger/blob/master/src/main.cpp
typedef pcl::PointXYZRGB PointType;
typedef pcl::PointXYZRGBNormal PointNormalType;
typedef pcl::PointCloud<PointType> Cloud;
typedef Cloud::Ptr CloudPtr;
typedef Cloud::ConstPtr CloudConstPtr;
Eigen::Matrix4f final_transform;


using namespace std;
using namespace pcl;
using namespace pcl::console;
using namespace pcl::registration;
using namespace pcl::io;
using namespace std::chrono_literals;

pcl::PointCloud<pcl::PointXYZRGB>::Ptr pre_passThrough_x(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,float x_min, float x_max)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_return(new pcl::PointCloud<pcl::PointXYZRGB>);
	//直通滤波
	pcl::PassThrough<pcl::PointXYZRGB> pass;//设置滤波器对象
	//参数设置
	pass.setInputCloud(cloud);
	pass.setFilterFieldName("x");
	//z轴区间设置
	pass.setFilterLimits(x_min, x_max);
	//设置为保留还是去除
	//pass_z.setFilterLimitsNegative(false);//设置过滤器限制负//设置保留范围内false
	pass.filter(*cloud_return);
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr pre_passThrough_y(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, float y_min, float y_max)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_return(new pcl::PointCloud<pcl::PointXYZRGB>);
	//直通滤波
	pcl::PassThrough<pcl::PointXYZRGB> pass;//设置滤波器对象
	//参数设置
	pass.setInputCloud(cloud);
	pass.setFilterFieldName("y");
	//z轴区间设置
	pass.setFilterLimits(y_min, y_max);
	//设置为保留还是去除
	//pass_z.setFilterLimitsNegative(false);//设置过滤器限制负//设置保留范围内false
	pass.filter(*cloud_return);
}
pcl::PointCloud<pcl::PointXYZRGB>::Ptr pre_passThrough_z(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, float z_min, float z_max)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_return(new pcl::PointCloud<pcl::PointXYZRGB>);
	//直通滤波
	pcl::PassThrough<pcl::PointXYZRGB> pass;//设置滤波器对象
	//参数设置
	pass.setInputCloud(cloud);
	pass.setFilterFieldName("z");
	//z轴区间设置
	pass.setFilterLimits(z_min, z_max);
	//设置为保留还是去除
	//pass_z.setFilterLimitsNegative(false);//设置过滤器限制负//设置保留范围内false
	pass.filter(*cloud_return);
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr pre_statistical_outlier_removal(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZRGB>);
	//统计滤波
	pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> outrem;//创建统计滤波对象

	//参数设置
	outrem.setInputCloud(cloud);
	outrem.setMeanK(200);//附近邻近点数
	outrem.setStddevMulThresh(1);//判断是否离群点
	outrem.filter(*cloud_f);

	return cloud_f;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr pre_voxel_filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
	//读取点云
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZRGB>);

	//体素滤波
	pcl::VoxelGrid<pcl::PointXYZRGB> vox;
	vox.setInputCloud(cloud);
	vox.setLeafSize(5.0f, 5.0f, 5.0f);//体素网格大小
	vox.filter(*cloud_f);

	return cloud_f;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr merge_two_pointscloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged_pointscloud((new pcl::PointCloud<pcl::PointXYZRGB>));
	*merged_pointscloud += *cloud;
	*merged_pointscloud += *cloud2;

	merged_pointscloud = pre_voxel_filter(merged_pointscloud);
	return  merged_pointscloud;
}

//打印旋转矩阵和平移矩阵
void print4x4Matrix(const Eigen::Matrix4d& matrix)    
{
	printf("Rotation matrix :\n");
	printf("    | %6.3f %6.3f %6.3f | \n", matrix(0, 0), matrix(0, 1), matrix(0, 2));
	printf("R = | %6.3f %6.3f %6.3f | \n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
	printf("    | %6.3f %6.3f %6.3f | \n", matrix(2, 0), matrix(2, 1), matrix(2, 2));
	printf("Translation vector :\n");
	printf("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix(0, 3), matrix(1, 3), matrix(2, 3));
}

//遍历后缀
vector<string> find_files_with_suffix(std::string path, std::string suffix)
{
	std::vector<std::string> files;
	
	regex reg_obj(suffix, regex::icase);

	fs::path full_path = fs::system_complete(fs::path(path));

	unsigned long file_count = 0;
	unsigned long err_count = 0;

	if (fs::is_directory(full_path))
	{
		fs::directory_iterator end_iter;
		for (fs::directory_iterator dir_itr(full_path);
			dir_itr != end_iter;
			++dir_itr)
		{
			try
			{
				//if (fs::is_directory(dir_itr->status()))
				//{
				//    ++dir_count;
				//    std::cout << dir_itr->path().filename() << " [directory]\n";
				//}
				if (fs::is_regular_file(dir_itr->status()) )
				{
					if (regex_search(dir_itr->path().filename().string(), reg_obj))
					{
						string suffix_path = path + dir_itr->path().filename().string();
						files.push_back(suffix_path);
						++file_count;
						std::cout << dir_itr->path().filename() << "\n";
					}
				}
				//else
				//{
				//    ++other_count;
				//    std::cout << dir_itr->path().filename() << " [other]\n";
				//}
			}
			catch (const std::exception& ex)
			{
				++err_count;
				std::cout << dir_itr->path().filename() << " " << ex.what() << std::endl;
			}
		}
	}
	return files;
}

//https://github.com/leaveitout/pcl_point_cloud_merger/blob/master/src/main.cpp
// Get the indices of the points in the cloud that belong to the dominant plane
pcl::PointIndicesPtr extractPlaneIndices(Cloud::Ptr cloud,
	double plane_thold = 0.02) {
	// Object for storing the plane model coefficients.
	auto coefficients = boost::make_shared<pcl::ModelCoefficients>();
	// Create the segmentation object.
	auto segmentation = pcl::SACSegmentation<PointType>{};
	segmentation.setInputCloud(cloud);
	// Configure the object to look for a plane.
	segmentation.setModelType(pcl::SACMODEL_PLANE);
	// Use RANSAC method.
	segmentation.setMethodType(pcl::SAC_RANSAC);
	// Set the maximum allowed distance to the model.
	segmentation.setDistanceThreshold(plane_thold);
	// Enable model coefficient refinement (optional).
	segmentation.setOptimizeCoefficients(true);

	auto inlierIndices = boost::make_shared<pcl::PointIndices>();
	segmentation.segment(*inlierIndices, *coefficients);

	return inlierIndices;
}


// Euclidean Clustering
pcl::PointIndicesPtr euclideanClustering(Cloud::Ptr cloud,
	double cluster_tolerance = 0.005,
	int min_size = 100,
	int max_size = 307200) {
	// kd-tree object for searches.
	auto kd_tree = boost::make_shared<pcl::search::KdTree<PointType>>();
	kd_tree->setInputCloud(cloud);

	// Euclidean clustering object.
	pcl::EuclideanClusterExtraction<PointType> clustering;
	// Set cluster tolerance to 1cm (small values may cause objects to be divided
	// in several clusters, whereas big values may join objects in a same cluster).
	clustering.setClusterTolerance(cluster_tolerance);
	// Set the minimum and maximum number of points that a cluster can have.
	clustering.setMinClusterSize(min_size);
	clustering.setMaxClusterSize(max_size);
	clustering.setSearchMethod(kd_tree);
	clustering.setInputCloud(cloud);
	std::vector<pcl::PointIndices> clusters;
	clustering.extract(clusters);

	// Find largest cluster and return indices
	size_t largest_cluster_index = 0;
	size_t current_cluster_index = 0;
	size_t max_points = 0;
	for (const auto& cluster : clusters) {
		if (cluster.indices.size() > max_points) {
			max_points = cluster.indices.size();
			largest_cluster_index = current_cluster_index;
		}
		current_cluster_index++;
	}

	pcl::PointIndicesPtr indicesPtr = boost::make_shared<pcl::PointIndices>(clusters.at(largest_cluster_index));
	return indicesPtr;
}


// Region Growing
pcl::PointIndicesPtr regionGrowing(Cloud::Ptr cloud,
	double smoothness_degrees = 7.0,
	double curvature_thold = 1.0,
	double normal_radius_search = 0.01,
	size_t num_neighbours = 30,
	size_t min_size = 100,
	size_t max_size = 307200) {
	// kd-tree object for searches.
	pcl::search::KdTree<PointType>::Ptr kdTree(new pcl::search::KdTree<PointType>);
	kdTree->setInputCloud(cloud);

	// Estimate the normals.
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::NormalEstimation<PointType, pcl::Normal> normalEstimation;
	normalEstimation.setInputCloud(cloud);
	normalEstimation.setRadiusSearch(normal_radius_search);
	normalEstimation.setSearchMethod(kdTree);
	normalEstimation.compute(*normals);

	// Region growing clustering object.
	pcl::RegionGrowing<PointType, pcl::Normal> clustering;
	clustering.setMinClusterSize((int)min_size);
	clustering.setMaxClusterSize((int)max_size);
	clustering.setSearchMethod(kdTree);
	clustering.setNumberOfNeighbours((int)num_neighbours);
	clustering.setInputCloud(cloud);
	clustering.setInputNormals(normals);
	// Set the angle in radians that will be the smoothness threshold
	// (the maximum allowable deviation of the normals).
	clustering.setSmoothnessThreshold((float)(smoothness_degrees / 180.0 * M_PI)); // degrees.
	// Set the curvature threshold. The disparity between curvatures will be
	// tested after the normal deviation check has passed.
	clustering.setCurvatureThreshold((float)curvature_thold);

	std::vector<pcl::PointIndices> clusters;
	clustering.extract(clusters);

	// Find largest cluster and return indices
	size_t largest_cluster_index = 0;
	size_t current_cluster_index = 0;
	size_t max_points = 0;
	for (const auto& cluster : clusters) {
		if (cluster.indices.size() > max_points) {
			max_points = cluster.indices.size();
			largest_cluster_index = current_cluster_index;
		}
		current_cluster_index++;
	}

	pcl::PointIndicesPtr indicesPtr = boost::make_shared<pcl::PointIndices>(clusters.at(largest_cluster_index));
	return indicesPtr;
}




pcl::PointCloud<PointNormalType>::Ptr addNormals(pcl::PointCloud<PointType>::Ptr cloud) {
	auto normal_cloud = boost::make_shared<pcl::PointCloud<pcl::Normal>>();
	auto cloud_xyz = boost::make_shared< pcl::PointCloud<pcl::PointXYZ>>();
	pcl::copyPointCloud(*cloud, *cloud_xyz);

	auto search_tree = boost::make_shared<pcl::search::KdTree<PointType>>();
	search_tree->setInputCloud(cloud);

	auto normal_estimator = std::make_unique<pcl::NormalEstimation<PointType, pcl::Normal>>();
	normal_estimator->setInputCloud(cloud);
	normal_estimator->setSearchMethod(search_tree);
	normal_estimator->setKSearch(30);
	normal_estimator->compute(*normal_cloud);

	auto result_cloud = boost::make_shared<pcl::PointCloud<PointNormalType>>();
	pcl::concatenateFields(*cloud_xyz, *normal_cloud, *result_cloud);
	return result_cloud;
}


void pairAlign(const CloudPtr cloud_src,
	const CloudPtr cloud_tgt,
	CloudPtr output,
	Eigen::Matrix4f& final_transform,
	bool downsample = false) {
	//
	// Downsample for consistency and speed
	// \note enable this for large datasets
	auto src = boost::make_shared<Cloud>();
	auto tgt = boost::make_shared<Cloud>();
	auto grid = pcl::VoxelGrid<PointType>{};
	if (downsample) {
		grid.setLeafSize(5, 5, 5);
		grid.setInputCloud(cloud_src);
		grid.filter(*src);

		grid.setInputCloud(cloud_tgt);
		grid.filter(*tgt);
	}
	else {
		src = cloud_src;
		tgt = cloud_tgt;
	}

	std::cout << src->size() << std::endl;
	std::cout << tgt->size() << std::endl;


	// Compute surface normals and curvature
	auto points_with_normals_src = addNormals(src);
	auto points_with_normals_tgt = addNormals(tgt);

	// Align
	pcl::IterativeClosestPointWithNormals<PointNormalType, PointNormalType> reg;
	reg.setTransformationEpsilon(1e-6);
	// Set the maximum distance between two correspondences (src<->tgt) to 10cm
	// Note: adjust this based on the size of your datasets
	reg.setMaxCorrespondenceDistance(100);

	reg.setInputSource(points_with_normals_src);
	reg.setInputTarget(points_with_normals_tgt);

	// Run the same optimization in a loop and visualize the results
	Eigen::Matrix4f prev, Ti;
	Ti = Eigen::Matrix4f::Identity();
	//  auto prev = Eigen::Matrix4f::Identity ();
	//  auto targetToSource = Eigen::Matrix4f::Identity ();
	auto reg_result = points_with_normals_src;
	reg.setMaximumIterations(2);
	for (int i = 0; i < 2; ++i)
	{
		PCL_INFO("Iteration Nr. %d.\n", i);

		// save cloud for visualization purpose
		points_with_normals_src = reg_result;

		// Estimate
		reg.setInputSource(points_with_normals_src);
		reg.align(*reg_result);

		//accumulate transformation between each Iteration
		Ti = static_cast<Eigen::Matrix4f>(reg.getFinalTransformation()) * Ti;

		//if the difference between this transformation and the previous one
		//is smaller than the threshold, refine the process by reducing
		//the maximal correspondence distance
		auto incremental_diff = fabs((reg.getLastIncrementalTransformation() - prev).sum());
		if (incremental_diff < reg.getTransformationEpsilon()) {
			reg.setMaxCorrespondenceDistance(reg.getMaxCorrespondenceDistance() * 0.95);
		}

		prev = reg.getLastIncrementalTransformation();
	}

	//add the source to the transformed target
	*output += *cloud_src;

	final_transform = Ti;
}


void saveTransformation() {
	std::cout <<"Saving extrinsics...." << endl;
}


void keyboardEventHandler(const pcl::visualization::KeyboardEvent& event, void*) {
	if (event.getKeySym() == "s" && event.keyDown()) {
		saveTransformation();
	}
}

int main(int argc, char** argv)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr src(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr src_to_dest(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr dest(new pcl::PointCloud<pcl::PointXYZRGB>);
	
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged_pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr final_merged_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr final_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr first_pc(new pcl::PointCloud<pcl::PointXYZRGB>);

	//read .xyz suffix file list
	string path = "./";
	vector<string> file_list_xyz = find_files_with_suffix(path, ".xyz");
	vector<Eigen::Matrix4f> transform_vec;
	transform_vec.clear();
	vector<Eigen::Matrix4f> transform_vec_nicp;
	transform_vec_nicp.clear();
	for(int pc_count=0;pc_count < file_list_xyz.size();pc_count++)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
		
		FILE* fp;
		string str = file_list_xyz[pc_count];
		char cstr[256];
		strcpy(cstr, str.c_str());	// or pass &s[0]
		fp = fopen(cstr, "r");

		int r, g, b;
		double x, y, z;
		int32_t  point_num = 0;

		pcl::PointXYZRGB point_xyzrgb;
		pcl::PointXYZ point_xyz;
		
		while (6 == fscanf(fp, "%lf %lf %lf %d %d %d\n", &x, &y, &z, &r, &g, &b))
		{
			//            if(abs(x) > 3000 || abs(y) > 3000 || abs(z) > 3000)
			//                continue;
			point_xyzrgb.x = x;
			point_xyzrgb.y = y;
			point_xyzrgb.z = z;
			point_xyzrgb.r = r;
			point_xyzrgb.g = g;
			point_xyzrgb.b = b;
			point_cloud->push_back(point_xyzrgb);

			point_xyz.x = x;
			point_xyz.y = y;
			point_xyz.z = z;

			tmp_cloud->push_back(point_xyzrgb);
			point_num++; 
		}
		std::cout <<"point_num: "<< point_num<<endl;
		fclose(fp);

		if (0 == pc_count)
		{
			dest = tmp_cloud;
			first_pc = dest;
			continue;
		}
		if ( pc_count != 0)
		{
			src = tmp_cloud;
		}
		std::cout << "origin input points size " << src->size() << std::endl;
		std::cout << "origin target points size " << dest->size() << std::endl;

		//https://github.com/leaveitout/pcl_point_cloud_merger/blob/master/src/main.cpp
		// Clean clouds of NaNs
		std::vector<int> mapping;
		pcl::removeNaNFromPointCloud(*src, *src, mapping);
		pcl::removeNaNFromPointCloud(*dest, *dest, mapping);
		//input_cloud = pre_PassThrough(input_cloud);
		src = pre_passThrough_x(src, -1145, 2000);
		src = pre_passThrough_y(src, 0.0, 1758);
		src = pre_passThrough_z(src, 0.0, 4500);

		dest = pre_passThrough_x(dest, -1145, 2000);
		dest = pre_passThrough_y(dest, 0.0, 1758);
		dest = pre_passThrough_z(dest, 0.0, 4500);

		std::cout << "PassThrough input points size " << src->size() << std::endl;
		std::cout << "PassThrough target points size " << dest->size() << std::endl;
		
		//point_cloud_src_filter = pre_statistical_outlier_removal(point_cloud_src_filter);
		//point_cloud_dest_filter = pre_statistical_outlier_removal(point_cloud_dest_filter);
		//std::cout << "statistical input points size " << point_cloud_src_filter->size() << std::endl;
		//std::cout << "statistical target points size " << point_cloud_dest_filter->size() << std::endl;

		// Filtering input scan to roughly 10% of original size to increase speed of registration.
		//src = pre_voxel_filter(src);
		//dest = pre_voxel_filter(dest);
		//std::cout << "voxel input points size " << src->size() << std::endl;
		//std::cout << "voxel target points size " << dest->size() << std::endl;

		// Initializing Normal Distributions Transform (NDT).
		pcl::NormalDistributionsTransform<pcl::PointXYZRGB, pcl::PointXYZRGB> ndt;

		// Setting scale dependent NDT parameters
		// Setting minimum transformation difference for termination condition.
		ndt.setTransformationEpsilon(0.01);
		// Setting maximum step size for More-Thuente line search.
		ndt.setStepSize(1);
		//Setting Resolution of NDT grid structure (VoxelGridCovariance).
		ndt.setResolution(10.0);
		// Setting max number of registration iterations.
		ndt.setMaximumIterations(35);
		// Setting point cloud to be aligned.
		ndt.setInputSource(src);
		// Setting point cloud to be aligned to.
		ndt.setInputTarget(dest);
		// Set initial alignment estimate found using robot odometry.
		Eigen::AngleAxisf init_rotation(0, Eigen::Vector3f::UnitZ());
		Eigen::Translation3f init_translation(0, 960, 0);
		Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix();
		// Calculating required rigid transform to align the input cloud to the target cloud.
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
		ndt.align(*output_cloud, init_guess);

		std::cout << "Normal Distributions Transform has converged:" << ndt.hasConverged()
			<< " score: " << ndt.getFitnessScore() << std::endl;

		//nicp
		final_transform = Eigen::Matrix4f::Identity();
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr nicp_final_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
		pairAlign(output_cloud, dest, nicp_final_pc, final_transform);//final_cloud  src add to dest
		std::cout << " transformation: RT = " << std::endl << final_transform * ndt.getFinalTransformation() << std::endl;

		
		transform_vec.push_back(final_transform * ndt.getFinalTransformation());
		//final_pc = nicp_final_pc;
		std::cout << "finish the "<< pc_count<<"th transformation";
		dest = src;
	}

	for (int fl_count = file_list_xyz.size() - 1; fl_count >= 0; fl_count--)
	{
		point_cloud->clear();
		FILE* fp;
		string str = file_list_xyz[fl_count];
		char cstr[256];
		strcpy(cstr, str.c_str());	// or pass &s[0]
		fp = fopen(cstr, "r");

		int r, g, b;
		double x, y, z;
		pcl::PointXYZRGB point_xyzrgb;
		while (6 == fscanf(fp, "%lf %lf %lf %d %d %d\n", &x, &y, &z, &r, &g, &b))
		{
			//            if(abs(x) > 3000 || abs(y) > 3000 || abs(z) > 3000)
			//                continue;
			point_xyzrgb.x = x;
			point_xyzrgb.y = y;
			point_xyzrgb.z = z;
			point_xyzrgb.r = r;
			point_xyzrgb.g = g;
			point_xyzrgb.b = b;
			point_cloud->push_back(point_xyzrgb);
		}
		fclose(fp);

		point_cloud = pre_passThrough_x(point_cloud, -1145, 2000);
		point_cloud = pre_passThrough_y(point_cloud, 0.0, 1758);
		point_cloud = pre_passThrough_z(point_cloud, 0.0, 4500);
		
		for (int i = fl_count - 1; i >= 0; i--)
		{
			pcl::transformPointCloud(*point_cloud, *point_cloud, transform_vec[i]);
		}

		std::cout << "finish the " << fl_count << "th point cloud transformation";
  
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
		tmp_cloud = point_cloud;
		*final_pc += *tmp_cloud;
	}
	// Saving transformed input cloud.
	pcl::io::savePCDFileASCII("car_transformed.pcd", *final_pc);

	
	// Initializing point cloud visualizer
	pcl::visualization::PCLVisualizer::Ptr
		viewer_final(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer_final->setBackgroundColor(0, 0, 0);

	 //Coloring and visualizing target cloud (red).
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB>
		target_color(first_pc, 255, 0, 0);
	viewer_final->addPointCloud<pcl::PointXYZRGB>(first_pc, target_color, "target cloud");
	viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
		1, "target cloud");


	// Coloring and visualizing src cloud (yellow).
	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB>
	//    src_color(src, 125, 125, 0);
	//viewer_final->addPointCloud<pcl::PointXYZRGB>(src, src_color, "src cloud");
	//viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
	//    1, "src cloud");

	
	// Coloring and visualizing src_dest cloud (green).
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB>
		src_dest_color(final_pc, 0, 255, 0);
	viewer_final->addPointCloud<pcl::PointXYZRGB>(final_pc, src_dest_color, "final_pc cloud");
	viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
		1, "final_pc cloud");

	// Starting visualizer
	viewer_final->addCoordinateSystem(1.0, "global");
	viewer_final->initCameraParameters();

	// Wait until visualizer window is closed.
	while (!viewer_final->wasStopped())
	{
		viewer_final->spinOnce(100);
		std::this_thread::sleep_for(100ms);
	}

	return (0);
}