// SPDX-License-Identifier: BSD-3-Clause
// Camera-LiDAR projection node (ROS 2)

#include <memory>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"

#include <tf2_ros/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/ransac.h>

using PointCloud2 = sensor_msgs::msg::PointCloud2;
using ImageMsg    = sensor_msgs::msg::Image;
using SyncPolicy  = message_filters::sync_policies::ApproximateTime<PointCloud2, ImageMsg>;

class LidarImageProjection : public rclcpp::Node
{
public:
  LidarImageProjection() : Node("cam_lidar_proj_node")
  {
    camera_in_topic_ = declare_and_get<std::string>("camera_in_topic", "/camera/image_raw");
    lidar_in_topic_  = declare_and_get<std::string>("lidar_in_topic",  "/points_raw");
    dist_cut_off_    = declare_and_get<int>("dist_cut_off", 50);
    camera_name_     = declare_and_get<std::string>("camera_name", "camera");
    result_str_      = declare_and_get<std::string>("result_file", "result/C_T_L.txt");
    project_only_plane_ = declare_and_get<bool>("project_only_plane", false);
    cam_config_file_path_ = declare_and_get<std::string>("cam_config_file_path", "config.yaml");

    projection_matrix_ = cv::Mat::zeros(3,3,CV_64F);
    distCoeff_         = cv::Mat::zeros(5,1,CV_64F);
    readCameraParams(cam_config_file_path_, image_height_, image_width_, distCoeff_, projection_matrix_);

    readCalibrationFile();

    std::string lidarOutTopic  = lidar_in_topic_ + "/projected_cloud";
    std::string imageOutTopic  = camera_in_topic_ + "/projected_image";
    cloud_pub_ = this->create_publisher<PointCloud2>(lidarOutTopic, rclcpp::SensorDataQoS());
    image_pub_ = this->create_publisher<ImageMsg>(imageOutTopic, 10);

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    cloud_sub_ = std::make_shared<message_filters::Subscriber<PointCloud2>>(this, lidar_in_topic_, rclcpp::SensorDataQoS().get_rmw_qos_profile());
    image_sub_ = std::make_shared<message_filters::Subscriber<ImageMsg>>(this, camera_in_topic_, rclcpp::SensorDataQoS().get_rmw_qos_profile());
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), *cloud_sub_, *image_sub_);
    sync_->registerCallback(std::bind(&LidarImageProjection::callback, this, std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "cam_lidar_proj_node initialised");
  }

private:
  template<typename T>
  T declare_and_get(const std::string &name, const T &default_value)
  {
    this->declare_parameter<T>(name, default_value);
    T value = default_value;
    this->get_parameter(name, value);
    return value;
  }

  void readCameraParams(const std::string &file_path,
                        int &image_height,
                        int &image_width,
                        cv::Mat &D,
                        cv::Mat &K)
  {
    cv::FileStorage fs(file_path, cv::FileStorage::READ);
    if(!fs.isOpened())
      throw std::runtime_error("Cannot open camera config file: " + file_path);
    fs["image_height"] >> image_height;
    fs["image_width"]  >> image_width;
    fs["k1"] >> D.at<double>(0);
    fs["k2"] >> D.at<double>(1);
    fs["p1"] >> D.at<double>(2);
    fs["p2"] >> D.at<double>(3);
    fs["k3"] >> D.at<double>(4);
    fs["fx"] >> K.at<double>(0,0);
    fs["fy"] >> K.at<double>(1,1);
    fs["cx"] >> K.at<double>(0,2);
    fs["cy"] >> K.at<double>(1,2);
  }

  void readCalibrationFile()
  {
    C_T_L_ = Eigen::Matrix4d::Identity();
    std::ifstream ifs(result_str_);
    if(!ifs.is_open())
      throw std::runtime_error("Cannot open calibration result file: " + result_str_);
    for(int i=0;i<3;i++)
      for(int j=0;j<4;j++)
        ifs >> C_T_L_(i,j);

    L_T_C_ = C_T_L_.inverse();

    C_R_L_ = C_T_L_.block(0,0,3,3);
    C_t_L_ = C_T_L_.block(0,3,3,1);
    L_R_C_ = L_T_C_.block(0,0,3,3);
    L_t_C_ = L_T_C_.block(0,3,3,1);

    C_R_L_quat_ = Eigen::Quaterniond(C_R_L_);
    L_R_C_quat_ = Eigen::Quaterniond(L_R_C_);

    cv::eigen2cv(C_R_L_, c_R_l_);
    cv::Rodrigues(c_R_l_, rvec_);
    cv::eigen2cv(C_t_L_, tvec_);
  }

  void publishTransform(const std::string &lidar_frame)
  {
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = this->now();
    t.header.frame_id = lidar_frame;
    t.child_frame_id  = camera_name_;
    t.transform.translation.x = L_t_C_(0);
    t.transform.translation.y = L_t_C_(1);
    t.transform.translation.z = L_t_C_(2);
    tf2::Quaternion q;
    q.setX(L_R_C_quat_.x());
    q.setY(L_R_C_quat_.y());
    q.setZ(L_R_C_quat_.z());
    q.setW(L_R_C_quat_.w());
    t.transform.rotation = tf2::toMsg(q);
    tf_broadcaster_->sendTransform(t);
  }

  static cv::Vec3b atf(const cv::Mat &rgb, const cv::Point2d &xy_f)
  {
    cv::Vec3i color_i{0,0,0};
    int x = static_cast<int>(xy_f.x);
    int y = static_cast<int>(xy_f.y);
    for(int row=0;row<=1;row++)
      for(int col=0;col<=1;col++)
        if((x+col)<rgb.cols && (y+row)<rgb.rows)
          for(int i=0;i<3;i++)
            color_i[i] += rgb.at<cv::Vec3b>(cv::Point(x+col,y+row))[i];
    return cv::Vec3b(color_i[0]/4, color_i[1]/4, color_i[2]/4);
  }

  void callback(const PointCloud2::ConstSharedPtr &cloud_msg,
                const ImageMsg::ConstSharedPtr &image_msg)
  {
    std::string lidar_frameId = cloud_msg->header.frame_id;
    publishTransform(lidar_frameId);

    objectPoints_L_.clear();
    objectPoints_C_.clear();
    imagePoints_.clear();

    image_in_ = cv_bridge::toCvCopy(image_msg, "bgr8")->image;

    double fov_x = 2*atan2(image_width_, 2*projection_matrix_.at<double>(0,0))*180/CV_PI;
    double fov_y = 2*atan2(image_height_,2*projection_matrix_.at<double>(1,1))*180/CV_PI;

    double max_range = -std::numeric_limits<double>::infinity();
    double min_range = std::numeric_limits<double>::infinity();

    pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if(project_only_plane_)
    {
      in_cloud = planeFilter(cloud_msg);
      for(const auto &pt : in_cloud->points)
        objectPoints_L_.emplace_back(pt.x, pt.y, pt.z);
      cv::projectPoints(objectPoints_L_, rvec_, tvec_, projection_matrix_, distCoeff_, imagePoints_, cv::noArray());
    }
    else
    {
      pcl::PCLPointCloud2 cloud_in;
      pcl_conversions::toPCL(*cloud_msg, cloud_in);
      pcl::fromPCLPointCloud2(cloud_in, *in_cloud);

      for(const auto &pt_in : in_cloud->points)
      {
        if(pt_in.x < 0 || pt_in.x > dist_cut_off_) continue;
        Eigen::Vector4d pt_L(pt_in.x, pt_in.y, pt_in.z, 1.0);
        Eigen::Vector3d pt_C = C_T_L_.block<3,4>(0,0) * pt_L;
        double X = pt_C(0), Y = pt_C(1), Z = pt_C(2);
        double Xangle = atan2(X, Z)*180/CV_PI;
        double Yangle = atan2(Y, Z)*180/CV_PI;
        if(Xangle < -fov_x/2 || Xangle > fov_x/2) continue;
        if(Yangle < -fov_y/2 || Yangle > fov_y/2) continue;
        double range = sqrt(X*X+Y*Y+Z*Z);
        max_range = std::max(max_range, range);
        min_range = std::min(min_range, range);
        objectPoints_L_.emplace_back(pt_L(0), pt_L(1), pt_L(2));
        objectPoints_C_.emplace_back(X, Y, Z);
      }
      cv::projectPoints(objectPoints_L_, rvec_, tvec_, projection_matrix_, distCoeff_, imagePoints_, cv::noArray());
    }

    colorPointCloud();

    PointCloud2 out_cloud_ros;
    pcl::toROSMsg(out_cloud_pcl_, out_cloud_ros);
    out_cloud_ros.header = cloud_msg->header;
    cloud_pub_->publish(out_cloud_ros);

    colorLidarPointsOnImage(min_range, max_range);
    auto img_msg_out = cv_bridge::CvImage(image_msg->header, "bgr8", image_in_).toImageMsg();
    image_pub_->publish(*img_msg_out);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr planeFilter(const PointCloud2::ConstSharedPtr &cloud_msg)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *in_cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_x(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_y(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_filtered(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(in_cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(0.0, 10.0);
    pass.filter(*cloud_filtered_x);

    pass.setInputCloud(cloud_filtered_x);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-3.0, 3.0);
    pass.filter(*cloud_filtered_y);

    pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p(new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud_filtered_y));
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_p);
    ransac.setDistanceThreshold(0.01);
    ransac.computeModel();
    std::vector<int> inliers;
    ransac.getInliers(inliers);
    pcl::copyPointCloud(*cloud_filtered_y, inliers, *plane);

    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(plane);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1);
    sor.filter(*plane_filtered);
    return plane_filtered;
    }

  void colorPointCloud()
  {
    out_cloud_pcl_.clear();
    out_cloud_pcl_.resize(objectPoints_L_.size());
    for(size_t i=0;i<objectPoints_L_.size();++i)
    {
      cv::Vec3b rgb = atf(image_in_, imagePoints_[i]);
      pcl::PointXYZRGB pt(rgb[2], rgb[1], rgb[0]);
      pt.x = objectPoints_L_[i].x;
      pt.y = objectPoints_L_[i].y;
      pt.z = objectPoints_L_[i].z;
      out_cloud_pcl_.push_back(pt);
    }
  }

  void colorLidarPointsOnImage(double min_range, double max_range)
  {
    for(size_t i=0;i<imagePoints_.size();++i)
    {
      double X = objectPoints_C_[i].x;
      double Y = objectPoints_C_[i].y;
      double Z = objectPoints_C_[i].z;
      double range = sqrt(X*X+Y*Y+Z*Z);
      double red_field   = 255*(range - min_range)/(max_range - min_range);
      double green_field = 255*(max_range - range)/(max_range - min_range);
      cv::circle(image_in_, imagePoints_[i], 2, CV_RGB(red_field, green_field, 0), -1, 1, 0);
    }
  }

  std::string camera_in_topic_;
  std::string lidar_in_topic_;
  int dist_cut_off_;
  std::string camera_name_;
  std::string result_str_;
  bool project_only_plane_;
  std::string cam_config_file_path_;
  int image_width_{0}, image_height_{0};

  Eigen::Matrix4d C_T_L_;
  Eigen::Matrix4d L_T_C_;
  Eigen::Matrix3d C_R_L_, L_R_C_;
  Eigen::Vector3d C_t_L_, L_t_C_;
  Eigen::Quaterniond C_R_L_quat_, L_R_C_quat_;
  cv::Mat c_R_l_;
  cv::Mat rvec_;
  cv::Mat tvec_;

  cv::Mat projection_matrix_;
  cv::Mat distCoeff_;

  std::shared_ptr<message_filters::Subscriber<PointCloud2>> cloud_sub_;
  std::shared_ptr<message_filters::Subscriber<ImageMsg>> image_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
  rclcpp::Publisher<PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::Publisher<ImageMsg>::SharedPtr image_pub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  cv::Mat image_in_;
  std::vector<cv::Point3d> objectPoints_L_, objectPoints_C_;
  std::vector<cv::Point2d> imagePoints_;
  pcl::PointCloud<pcl::PointXYZRGB> out_cloud_pcl_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LidarImageProjection>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}