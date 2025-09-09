// SPDX-License-Identifier: BSD-3-Clause
// Camera-LiDAR extrinsic calibration node (ROS 2)
// ported from ROS 1 implementation to rclcpp

#include <algorithm>
#include <random>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/image.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <cv_bridge/cv_bridge.h>

#include <Eigen/Dense>

#include "calibration_error_term.h"
#include "ceres/ceres.h"
#include "glog/logging.h"


using PointCloud2   = sensor_msgs::msg::PointCloud2;
using ImageMsg      = sensor_msgs::msg::Image;
using SyncPolicy    = message_filters::sync_policies::ApproximateTime<PointCloud2, ImageMsg>;

class CamLidarCalib : public rclcpp::Node
{
public:
  CamLidarCalib() : Node("cam_lidar_calib_node")
  {
    dx_                      = declare_and_get<double>("dx", 0.075);
    dy_                      = declare_and_get<double>("dy", 0.075);
    checkerboard_rows_       = declare_and_get<int   >("checkerboard_rows", 9);
    checkerboard_cols_       = declare_and_get<int   >("checkerboard_cols", 6);
    min_points_on_plane_     = declare_and_get<int   >("min_points_on_plane", 450);
    num_views_               = declare_and_get<int   >("num_views", 10);
    no_of_initializations_   = declare_and_get<int   >("no_of_initializations", 5);

    camera_in_topic_         = declare_and_get<std::string>("camera_in_topic", "/camera/image_raw");
    lidar_in_topic_          = declare_and_get<std::string>("lidar_in_topic",  "/points_raw");

    cam_config_file_path_    = declare_and_get<std::string>("cam_config_file_path", "config.yaml");
    result_str_              = declare_and_get<std::string>("result_file", "result/C_T_L.txt");
    result_rpy_              = declare_and_get<std::string>("result_rpy_file", "result/rpy_txyz.txt");
    initializations_file_    = declare_and_get<std::string>("initializations_file", "result/initializations.txt");

    x_min_          = declare_and_get<double>("x_min", -10.0);
    x_max_          = declare_and_get<double>("x_max",  10.0);
    y_min_          = declare_and_get<double>("y_min", -10.0);
    y_max_          = declare_and_get<double>("y_max",  10.0);
    z_min_          = declare_and_get<double>("z_min", -2.0);
    z_max_          = declare_and_get<double>("z_max",  2.0);
    ransac_threshold_ = declare_and_get<double>("ransac_threshold", 0.01);

    projection_matrix_ = cv::Mat::zeros(3,3,CV_64F);
    dist_coeff_        = cv::Mat::zeros(5,1,CV_64F);
    readCameraParams(cam_config_file_path_, image_height_, image_width_, dist_coeff_, projection_matrix_);

    cloud_pub_ = this->create_publisher<PointCloud2>("points_out", rclcpp::SensorDataQoS());

    cloud_sub_  = std::make_shared<message_filters::Subscriber<PointCloud2>>(this, lidar_in_topic_, rclcpp::SensorDataQoS().get_rmw_qos_profile());
    image_sub_  = std::make_shared<message_filters::Subscriber<ImageMsg>>(this, camera_in_topic_, rclcpp::SensorDataQoS().get_rmw_qos_profile());
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), *cloud_sub_, *image_sub_);
    sync_->registerCallback(std::bind(&CamLidarCalib::callback, this, std::placeholders::_1, std::placeholders::_2));

    // generate 3D coordinates of checkerboard
    for (int i = 0; i < checkerboard_rows_; ++i)
      for (int j = 0; j < checkerboard_cols_; ++j)
        object_points_.emplace_back(cv::Point3f(i * dx_, j * dy_, 0.0));

    boardDetectedInCam_ = false;
    tvec_ = cv::Mat::zeros(3, 1, CV_64F);
    rvec_ = cv::Mat::zeros(3, 1, CV_64F);
    C_R_W_ = cv::Mat::eye(3, 3, CV_64F);
    c_R_w_ = Eigen::Matrix3d::Identity();

    RCLCPP_INFO(this->get_logger(), "Cam-LiDAR calibration node initialised");
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

  // ------------------------------ read camera parameters -----------------------
  void readCameraParams(const std::string &cam_config_file_path,
                        int &image_height,
                        int &image_width,
                        cv::Mat &D,
                        cv::Mat &K)
  {
    cv::FileStorage fs(cam_config_file_path, cv::FileStorage::READ);
    if(!fs.isOpened())
    {
      RCLCPP_ERROR(this->get_logger(), "Cannot open camera config: %s", cam_config_file_path.c_str());
      throw std::runtime_error("Camera config read error");
    }
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

  void callback(const PointCloud2::ConstSharedPtr &cloud_msg, const ImageMsg::ConstSharedPtr &image_msg)
  {
    imageHandler(image_msg);
    cloudHandler(cloud_msg);
    runSolver();
    }

  void cloudHandler(const PointCloud2::ConstSharedPtr &cloud_msg)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *in_cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_x(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_y(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_z(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_filtered(new pcl::PointCloud<pcl::PointXYZ>);

    // pass through
    pcl::PassThrough<pcl::PointXYZ> pass_x;
    pass_x.setInputCloud(in_cloud);
    pass_x.setFilterFieldName("x");
    pass_x.setFilterLimits(x_min_, x_max_);
    pass_x.filter(*cloud_filtered_x);

    pcl::PassThrough<pcl::PointXYZ> pass_y;
    pass_y.setInputCloud(cloud_filtered_x);
    pass_y.setFilterFieldName("y");
    pass_y.setFilterLimits(y_min_, y_max_);
    pass_y.filter(*cloud_filtered_y);

    pcl::PassThrough<pcl::PointXYZ> pass_z;
    pass_z.setInputCloud(cloud_filtered_y);
    pass_z.setFilterFieldName("z");
    pass_z.setFilterLimits(z_min_, z_max_);
    pass_z.filter(*cloud_filtered_z);

    // plane extraction
    pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p(new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud_filtered_z));
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_p);
    ransac.setDistanceThreshold(ransac_threshold_);
    ransac.computeModel();
    std::vector<int> inliers;
    ransac.getInliers(inliers);
    pcl::copyPointCloud<pcl::PointXYZ>(*cloud_filtered_z, inliers, *plane_cloud);

    // remove outliers
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(plane);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1);
    sor.filter(*plane_filtered);

    // save lidar points
    lidar_points_.clear();
    for(const auto &pt : plane_filtered->points)
      lidar_points_.emplace_back(Eigen::Vector3d(pt.x, pt.y, pt.z));

    RCLCPP_WARN_STREAM(this->get_logger(), "No of planar_pts: " << plane_filtered->points.size());

    sensor_msgs::msg::PointCloud2 out_cloud;
    pcl::toROSMsg(*plane_filtered, out_cloud);
    out_cloud.header.frame_id = cloud_msg->header.frame_id;
    out_cloud.header.stamp = cloud_msg->header.stamp;
    cloud_pub_->publish(out_cloud);
    }

  void imageHandler(const ImageMsg::ConstSharedPtr &image_msg)
  {
    try
    {
      auto cv_ptr = cv_bridge::toCvCopy(image_msg, "bgr8");
      image_in_ = cv_ptr->image;

      boardDetectedInCam_ = cv::findChessboardCorners(image_in_, cv::Size(checkerboard_cols_, checkerboard_rows_), image_points_,
                               cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
      cv::drawChessboardCorners(image_in_, cv::Size(checkerboard_cols_, checkerboard_rows_), image_points_, boardDetectedInCam_);

      if(image_points_.size() == object_points_.size())
      {
        cv::solvePnP(object_points_, image_points_, projection_matrix_, dist_coeff_, rvec_, tvec_, false, cv::SOLVEPNP_ITERATIVE);
        projected_points_.clear();
        cv::projectPoints(object_points_, rvec_, tvec_, projection_matrix_, dist_coeff_, projected_points_, cv::noArray());
        for(const auto &pp : projected_points_)
        cv::circle(image_in_, pp, 16, cv::Scalar(0,255,0), 10, cv::LINE_AA);

        cv::Rodrigues(rvec_, C_R_W_);
        cv::cv2eigen(C_R_W_, c_R_w_);
        c_t_w_ = Eigen::Vector3d(tvec_.at<double>(0), tvec_.at<double>(1), tvec_.at<double>(2));

        r3_ = c_R_w_.block<3,1>(0,2);
        Nc_ = (r3_.dot(c_t_w_)) * r3_;
      }

      cv::Mat image_resized;
      cv::resize(image_in_, image_resized, cv::Size(), 0.25, 0.25);
      cv::imshow("cam_lidar_calib/view", image_resized);
      cv::waitKey(1);
    }
    catch(const cv_bridge::Exception &e)
    {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

  void addGaussianNoise(Eigen::Matrix4d &transformation)
  {
    std::vector<double> data_rot(3, 0.0);
    const double mean_rot = 0.0;
    std::default_random_engine gen_rot(static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count()));
    std::normal_distribution<double> dist_rot(mean_rot, 90);
    for(auto &v : data_rot) v += dist_rot(gen_rot);

    // Eigen::Matrix3d m = Eigen::AngleAxisd(data_rot[0]*M_PI/180, Eigen::Vector3d::UnitX()) *
    //                     Eigen::AngleAxisd(data_rot[1]*M_PI/180, Eigen::Vector3d::UnitY()) *
    //                     Eigen::AngleAxisd(data_rot[2]*M_PI/180, Eigen::Vector3d::UnitZ());

    Eigen::Quaterniond q = 
      Eigen::AngleAxisd(data_rot[0] * M_PI / 180.0, Eigen::Vector3d::UnitX()) *
      Eigen::AngleAxisd(data_rot[1] * M_PI / 180.0, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(data_rot[2] * M_PI / 180.0, Eigen::Vector3d::UnitZ());

    Eigen::Matrix3d m = q.toRotationMatrix();

    std::vector<double> data_trans(3,0.0);
    std::default_random_engine gen_trans(static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count()));
    std::normal_distribution<double> dist_trans(0.0, 0.5);
    for(auto &v : data_trans) v += dist_trans(gen_trans);

    Eigen::Vector3d trans(data_trans[0], data_trans[1], data_trans[2]);

    Eigen::Matrix4d trans_noise = Eigen::Matrix4d::Identity();
    trans_noise.block(0,0,3,3) = m;
    trans_noise.block(0,3,3,1) = trans;
    transformation = transformation * trans_noise;
    }

  void runSolver()
  {
    if(lidar_points_.size() > static_cast<size_t>(min_points_on_plane_) && boardDetectedInCam_)
    {
      if(r3_.dot(r3_old_) < 0.90)
      {
        r3_old_ = r3_;
        all_normals_.push_back(Nc_);
        all_lidar_points_.push_back(lidar_points_);
        RCLCPP_INFO_STREAM(this->get_logger(), "Recording view number: " << all_normals_.size());

        if(all_normals_.size() >= static_cast<size_t>(num_views_))
        {
          RCLCPP_INFO(this->get_logger(), "Starting optimization...");
          std::ofstream init_file(initializations_file_);
          for(int counter = 0; counter < no_of_initializations_; ++counter)
          {
            // initialize
            Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
            addGaussianNoise(transformation);
            Eigen::Matrix3d Rotn = transformation.block(0,0,3,3);
            Eigen::Vector3d axis_angle;
            ceres::RotationMatrixToAngleAxis(Rotn.data(), axis_angle.data());
            Eigen::Vector3d Translation = transformation.block(0,3,3,1);
            Eigen::Vector3d rpy_init = Rotn.eulerAngles(0,1,2) * 180 / M_PI;
            Eigen::Vector3d tran_init = Translation;

            Eigen::VectorXd R_t(6);
            R_t << axis_angle(0), axis_angle(1), axis_angle(2), Translation(0), Translation(1), Translation(2);

            ceres::Problem problem;
            problem.AddParameterBlock(R_t.data(), 6);
            for(size_t i = 0; i < all_normals_.size(); ++i)
            {
              Eigen::Vector3d normal_i = all_normals_[i];
              const auto &lidar_pts = all_lidar_points_[i];
              for(const auto &lidar_pt : lidar_pts)
              {
                auto *cost_function = new ceres::AutoDiffCostFunction<CalibrationErrorTerm,1,6>(new CalibrationErrorTerm(lidar_pt, normal_i));
                problem.AddResidualBlock(cost_function, nullptr, R_t.data());
              }
            }

            ceres::Solver::Options options;
            options.max_num_iterations = 200;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            options.minimizer_progress_to_stdout = false;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            ceres::AngleAxisToRotationMatrix(R_t.data(), Rotn.data());
            Eigen::MatrixXd C_T_L(3,4);
            C_T_L.block(0,0,3,3) = Rotn;
            C_T_L.block(0,3,3,1) = Eigen::Vector3d(R_t[3],R_t[4],R_t[5]);

            std::ofstream results(result_str_);
            results << C_T_L;
            results.close();

            std::ofstream results_rpy(result_rpy_);
            results_rpy << Rotn.eulerAngles(0,1,2)*180/M_PI << "\n" << C_T_L.block(0,3,3,1);
            results_rpy.close();

            // log & save initial values
            init_file << rpy_init.transpose() << "," << tran_init.transpose() << "\n";
            init_file << (Rotn.eulerAngles(0,1,2)*180/M_PI).transpose() << "," << R_t.segment<3>(3).transpose() << "\n";
            RCLCPP_INFO(this->get_logger(), "Initialization %d finished", counter);
      }
          init_file.close();
          rclcpp::shutdown();
        }
      }
      else
      {
        RCLCPP_WARN(this->get_logger(), "Not enough rotation, view not recorded");
      }
    }
    else
    {
      if(!boardDetectedInCam_)
        RCLCPP_WARN(this->get_logger(), "Checker board not detected in image");
      else
        RCLCPP_WARN_STREAM(this->get_logger(), "Checker Board Detected: " << boardDetectedInCam_
                           << "  LiDAR pts: " << lidar_points_.size());
    }
  }

  std::shared_ptr<message_filters::Subscriber<PointCloud2>> cloud_sub_;
  std::shared_ptr<message_filters::Subscriber<ImageMsg>>    image_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
  rclcpp::Publisher<PointCloud2>::SharedPtr cloud_pub_;

  cv::Mat image_in_;
  cv::Mat projection_matrix_;
  cv::Mat dist_coeff_;
  std::vector<cv::Point2f> image_points_;
  std::vector<cv::Point3f> object_points_;
  std::vector<cv::Point2f> projected_points_;
  bool boardDetectedInCam_;

  cv::Mat tvec_, rvec_, C_R_W_;
  Eigen::Matrix3d c_R_w_;
  Eigen::Vector3d c_t_w_;
  Eigen::Vector3d r3_, r3_old_;
  Eigen::Vector3d Nc_;

  std::vector<Eigen::Vector3d> lidar_points_;
  std::vector<std::vector<Eigen::Vector3d>> all_lidar_points_;
  std::vector<Eigen::Vector3d> all_normals_;

  double dx_, dy_;
  int checkerboard_rows_, checkerboard_cols_;
  int min_points_on_plane_;
  int num_views_;
  int no_of_initializations_;
  int image_width_{0}, image_height_{0};
  double x_min_, x_max_, y_min_, y_max_, z_min_, z_max_;
  double ransac_threshold_;

  std::string camera_in_topic_;
  std::string lidar_in_topic_;
  std::string cam_config_file_path_;
  std::string result_str_;
  std::string result_rpy_;
  std::string initializations_file_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CamLidarCalib>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}