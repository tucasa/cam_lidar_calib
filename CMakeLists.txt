cmake_minimum_required(VERSION 3.5)
project(cam_lidar_calib VERSION 0.0.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(ament_index_cpp REQUIRED)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)
add_definitions(${PCL_DEFINITIONS})

add_executable(cam_lidar_calib_node src/cam_lidar_calib_node.cpp)
ament_target_dependencies(cam_lidar_calib_node
  rclcpp sensor_msgs message_filters cv_bridge tf2 tf2_ros tf2_eigen tf2_geometry_msgs
  pcl_conversions PCL OpenCV ament_index_cpp
)
target_link_libraries(cam_lidar_calib_node ${CERES_LIBRARIES} ${PCL_LIBRARIES})

add_executable(cam_lidar_proj_node src/cam_lidar_proj_node.cpp)
ament_target_dependencies(cam_lidar_proj_node
  rclcpp sensor_msgs message_filters cv_bridge tf2 tf2_ros tf2_eigen tf2_geometry_msgs
  pcl_conversions PCL OpenCV
)
target_link_libraries(cam_lidar_proj_node ${PCL_LIBRARIES})

install(TARGETS
  cam_lidar_calib_node
  cam_lidar_proj_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config include result
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
