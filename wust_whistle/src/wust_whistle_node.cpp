#include <ros/ros.h>
#include <wust_msgs/WhistleResult.h>
#include <wust_msgs/EnemyInfo.h>
#include <tf/transform_listener.h>
#include <wust_whistle/WhistleParamConfig.h>
#include <dynamic_reconfigure/server.h>
#include <mutex>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/PointStamped.h>


class WhistleNode
{
public:
    WhistleNode(ros::NodeHandle* nh, ros::NodeHandle* pnh)
    {
        nh->param<std::string>("tf_prefix", tf_prefix_, "");
        nh->param<std::string>("robot_name", robot_name_, "rm");
        nh->param<std::string>("friend_name", friend_name_, "nvidia");

        whistle_sub_ = nh->subscribe("/whistle_result", 2, &WhistleNode::WhistleResultCallback, this);

        enemy_info_pub_ = nh->advertise<wust_msgs::EnemyInfo>("whistle_enemy_info", 1);

        wust_whistle::WhistleParamConfig config;
        server_.getConfigDefault(config);
        distance_thresh_ = config.distance_thresh;
        clean_time_ = config.clean_time;
        server_.setCallback(std::bind(&WhistleNode::ConfigCallback, this, std::placeholders::_1, std::placeholders::_2));

        marker_pub_ = nh->advertise<visualization_msgs::Marker>("visualization_marker", 1);
        rviz_point_sub_ = nh->subscribe("clicked_point", 1, &WhistleNode::ClickedPointCallback, this);;

        loop_timer_ = pnh->createTimer(ros::Duration(0.05), &WhistleNode::CalculationLoop, this);

        ROS_INFO("Initialize whistle node successful.");
    }

private:

    std::string tf_prefix_;
    std::string robot_name_;
    std::string friend_name_;

    ros::Publisher marker_pub_;
    ros::Subscriber rviz_point_sub_;

    ros::Subscriber whistle_sub_;

    tf::TransformListener listener_;

    double distance_thresh_;
    double clean_time_;

    std::mutex whistle_result_buffer_mux;
    std::vector<wust_msgs::WhistleResult> whistle_result_buffer;

    ros::Publisher enemy_info_pub_;

    ros::Timer loop_timer_;

    dynamic_reconfigure::Server<wust_whistle::WhistleParamConfig> server_;

    void ConfigCallback(wust_whistle::WhistleParamConfig &config, uint32_t level)
    {
        distance_thresh_ = config.distance_thresh;
        clean_time_ = config.clean_time;
    }

    void ClickedPointCallback(const geometry_msgs::PointStamped::ConstPtr& msg)
    {
        std::lock_guard<std::mutex> lock(whistle_result_buffer_mux);
        wust_msgs::WhistleResult fake_result;
        wust_msgs::CarPosition fake_car_position;
        fake_car_position.x = msg->point.x;
        fake_car_position.y = msg->point.y;
        fake_result.header = msg->header;
        fake_result.car_positions.push_back(fake_car_position);
        whistle_result_buffer.push_back(fake_result);
    }

    void CalculationLoop(const ros::TimerEvent& event)
    {
        // 获取己方位置
        tf::StampedTransform robot_position;
        tf::StampedTransform friend_position;
        try
        {
            listener_.lookupTransform("map", robot_name_ + "/va_base_link", ros::Time(0), robot_position);
        }
        catch (tf::TransformException ex)
        {
            // 如果读不到己方位置，返回
            ROS_ERROR("Cannot read self position: %s", ex.what());
            return;
        }
        try
        {
            listener_.lookupTransform("map", friend_name_ + "/va_base_link", ros::Time(0), friend_position);
        }
        catch (tf::TransformException ex)
        {
            ROS_DEBUG("Cannot read friend position, using self position: %s", ex.what());
            friend_position = robot_position;
        }

        std::vector<wust_msgs::CarPosition> enemy_car_positions;

        {
            // 删除超时的结果
            std::lock_guard<std::mutex> lock(whistle_result_buffer_mux);
            whistle_result_buffer.erase(std::remove_if(whistle_result_buffer.begin(), whistle_result_buffer.end(), [this](wust_msgs::WhistleResult result){
                return (ros::Time::now().toSec() - result.header.stamp.toSec() > clean_time_);
            }), whistle_result_buffer.end());

            // 敌我判断
            for(const auto& result : whistle_result_buffer)
            {
                for(const auto& car_position : result.car_positions)
                {
                    double distance_to_self = std::sqrt(std::pow(robot_position.getOrigin().getX() - car_position.x, 2) + std::pow(robot_position.getOrigin().getY() - car_position.y, 2));
                    double distance_to_friend = std::sqrt(std::pow(friend_position.getOrigin().getX() - car_position.x, 2) + std::pow(friend_position.getOrigin().getY() - car_position.y, 2));

                    if(distance_to_self > distance_thresh_ && distance_to_friend > distance_thresh_)
                    {
                        enemy_car_positions.push_back(car_position);
                    }
                }
            }
        }

        wust_msgs::EnemyInfo enemy_msg;
        if(enemy_car_positions.empty())
        {
            enemy_msg.detected = false;
        }
        else
        {
            std::sort(enemy_car_positions.begin(), enemy_car_positions.end(), [&](wust_msgs::CarPosition a, wust_msgs::CarPosition b) {
                double distance_to_self_a = std::sqrt(std::pow(robot_position.getOrigin().getX() - a.x, 2) + std::pow(robot_position.getOrigin().getY() - a.y, 2));
                double distance_to_self_b = std::sqrt(std::pow(robot_position.getOrigin().getX() - b.x, 2) + std::pow(robot_position.getOrigin().getY() - b.y, 2));
                return distance_to_self_a < distance_to_self_b;
            });

            wust_msgs::CarPosition nearest_car = enemy_car_positions[0];

            enemy_msg.detected = true;
            enemy_msg.enemy_pos.pose.position.x = nearest_car.x;
            enemy_msg.enemy_pos.pose.position.y = nearest_car.y;
        }
        
        // 发布enemy_msg消息
        enemy_info_pub_.publish(enemy_msg);

        // 可视化
        visualization_msgs::Marker marker;
        for(int i = 0; i < enemy_car_positions.size(); ++i)
        {
            marker.header.frame_id = "map";
            marker.header.stamp = ros::Time::now();
            marker.ns = "whistle_result";
            marker.id = i;
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.action = visualization_msgs::Marker::ADD;

            marker.pose.position.x = enemy_car_positions[i].x;
            marker.pose.position.y = enemy_car_positions[i].y;
            marker.pose.position.z = 0.0;
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;

            marker.scale.x = 0.6;
            marker.scale.y = 0.6;
            marker.scale.z = 0.6;

            marker.color.r = 1.0f;
            marker.color.g = 0.0f;
            marker.color.b = 0.0f;
            marker.color.a = 1.0;

            marker.lifetime = ros::Duration(clean_time_);
            
            marker_pub_.publish(marker);
        }
        
    }

    void WhistleResultCallback(const wust_msgs::WhistleResult::ConstPtr& msg)
    {
        std::lock_guard<std::mutex> lock(whistle_result_buffer_mux);
        whistle_result_buffer.push_back(wust_msgs::WhistleResult(*msg));
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "wust_whistle");

    ros::NodeHandle nh("");
    ros::NodeHandle pnh("~");

    WhistleNode whistle_node(&nh, &pnh);

    ros::spin();
}