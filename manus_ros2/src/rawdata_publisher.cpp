#include <memory>

#include "rclcpp/rclcpp.hpp"

#include "SDKMinimalClient.hpp"

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SDKMinimalClient>());
    rclcpp::shutdown();
    return 0;
}
