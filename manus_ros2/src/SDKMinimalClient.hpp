#ifndef _SDK_MINIMAL_CLIENT_HPP_
#define _SDK_MINIMAL_CLIENT_HPP_

// Set up a Doxygen group.
/** @addtogroup SDKMinimalClient
 *  @{
 */

#include "ClientPlatformSpecific.hpp"
#include "ManusSDK.h"
#include <mutex>
#include <vector>
#include <memory>
#include <deque>

#include "rclcpp/rclcpp.hpp"
#include "manus_ros2_msgs/msg/manus_node_hierarchy.hpp"
#include "manus_ros2_msgs/msg/manus_node_poses.hpp"

/// @brief The type of connection to core.
enum class ConnectionType : int
{
    ConnectionType_Invalid = 0,
    ConnectionType_Integrated,
    ConnectionType_Local,
    ConnectionType_Remote,
    ClientState_MAX_CLIENT_STATE_SIZE
};

/// @brief Values that can be returned by this application.
enum class ClientReturnCode : int
{
    ClientReturnCode_Success = 0,
    ClientReturnCode_FailedPlatformSpecificInitialization,
    ClientReturnCode_FailedToResizeWindow,
    ClientReturnCode_FailedToInitialize,
    ClientReturnCode_FailedToFindHosts,
    ClientReturnCode_FailedToConnect,
    ClientReturnCode_UnrecognizedStateEncountered,
    ClientReturnCode_FailedToShutDownSDK,
    ClientReturnCode_FailedPlatformSpecificShutdown,
    ClientReturnCode_FailedToRestart,
    ClientReturnCode_FailedWrongTimeToGetData,

    ClientReturnCode_MAX_CLIENT_RETURN_CODE_SIZE
};

/// @brief Used to store the information about the skeleton data coming from the
/// estimation system in Core.
class ClientRawSkeleton
{
public:
    RawSkeletonInfo info;
    std::vector<SkeletonNode> nodes;
};

/// @brief Used to store all the skeleton data coming from the estimation system
/// in Core.
class ClientRawSkeletonCollection
{
public:
    std::vector<ClientRawSkeleton> skeletons;
};

struct GloveRawSkeletonData
{
    GloveRawSkeletonData() = default;

    std::unique_ptr<ClientRawSkeletonCollection> rawSkeleton;
    std::unique_ptr<ClientRawSkeletonCollection> nextRawSkeleton;

    rclcpp::Publisher<manus_ros2_msgs::msg::ManusNodeHierarchy>::SharedPtr nodeHierarchyPub;
    rclcpp::Publisher<manus_ros2_msgs::msg::ManusNodePoses>::SharedPtr nodePosesPub;
};

class SDKMinimalClient : public SDKClientPlatformSpecific, public rclcpp::Node
{
public:
    SDKMinimalClient();

    ~SDKMinimalClient();

    ClientReturnCode Initialize();

    ClientReturnCode InitializeSDK();

    ClientReturnCode ShutDown();

    ClientReturnCode RegisterAllCallbacks();

    static void OnRawSkeletonStreamCallback(
        const SkeletonStreamInfo *const p_RawSkeletonStreamInfo);

    void TimerPosesCallback();

    void TimerHierarchyCallback();

protected:
    ClientReturnCode Connect();

    static SDKMinimalClient *s_Instance;

    ConnectionType m_ConnectionType = ConnectionType::ConnectionType_Invalid;

    std::mutex m_RawSkeletonMutex;
    std::map<uint32_t, GloveRawSkeletonData> m_GloveDataMap;

    // manus message publishers

    rclcpp::TimerBase::SharedPtr m_TimerPoses;
    rclcpp::TimerBase::SharedPtr m_TimerHierarchy;
};

// Close the Doxygen group.
/** @} */
#endif
