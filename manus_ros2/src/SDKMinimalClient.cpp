// SDKMinimalClient.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "SDKMinimalClient.hpp"
#include "ManusSDKTypes.h"
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>

#include "ClientLogging.hpp"

using ManusSDK::ClientLog;
using namespace std::chrono_literals;

SDKMinimalClient *SDKMinimalClient::s_Instance = nullptr;

SDKMinimalClient::SDKMinimalClient() : Node("manus_minimal_client")
{
    if (s_Instance != nullptr) {
        throw std::runtime_error("SDKMinimalClient can only be initialized once.");
    }
    s_Instance = this;

    m_TimerPoses = create_wall_timer(10ms, [this] { TimerPosesCallback(); });
    m_TimerHierarchy = create_wall_timer(200ms, [this] { TimerHierarchyCallback(); });

    // initialize client
    ClientLog::print("Starting minimal client!");
    auto t_Response = Initialize();
    if (t_Response != ClientReturnCode::ClientReturnCode_Success) {
        ClientLog::error("Failed to initialize the SDK. Are you sure the correct ManusSDKLibary is used?");
        throw std::runtime_error("Failed to initialize the SDK. Are you sure the correct ManusSDKLibary is used?");
    }
    ClientLog::print("minimal client is initialized.");

    // SDK is setup. so now go to main loop of the program.
    // first loop until we get a connection
    m_ConnectionType == ConnectionType::ConnectionType_Integrated
        ? ClientLog::print("minimal client is running in integrated mode.")
        : ClientLog::print("minimal client is connecting to MANUS Core. (make sure it is running)");

    while (Connect() != ClientReturnCode::ClientReturnCode_Success) {
        // not yet connected. wait
        ClientLog::print("minimal client could not connect.trying again in a second.");
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    if (m_ConnectionType != ConnectionType::ConnectionType_Integrated)
        ClientLog::print("minimal client is connected, setting up skeletons.");

    // set the hand motion mode of the RawSkeletonStream. This is optional and can be set to any of the HandMotion enum values. Default = None
    // auto will make it move based on available tracking data. If none is available IMU rotation will be used.
    const SDKReturnCode t_HandMotionResult = CoreSdk_SetRawSkeletonHandMotion(HandMotion_Auto);
    if (t_HandMotionResult != SDKReturnCode::SDKReturnCode_Success) {
        ClientLog::error("Failed to set hand motion mode. The value returned was {}.", (int32_t) t_HandMotionResult);
    }
}

SDKMinimalClient::~SDKMinimalClient()
{
    // loop is over. disconnect it all
    ClientLog::print("minimal client is done, shutting down.");
    ShutDown();

    s_Instance = nullptr;
}

/// @brief Initialize the sample console and the SDK.
/// This function attempts to resize the console window and then proceeds to initialize the SDK's interface.
ClientReturnCode SDKMinimalClient::Initialize()
{
    // this corrupts the terminal behavior...
    // if (!PlatformSpecificInitialization())
    // {
    //     return ClientReturnCode::ClientReturnCode_FailedPlatformSpecificInitialization;
    // }

    const ClientReturnCode t_IntializeResult = InitializeSDK();
    if (t_IntializeResult != ClientReturnCode::ClientReturnCode_Success) {
        return ClientReturnCode::ClientReturnCode_FailedToInitialize;
    }

    return ClientReturnCode::ClientReturnCode_Success;
}

/// @brief Initialize the sdk, register the callbacks and set the coordinate system.
/// This needs to be done before any of the other SDK functions can be used.
ClientReturnCode SDKMinimalClient::InitializeSDK()
{
    ClientLog::print("Select what mode you would like to start in (and press enter to submit)");
    ClientLog::print("[1] Core Integrated - This will run standalone without the need for a MANUS Core connection");
    ClientLog::print("[2] Core Local - This will connect to a MANUS Core running locally on your machine");
    ClientLog::print("[3] Core Remote - This will search for a MANUS Core running locally on your network");
    std::string t_ConnectionTypeInput;
    // std::cin >> t_ConnectionTypeInput;
    t_ConnectionTypeInput = "1";

    switch (t_ConnectionTypeInput[0]) {
        case '1':
            m_ConnectionType = ConnectionType::ConnectionType_Integrated;
            ClientLog::print("using connection type: integrated");
            break;
        case '2':
            m_ConnectionType = ConnectionType::ConnectionType_Local;
            ClientLog::print("using connection type: local");
            break;
        case '3':
            m_ConnectionType = ConnectionType::ConnectionType_Remote;
            ClientLog::print("using connection type: remote");
            break;
        default:
            m_ConnectionType = ConnectionType::ConnectionType_Invalid;
            ClientLog::print("Invalid input, try again");
            return InitializeSDK();
    }

    // Invalid connection type detected
    if (m_ConnectionType == ConnectionType::ConnectionType_Invalid ||
        m_ConnectionType == ConnectionType::ClientState_MAX_CLIENT_STATE_SIZE)
        return ClientReturnCode::ClientReturnCode_FailedToInitialize;

    // before we can use the SDK, some internal SDK bits need to be initialized.
    bool t_Remote = m_ConnectionType != ConnectionType::ConnectionType_Integrated;
    const SDKReturnCode t_InitializeResult = CoreSdk_Initialize(SessionType::SessionType_CoreSDK, t_Remote);
    if (t_InitializeResult != SDKReturnCode::SDKReturnCode_Success) {
        return ClientReturnCode::ClientReturnCode_FailedToInitialize;
    }

    const ClientReturnCode t_CallBackResults = RegisterAllCallbacks();
    if (t_CallBackResults != ::ClientReturnCode::ClientReturnCode_Success) {
        return t_CallBackResults;
    }

    // after everything is registered and initialized
    // We specify the coordinate system in which we want to receive the data.
    // (each client can have their own settings. unreal and unity for instance use different coordinate systems)
    // if this is not set, the SDK will not function.
    // The coordinate system used for this example is z-up, x-positive, right-handed and in meter scale.
    CoordinateSystemVUH t_VUH;
    CoordinateSystemVUH_Init(&t_VUH);
    t_VUH.handedness = Side::Side_Right;
    t_VUH.up = AxisPolarity::AxisPolarity_PositiveZ;
    t_VUH.view = AxisView::AxisView_XFromViewer;
    t_VUH.unitScale = 1.0f; // 1.0 is meters, 0.01 is cm, 0.001 is mm.

    // The above specified coordinate system is used to initialize and the coordinate space is specified (world vs local).
    const SDKReturnCode t_CoordinateResult = CoreSdk_InitializeCoordinateSystemWithVUH(t_VUH, true);

    /* this is an example of an alternative way of setting up the coordinate system instead of VUH (view, up, handedness)
    CoordinateSystemDirection t_Direction;
    t_Direction.x = AxisDirection::AD_Right;
    t_Direction.y = AxisDirection::AD_Up;
    t_Direction.z = AxisDirection::AD_Forward;
    const SDKReturnCode t_InitializeResult = CoreSdk_InitializeCoordinateSystemWithDirection(t_Direction, true);
    */

    if (t_CoordinateResult != SDKReturnCode::SDKReturnCode_Success) {
        return ClientReturnCode::ClientReturnCode_FailedToInitialize;
    }

    return ClientReturnCode::ClientReturnCode_Success;
}

/// @brief When shutting down the application, it's important to clean up after the SDK and call it's shutdown function.
/// this will close all connections to the host, close any threads.
/// after this is called it is expected to exit the client program. If not you would need to reinitalize the SDK.
ClientReturnCode SDKMinimalClient::ShutDown()
{
    const SDKReturnCode t_Result = CoreSdk_ShutDown();
    if (t_Result != SDKReturnCode::SDKReturnCode_Success) {
        return ClientReturnCode::ClientReturnCode_FailedToShutDownSDK;
    }

    if (!PlatformSpecificShutdown()) {
        return ClientReturnCode::ClientReturnCode_FailedPlatformSpecificShutdown;
    }

    return ClientReturnCode::ClientReturnCode_Success;
}

/// @brief Used to register all the stream callbacks.
/// Callbacks that are registered functions that get called when a certain 'event' happens, such as data coming in.
/// All of these are optional, but depending on what data you require you may or may not need all of them. For this example we only implement the raw skeleton data.
ClientReturnCode SDKMinimalClient::RegisterAllCallbacks()
{
    // Register the callback to receive Raw Skeleton data
    // it is optional, but without it you can not see any resulting skeleton data.
    // see OnRawSkeletonStreamCallback for more details.
    const SDKReturnCode t_RegisterRawSkeletonCallbackResult = CoreSdk_RegisterCallbackForRawSkeletonStream(
        *OnRawSkeletonStreamCallback);
    if (t_RegisterRawSkeletonCallbackResult != SDKReturnCode::SDKReturnCode_Success) {
        ClientLog::error(
            "Failed to register callback function for processing raw skeletal data from Manus Core. The value returned was {}.",
            (int32_t) t_RegisterRawSkeletonCallbackResult);
        return ClientReturnCode::ClientReturnCode_FailedToInitialize;
    }

    return ClientReturnCode::ClientReturnCode_Success;
}

/// Read latest data from the gloves and publish them as ros2 messages.
void SDKMinimalClient::TimerPosesCallback()
{
    // check if there is new data available.

    m_RawSkeletonMutex.lock();
    for (auto &[_, gloveData]: m_GloveDataMap) {
        if (gloveData.nextRawSkeleton) {
            gloveData.rawSkeleton = std::move(gloveData.nextRawSkeleton);
            gloveData.nextRawSkeleton.reset();
        }
    }
    m_RawSkeletonMutex.unlock();

    for (const auto &[gloveId, gloveData]: m_GloveDataMap) {
        if (!gloveData.rawSkeleton || gloveData.rawSkeleton->skeletons.empty())
            continue;

        manus_ros2_msgs::msg::ManusNodePoses msg;
        msg.glove_id = gloveData.rawSkeleton->skeletons[0].info.gloveId;
        msg.node_count = (int) gloveData.rawSkeleton->skeletons[0].info.nodesCount;

        for (const auto &node: gloveData.rawSkeleton->skeletons[0].nodes) {
            // prints the position and rotation of the first node in the first skeleton
            ManusVec3 t_Pos = node.transform.position;
            ManusQuaternion t_Rot = node.transform.rotation;

            msg.node_ids.push_back(node.id);

            auto &pose = msg.poses.emplace_back();
            pose.position.x = t_Pos.x;
            pose.position.y = t_Pos.y;
            pose.position.z = t_Pos.z;
            pose.orientation.x = t_Rot.x;
            pose.orientation.y = t_Rot.y;
            pose.orientation.z = t_Rot.z;
            pose.orientation.w = t_Rot.w;
        }

        ClientLog::print("node pose message published: {}", gloveId);
        gloveData.nodePosesPub->publish(msg);
    }
}

void SDKMinimalClient::TimerHierarchyCallback()
{
    for (const auto &[gloveId, gloveData]: m_GloveDataMap) {
        if (!gloveData.rawSkeleton || gloveData.rawSkeleton->skeletons.empty())
            continue;

        // this section demonstrates how to interpret the raw skeleton data.
        // how to get the hierarchy of the skeleton, and how to know bone each node represents.

        manus_ros2_msgs::msg::ManusNodeHierarchy msg;

        uint32_t t_GloveId = 0;
        uint32_t t_NodeCount = 0;

        t_GloveId = gloveData.rawSkeleton->skeletons[0].info.gloveId;
        t_NodeCount = 0;

        SDKReturnCode t_Result = CoreSdk_GetRawSkeletonNodeCount(t_GloveId, t_NodeCount);
        if (t_Result != SDKReturnCode::SDKReturnCode_Success) {
            ClientLog::error("Failed to get Raw Skeleton Node Count. The error given was {}.", (int32_t) t_Result);
            return;
        }

        msg.glove_id = t_GloveId;

        // now get the hierarchy data, this needs to be used to reconstruct the positions of each node in case the user set up the system with a local coordinate system.
        // having a node position defined as local means that this will be related to its parent.

        NodeInfo *t_NodeInfo = new NodeInfo[t_NodeCount];
        t_Result = CoreSdk_GetRawSkeletonNodeInfoArray(t_GloveId, t_NodeInfo, t_NodeCount);
        if (t_Result != SDKReturnCode::SDKReturnCode_Success) {
            ClientLog::error("Failed to get Raw Skeleton Hierarchy. The error given was {}.", (int32_t) t_Result);
            return;
        }

        for (int i = 0; i < t_NodeCount; ++i) {
            // skip invalid nodes
            if (t_NodeInfo[i].side == Side_Invalid || t_NodeInfo[i].chainType == ChainType_Invalid) {
                continue;
            }

            msg.node_ids.push_back(t_NodeInfo[i].nodeId);
            msg.parent_node_ids.push_back(t_NodeInfo[i].parentId);
            // prints the position and rotation of the first node in the first skeleton

            ManusVec3 t_Pos = gloveData.rawSkeleton->skeletons[0].nodes[i].transform.position;
            ManusQuaternion t_Rot = gloveData.rawSkeleton->skeletons[0].nodes[i].transform.rotation;

            auto &pose = msg.poses.emplace_back();
            pose.position.x = t_Pos.x;
            pose.position.y = t_Pos.y;
            pose.position.z = t_Pos.z;
            pose.orientation.x = t_Rot.x;
            pose.orientation.y = t_Rot.y;
            pose.orientation.z = t_Rot.z;
            pose.orientation.w = t_Rot.w;
        }

        msg.node_count = (int) msg.node_ids.size();

        ClientLog::print("node hierarchy message published: {}", gloveId);
        gloveData.nodeHierarchyPub->publish(msg);

        delete[] t_NodeInfo;
    }
}

/// @brief the client will now try to connect to MANUS Core via the SDK when the ConnectionType is not integrated. These steps still need to be followed when using the integrated ConnectionType.
ClientReturnCode SDKMinimalClient::Connect()
{
    bool t_ConnectLocally = m_ConnectionType == ConnectionType::ConnectionType_Local;
    SDKReturnCode t_StartResult = CoreSdk_LookForHosts(1, t_ConnectLocally);
    if (t_StartResult != SDKReturnCode::SDKReturnCode_Success) {
        return ClientReturnCode::ClientReturnCode_FailedToFindHosts;
    }

    uint32_t t_NumberOfHostsFound = 0;
    SDKReturnCode t_NumberResult = CoreSdk_GetNumberOfAvailableHostsFound(&t_NumberOfHostsFound);
    if (t_NumberResult != SDKReturnCode::SDKReturnCode_Success) {
        return ClientReturnCode::ClientReturnCode_FailedToFindHosts;
    }

    if (t_NumberOfHostsFound == 0) {
        return ClientReturnCode::ClientReturnCode_FailedToFindHosts;
    }

    std::unique_ptr<ManusHost[]> t_AvailableHosts;
    t_AvailableHosts.reset(new ManusHost[t_NumberOfHostsFound]);

    SDKReturnCode t_HostsResult = CoreSdk_GetAvailableHostsFound(t_AvailableHosts.get(), t_NumberOfHostsFound);
    if (t_HostsResult != SDKReturnCode::SDKReturnCode_Success) {
        return ClientReturnCode::ClientReturnCode_FailedToFindHosts;
    }

    uint32_t t_HostSelection = 0;
    if (!t_ConnectLocally && t_NumberOfHostsFound > 1) {
        ClientLog::print("Select which host you want to connect to (and press enter to submit)");
        for (size_t i = 0; i < t_NumberOfHostsFound; i++) {
            auto t_HostInfo = t_AvailableHosts[i];
            ClientLog::print("[{}] hostname: {}, IP address: {}, version {}.{}.{}", i + 1, t_HostInfo.hostName,
                             t_HostInfo.ipAddress, t_HostInfo.manusCoreVersion.major, t_HostInfo.manusCoreVersion.minor,
                             t_HostInfo.manusCoreVersion.patch);
        }
        uint32_t t_HostSelectionInput = 0;
        std::cin >> t_HostSelectionInput;
        if (t_HostSelectionInput <= 0 || t_HostSelectionInput > t_NumberOfHostsFound)
            return ClientReturnCode::ClientReturnCode_FailedToConnect;

        t_HostSelection = t_HostSelectionInput - 1;
    }

    SDKReturnCode t_ConnectResult = CoreSdk_ConnectToHost(t_AvailableHosts[t_HostSelection]);

    if (t_ConnectResult == SDKReturnCode::SDKReturnCode_NotConnected) {
        return ClientReturnCode::ClientReturnCode_FailedToConnect;
    }

    return ClientReturnCode::ClientReturnCode_Success;
}

/// @brief This gets called when the client is connected and there is glove data available.
/// @param p_RawSkeletonStreamInfo contains the meta data on what data is available and needs to be retrieved from the SDK.
/// The data is not directly passed to the callback, but needs to be retrieved from the SDK for it to be used. This is demonstrated in the function below.
void SDKMinimalClient::OnRawSkeletonStreamCallback(const SkeletonStreamInfo *const p_RawSkeletonStreamInfo)
{
    if (s_Instance) {
        ClientRawSkeletonCollection *t_NxtClientRawSkeleton = new ClientRawSkeletonCollection();
        t_NxtClientRawSkeleton->skeletons.resize(p_RawSkeletonStreamInfo->skeletonsCount);

        for (uint32_t i = 0; i < p_RawSkeletonStreamInfo->skeletonsCount; i++) {
            // Retrieves info on the skeletonData, like deviceID and the amount of nodes.
            CoreSdk_GetRawSkeletonInfo(i, &t_NxtClientRawSkeleton->skeletons[i].info);
            t_NxtClientRawSkeleton->skeletons[i].nodes.resize(t_NxtClientRawSkeleton->skeletons[i].info.nodesCount);
            t_NxtClientRawSkeleton->skeletons[i].info.publishTime = p_RawSkeletonStreamInfo->publishTime;

            // Retrieves the skeletonData, which contains the node data.
            CoreSdk_GetRawSkeletonData(i, t_NxtClientRawSkeleton->skeletons[i].nodes.data(),
                                       t_NxtClientRawSkeleton->skeletons[i].info.nodesCount);
        }

        // retrieve glove id
        const auto &rawSkeleton = t_NxtClientRawSkeleton;
        uint32_t t_GloveId = rawSkeleton->skeletons[0].info.gloveId;

        auto &gloveDataMap = s_Instance->m_GloveDataMap;
        s_Instance->m_RawSkeletonMutex.lock();

        // create new publishers for new glove
        if (gloveDataMap.find(t_GloveId) == gloveDataMap.end()) {
            auto gloveIndex = gloveDataMap.size();
            auto iter = gloveDataMap.emplace(t_GloveId, GloveRawSkeletonData{}).first;

            iter->second.nodeHierarchyPub = s_Instance->create_publisher<manus_ros2_msgs::msg::ManusNodeHierarchy>(
                "manus_node_hierarchy_" + std::to_string(gloveIndex), 10);
            iter->second.nodePosesPub = s_Instance->create_publisher<manus_ros2_msgs::msg::ManusNodePoses>(
                "manus_node_poses_" + std::to_string(gloveIndex), 10);
        }

        // fetch new data
        gloveDataMap[t_GloveId].nextRawSkeleton = std::unique_ptr<ClientRawSkeletonCollection>(t_NxtClientRawSkeleton);

        s_Instance->m_RawSkeletonMutex.unlock();
    }
}
