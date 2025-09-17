import copy
import sys
import threading
from pathlib import Path

from scipy.spatial.transform import Rotation as rot
import cv2
import mink
import mujoco
import mujoco.viewer
import numpy as np
import open3d as o3d
import rclpy
from loop_rate_limiters import RateLimiter
from manus_ros2_msgs.msg import ManusNodeHierarchy, ManusNodePoses
from geometry_msgs.msg import Pose

from rclpy.node import Node

SHADOW_HAND_XML = f"{Path(__file__).parent.as_posix()}/mink_repo/examples/shadow_hand/scene_left.xml"


class HandControl:
    """class for controlling the shadow hand in mujoco"""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(SHADOW_HAND_XML)
        self.configuration = mink.Configuration(self.model)

        # set up finger target reaching tasks
        self.fingers = ["thumb", "first", "middle", "ring", "little"]
        self.finger_tasks: list[mink.FrameTask] = []
        for finger in self.fingers:
            task = mink.FrameTask(
                frame_name=finger,
                frame_type="site",
                position_cost=1.0,
                orientation_cost=0.0,
                lm_damping=1.0,
            )
            self.finger_tasks.append(task)

        self.tasks = [
            *self.finger_tasks,
        ]

        self.model = self.configuration.model
        self.data = self.configuration.data
        self.solver = "quadprog"

        # launch mujoco viewer
        self.viewer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
            # show_left_ui=False,
            # show_right_ui=False,
        )
        mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)
        # self.configuration.update_from_keyframe("open hand")

        # Initialize mocap bodies at their respective sites.
        # posture_task.set_target_from_configuration(self.configuration)
        # for finger in self.fingers:
        #     mink.move_mocap_to_frame(self.model, self.data, f"{finger}_target", finger, "site")

        # print("timestep:", self.dt)

        # visualize mujoco sites
        self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        # Some options here below
        # self.viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1

        self.viewer.opt.sitegroup[4] = 1
        self.viewer.sync()

        self.target_updated = False
        self.raw_targets = None

        # lock for multi-threading
        self.lock = threading.Lock()
        self.rate = RateLimiter(frequency=100.0, warn=False)
        self.pos_from_cam = np.zeros(3)
        self.rot_from_cam = np.zeros(4)
        self.rot_from_cam[0] = 1.0

    def update_target(self, finger_positions):
        """update finger tip target positions"""
        with self.lock:
            self.target_updated = True
            self.raw_targets = copy.deepcopy(finger_positions)

    def step(self, dt):
        with self.lock:
            # run inverse kinematics to reach the target
            if self.target_updated:
                # Update task target.
                for finger, task in zip(self.fingers, self.finger_tasks):
                    task.set_target(mink.SE3.from_translation(self.raw_targets[finger]))

                # print(self.data.body("maniobj"))
                vel = mink.solve_ik(self.configuration, self.tasks, dt, self.solver, 1e-5)
                self.configuration.integrate_inplace(vel, dt)

                mujoco.mj_camlight(self.model, self.data)

                # visualize targets
                self.viewer.user_scn.ngeom = 0
                for i, target in enumerate(self.raw_targets.values()):
                    mujoco.mjv_initGeom(
                        self.viewer.user_scn.geoms[i],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.01, 0, 0],
                        pos=target,
                        mat=np.eye(3).flatten(),
                        rgba=[1, 0.2, 0.7, 1],
                    )
                self.viewer.user_scn.ngeom = i + 1

            # print(self.model.body("lh_forearm").pos)
            self.model.body("lh_forearm").pos = self.pos_from_cam # Replace with some pos from my aruco marker detection
            self.model.body("lh_forearm").quat = self.rot_from_cam

            # step environment
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()


class GloveViz:
    """open3d visualization for glove data"""

    def __init__(self, glove_id):
        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window()

        self.glove_id = glove_id
        self.node_meshes = {}
        self.line_sets = None

        self.frame_mesh_restore_rot_mat = np.eye(3)
        self.frame_mesh = None


class MinimalSubscriber(Node):
    def __init__(self, hand_control: HandControl):
        super().__init__("manus_ros2_client_py")

        # subscribe to glove data topics
        # /manus_node_poses_X: poses for all nodes on the glove
        # /manus_node_hierarchy_X: hierarchy of nodes on the glove
        self.sub_poses = self.create_subscription(
            ManusNodePoses,
            "/manus_node_poses_0",
            self.node_callback,
            20,
        )
        self.sub_hierarchies = self.create_subscription(
            ManusNodeHierarchy,
            "/manus_node_hierarchy_0",
            self.hierarchy_callback,
            20,
        )
        self.sub_cam_poses = self.create_subscription(
            Pose,
            "camera_pose",
            self.cam_callback,
            20,
        )

        self.timer = self.create_timer(0.02, self.timer_callback)
        self.glove_viz_map: dict[str, GloveViz] = {}

        self.hand_ctl = hand_control

    def node_callback(self, msg: ManusNodePoses):
        """callback for glove node poses"""
        if msg.glove_id not in self.glove_viz_map:
            return

        glove_viz = self.glove_viz_map[msg.glove_id]

        points = []
        root_pose: mink.SE3 = None

        # update open3d visualization
        for node_id, pose in zip(msg.node_ids, msg.poses):
            if node_id not in glove_viz.node_meshes:
                continue

            mesh = glove_viz.node_meshes[node_id]
            mesh.translate([pose.position.x, pose.position.y, pose.position.z], relative=False)
            glove_viz.viz.update_geometry(mesh)

            points.append([pose.position.x, pose.position.y, pose.position.z])

            # draw axis aligned with the hand root node
            if node_id == 0:
                rot_mat = o3d.geometry.TriangleMesh.get_rotation_matrix_from_quaternion(
                    [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
                )
                glove_viz.frame_mesh.rotate(glove_viz.frame_mesh_restore_rot_mat)
                glove_viz.frame_mesh_restore_rot_mat = np.linalg.inv(rot_mat)
                glove_viz.frame_mesh.rotate(rot_mat)
                glove_viz.frame_mesh.translate([pose.position.x, pose.position.y, pose.position.z], relative=False)
                glove_viz.viz.update_geometry(glove_viz.frame_mesh)

                root_pose = mink.SE3.from_rotation_and_translation(
                    mink.SO3.from_matrix(rot_mat),
                    np.array([pose.position.x, pose.position.y, pose.position.z]),
                )

        assert root_pose is not None
        glove_viz.line_sets.points = o3d.utility.Vector3dVector(points)
        glove_viz.viz.update_geometry(glove_viz.line_sets)

        # get finger tip poses
        tip_positions = {}

        for node_id, pose in zip(msg.node_ids, msg.poses):
            if node_id not in glove_viz.node_meshes:
                continue

            rot_mat = o3d.geometry.TriangleMesh.get_rotation_matrix_from_quaternion(
                [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
            )
            tip_pose = mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(rot_mat),
                np.array([pose.position.x, pose.position.y, pose.position.z]),
            )

            # thumb: 24
            if node_id == 24:
                pos = root_pose.inverse().multiply(tip_pose).translation()
                pos = np.array([pos[0], -pos[1], -pos[2]])
                pos *= 1.0
                pos += self.hand_ctl.data.site("wrist").xpos
                tip_positions["thumb"] = pos
            # first: 5
            elif node_id == 5:
                pos = root_pose.inverse().multiply(tip_pose).translation()
                pos = np.array([pos[0], -pos[1], -pos[2]])
                pos *= 1.0
                pos += self.hand_ctl.data.site("wrist").xpos
                tip_positions["first"] = pos
            # middle: 10
            elif node_id == 10:
                pos = root_pose.inverse().multiply(tip_pose).translation()
                pos = np.array([pos[0], -pos[1], -pos[2]])
                pos *= 1.0
                pos += self.hand_ctl.data.site("wrist").xpos
                tip_positions["middle"] = pos
            # ring 20
            elif node_id == 20:
                pos = root_pose.inverse().multiply(tip_pose).translation()
                pos = np.array([pos[0], -pos[1], -pos[2]])
                pos *= 1.0
                pos += self.hand_ctl.data.site("wrist").xpos
                tip_positions["ring"] = pos
            # little: 15
            elif node_id == 15:
                pos = root_pose.inverse().multiply(tip_pose).translation()
                pos = np.array([pos[0], -pos[1], -pos[2]])
                pos *= 1.1
                pos += self.hand_ctl.data.site("wrist").xpos
                tip_positions["little"] = pos

        assert len(tip_positions) == 5
        self.hand_ctl.update_target(tip_positions)
        # print(tip_positions)

    def hierarchy_callback(self, msg: ManusNodeHierarchy):
        """callback for glove node hierarchy"""

        # add a new glove visualization
        if msg.glove_id not in self.glove_viz_map:
            glove_viz = GloveViz(msg.glove_id)

            # frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            # glove_viz.viz.add_geometry(frame_mesh)

            for node_id, pose in zip(msg.node_ids, msg.poses):
                mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                mesh.compute_vertex_normals()
                mesh.translate([pose.position.x, pose.position.y, pose.position.z], relative=False)
                glove_viz.viz.add_geometry(mesh)
                glove_viz.node_meshes[node_id] = mesh

                # draw axis aligned with the hand root node
                if node_id == 0:
                    frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

                    rot_mat = o3d.geometry.TriangleMesh.get_rotation_matrix_from_quaternion(
                        [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
                    )
                    glove_viz.frame_mesh = frame_mesh
                    glove_viz.frame_mesh_restore_rot_mat = np.linalg.inv(rot_mat)

                    frame_mesh.rotate(rot_mat)
                    frame_mesh.translate([pose.position.x, pose.position.y, pose.position.z], relative=False)
                    glove_viz.viz.add_geometry(frame_mesh)

            # draw glove skeletons
            points = [[pose.position.x, pose.position.y, pose.position.z] for pose in msg.poses]
            node_id_to_index = {node_id: i for i, node_id in enumerate(msg.node_ids)}
            lines = [
                [node_id_to_index[node_id], node_id_to_index[parent_node_id]]
                for node_id, parent_node_id in zip(msg.node_ids, msg.parent_node_ids)
                if parent_node_id in node_id_to_index
            ]
            line_sets = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector(lines),
            )
            glove_viz.viz.add_geometry(line_sets)
            glove_viz.line_sets = line_sets

            self.glove_viz_map[msg.glove_id] = glove_viz

    def cam_callback(self, msg: Pose):
        """callback for camera pose"""

        # update hand position and orientation based on camera pose
        # print("pos:", msg.position.x, msg.position.y, msg.position.z, "quat:", msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z)
        # self.hand_ctl.pos_from_cam = [msg.position.x, msg.position.y, msg.position.z]
        self.hand_ctl.rot_from_cam = [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]

    def timer_callback(self):
        for glove_viz in self.glove_viz_map.values():
            glove_viz.viz.poll_events()
            glove_viz.viz.update_renderer()


def spin_node(hand_control):
    rclpy.init(args=sys.argv)

    minimal_subscriber = MinimalSubscriber(hand_control)
    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


def main():
    hand_control = HandControl()

    spin_thread = threading.Thread(target=spin_node, daemon=True, args=(hand_control,))
    spin_thread.start()

    rate = RateLimiter(frequency=100.0, warn=False)
    while hand_control.viewer.is_running():
        hand_control.step(rate.dt)
        rate.sleep()


if __name__ == "__main__":
    main()
