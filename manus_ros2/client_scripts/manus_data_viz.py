import rclpy
from manus_ros2_msgs.msg import ManusNodeHierarchy, ManusNodePoses
from rclpy.node import Node
import numpy as np
import open3d as o3d


class GloveViz:
    def __init__(self, glove_id):
        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window()

        self.glove_id = glove_id
        self.node_meshes = {}
        self.line_sets = None


class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__("manus_ros2_client_py")
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
        self.sub_poses = self.create_subscription(
            ManusNodePoses,
            "/manus_node_poses_1",
            self.node_callback,
            20,
        )
        self.sub_hierarchies = self.create_subscription(
            ManusNodeHierarchy,
            "/manus_node_hierarchy_1",
            self.hierarchy_callback,
            20,
        )
        self.timer = self.create_timer(0.02, self.timer_callback)
        self.glove_viz_map: dict[str, GloveViz] = {}

    def node_callback(self, msg: ManusNodePoses):
        if msg.glove_id not in self.glove_viz_map:
            return

        glove_viz = self.glove_viz_map[msg.glove_id]

        points = []
        for node_id, pose in zip(msg.node_ids, msg.poses):
            if node_id not in glove_viz.node_meshes:
                continue

            mesh = glove_viz.node_meshes[node_id]
            mesh.translate([pose.position.x, pose.position.y, pose.position.z], relative=False)
            glove_viz.viz.update_geometry(mesh)

            points.append([pose.position.x, pose.position.y, pose.position.z])

        glove_viz.line_sets.points = o3d.utility.Vector3dVector(points)
        glove_viz.viz.update_geometry(glove_viz.line_sets)

    def hierarchy_callback(self, msg: ManusNodeHierarchy):
        if msg.glove_id not in self.glove_viz_map:
            glove_viz = GloveViz(msg.glove_id)

            for node_id, pose in zip(msg.node_ids, msg.poses):
                mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                mesh.compute_vertex_normals()
                mesh.translate([pose.position.x, pose.position.y, pose.position.z], relative=False)
                glove_viz.viz.add_geometry(mesh)
                glove_viz.node_meshes[node_id] = mesh

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

    def timer_callback(self):
        for glove_viz in self.glove_viz_map.values():
            glove_viz.viz.poll_events()
            glove_viz.viz.update_renderer()


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
