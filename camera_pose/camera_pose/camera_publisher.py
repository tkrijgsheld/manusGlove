import cv2
import numpy as np
import json
import scipy.spatial.transform.rotation as rot
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, Point, Quaternion

MARKER_LENGTH = 0.05

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(Pose, 'camera_pose', 10)
    
    def sendData(self, pos, orientation):
        msg = Pose()
        msg.position = Point(
            x=pos[0],
            y=pos[1],
            z=pos[2]
        )
        msg.orientation = Quaternion(w = orientation[0],
                                    x = orientation[1],
                                    y = orientation[2],
                                    z = orientation[3])
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg)

def load_camera_calibration(calib_file="src/camera_pose/camera_pose/CamData/rgbd.json"):
    """
    Load camera calibration parameters from a JSON file. 
    From Shady

    Parameters:
        calib_file: Path to the JSON file

    Returns:
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients
    """
    with open(calib_file, 'r') as f:
        calibration_data = json.load(f)

    camera_matrix = np.array([
        [calibration_data["fx"], 0, calibration_data["px"]],
        [0, calibration_data["fy"], calibration_data["py"]],
        [0, 0, 1]
    ])

    dist_coeffs = np.array(calibration_data["dist_coeffs"])

    return camera_matrix, dist_coeffs

def getTransformationMatrix(marker):
    """
    Creates transformation matrix from camera to marker

    Parameters:
        marker: Dictionary containing a rotaion vector and a translation vector of the marker

    Returns:
        T_camera_to_marker: Transformation matrix from camere to marker
    """
    rvec = marker["rvec"]
    tvec = marker["tvec"]
    tvec = np.array(tvec).flatten()

    # Some stuff from Shady
    R, _ = cv2.Rodrigues(np.array(rvec))
    T_camera_to_marker = np.eye(4)
    T_camera_to_marker[:3, :3] = R
    T_camera_to_marker[:3, 3] = tvec

    return T_camera_to_marker

def getTransformationCamera(markers):
    """
    Function to get the transformation from the marker to the camera.
    If multiple markers are detected, the average transformation is returned.
    TODO: Improve this by using a more robust method. Like least squares.
        Also this is relative to one of the markers, not to the world frame. Right? #TODO

    Parameters:
        markers: List of dictionaries containing a rotation vector and a translation vector

    Returns:
        rvecCam: Rotation vector from marker to camera
        tvecCam: Translation vector from marker to camera
    """
    if len(markers) == 0:
        return None, None

    T_matrices = [getTransformationMatrix(marker) for marker in markers]
    T_camera_to_marker = sum(T_matrices) / len(T_matrices)
    R_camera_to_marker = T_camera_to_marker[:3, :3]
    tvec_camera_to_marker = T_camera_to_marker[:3, 3]

    R_marker_to_camera = R_camera_to_marker.T
    tvec_marker_to_camera = -R_marker_to_camera @ tvec_camera_to_marker

    rvecCam, _ = cv2.Rodrigues(R_marker_to_camera)

    rotation = rot.Rotation.from_matrix(R_marker_to_camera)
    rvecCam = rotation.as_quat().reshape((4, 1)) # Scalar first = False TODO?

    tvecCam = tvec_marker_to_camera.reshape((3, 1))

    return rvecCam.flatten(), tvecCam.flatten()

def processVideo(cap, detector, cam_matrix, dist_coeffs, ros2Publisher):
    """
    Processes the video in cap, to paint in the axes of the marker and create a plot to visualize the 3D positions.

    Parameters:
        cap: cv2.VideoCapture containing the video to process
        detector: aruco marker detector
        cam_matrix: matrix containgn the intrinsic parameters of the camera
        dist_coeffs: Array containing the distortion coefficients of the camera
    """
    while True:
        ret, frame = cap.read()
        markers = []

        if not ret:
            break

        corners, ids, rejected = detector.detectMarkers(frame)

        object_points = np.array([[-MARKER_LENGTH/2, MARKER_LENGTH/2, 0], 
                        [MARKER_LENGTH/2, MARKER_LENGTH/2, 0], 
                        [MARKER_LENGTH/2, -MARKER_LENGTH/2, 0],
                        [-MARKER_LENGTH/2, -MARKER_LENGTH/2, 0]], dtype=np.float32)

        if np.any(ids == None):
            print("No markers detected")
            continue

        for (corner, id) in zip(corners, ids):
            success, rvec, tvec = cv2.solvePnP(object_points, corner.astype(np.float32), cam_matrix, dist_coeffs, cv2.SOLVEPNP_IPPE_SQUARE)
            if success:
                marker = {"rvec": rvec,
                        "tvec": tvec, 
                        "id": id}
                markers.append(marker)
                cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH*1.5, 2)

        rvecCam, tvecCam = getTransformationCamera(markers)
        pos = []
        quat = []
        for i in rvecCam:
            quat.append(float(i))
        for i in tvecCam:
            pos.append(float(i))
        print('pos:', pos, 'quat:', quat)
        if ros2Publisher is not None:
            ros2Publisher.sendData(pos, quat)

        key = cv2.waitKey(1)
        if key == 27:
            break  # Exit on ESC

def main(ros2Publisher=None):

    rclpy.init(args=None)

    minimal_publisher = MinimalPublisher()

    # rclpy.spin(minimal_publisher)

    camera_matrix, dist_coeffs = load_camera_calibration()

    video_capture = cv2.VideoCapture(0) # Now hardcoded as 6, since this is the rgb cam from the rgbd camera

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    processVideo(video_capture, detector, camera_matrix, dist_coeffs, minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()