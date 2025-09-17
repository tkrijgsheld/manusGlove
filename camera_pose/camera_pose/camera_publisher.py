import cv2
import numpy as np
import json
import pylab
import argparse
from scipy.spatial.transform import Rotation as rot
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, Point, Quaternion


MARKER_LENGTH = 0.05

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-v", "--Visualization", help = "Show Output", default=True)

# Read arguments from command line
args = parser.parse_args()

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

def load_camera_calibration(calib_file="src/camera_pose/camera_pose/CamData/rgbd_with_metal_thing.json"):
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

def clearFig(ax):
    """
    Clears the figure in order to show only one marker/camera position per frame

    Parameters:
        ax: figure to clear
    """
    ax.cla()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.5,0.5)
    ax.set_zlim(-0.5,0.5)

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

def plotZeroMarker(ax):
    points = np.array([[-MARKER_LENGTH/2, MARKER_LENGTH/2, 0, 1], 
                        [MARKER_LENGTH/2, MARKER_LENGTH/2, 0, 1], 
                        [MARKER_LENGTH/2, -MARKER_LENGTH/2, 0, 1],
                        [-MARKER_LENGTH/2, -MARKER_LENGTH/2, 0, 1]])

    
    ax.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], [points[0][2], points[1][2]], c='b')
    ax.plot([points[1][0], points[2][0]], [points[1][1], points[2][1]], [points[1][2], points[2][2]], c='b')
    ax.plot([points[2][0], points[3][0]], [points[2][1], points[3][1]], [points[2][2], points[3][2]], c='b')
    ax.plot([points[3][0], points[0][0]], [points[3][1], points[0][1]], [points[3][2], points[0][2]], c='b')

def plotScene(marker_poses, quatCam, tvecCam, ax):
    """
    Function to plot the scene with the marker 0 at the origin.
    The camera and other markers are plotted relative to marker 0.
    If marker 0 is not detected, it is still assumed to be the origin.
    Parameters:
        markers: List of dictionaries containing a rotation vector and a translation vector
        ax: figure to plot the scene in
    """
    clearFig(ax)

    plotZeroMarker(ax)

    rotation_cam = rot.from_quat(quatCam, scalar_first=True)

    xPoint = rotation_cam.apply([0.1, 0, 0]) + tvecCam
    yPoint = rotation_cam.apply([0, 0.1, 0]) + tvecCam
    zPoint = rotation_cam.apply([0, 0, 0.1]) + tvecCam
    ax.scatter(tvecCam[0], tvecCam[1], tvecCam[2], c='black')
    ax.plot([tvecCam[0], xPoint[0]], [tvecCam[1], xPoint[1]], [tvecCam[2], xPoint[2]], c='r')
    ax.plot([tvecCam[0], yPoint[0]], [tvecCam[1], yPoint[1]], [tvecCam[2], yPoint[2]], c='g')
    ax.plot([tvecCam[0], zPoint[0]], [tvecCam[1], zPoint[1]], [tvecCam[2], zPoint[2]], c='b')

    for marker in marker_poses:

        rotation_0_to_marker = rot.from_quat(marker["quat"], scalar_first=True)
        tvec_0_to_marker = marker["tvec"]
        R_0_to_marker = rotation_0_to_marker.as_matrix()
        T_0_to_marker = np.eye(4)
        T_0_to_marker[:3, :3] = R_0_to_marker
        T_0_to_marker[:3, 3] = tvec_0_to_marker

        object_points_from_object = np.array([[-MARKER_LENGTH/2, MARKER_LENGTH/2, 0, 1], 
                                              [MARKER_LENGTH/2, MARKER_LENGTH/2, 0, 1], 
                                              [MARKER_LENGTH/2, -MARKER_LENGTH/2, 0, 1],
                                              [-MARKER_LENGTH/2, -MARKER_LENGTH/2, 0, 1]])
        points = [T_0_to_marker @ v for v in object_points_from_object]
        ax.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], [points[0][2], points[1][2]], c='b')
        ax.plot([points[1][0], points[2][0]], [points[1][1], points[2][1]], [points[1][2], points[2][2]], c='b')
        ax.plot([points[2][0], points[3][0]], [points[2][1], points[3][1]], [points[2][2], points[3][2]], c='b')
        ax.plot([points[3][0], points[0][0]], [points[3][1], points[0][1]], [points[3][2], points[0][2]], c='b')
        textPos = T_0_to_marker@[0,0,0,1]
        ax.text(textPos[0], textPos[1], textPos[2], f"{marker['id']}")
    
    pylab.pause(0.01)

def average_quaternions(quaternions):
    """
    Compute the average quaternion using the method of Markley et al.
    quaternions: Nx4 numpy array
    Returns: 4-element numpy array (average quaternion)
    """
    M = np.zeros((4, 4))
    for q in quaternions:
        q = np.array(q, dtype=np.float64).reshape(4, 1)
        M += q @ q.T
    # Get the eigenvector corresponding to the largest eigenvalue
    eigvals, eigvecs = np.linalg.eigh(M)
    avg_quat = eigvecs[:, np.argmax(eigvals)]
    # Ensure the quaternion has positive scalar part
    if avg_quat[0] < 0:
        avg_quat = -avg_quat
    return avg_quat

def getPoseCamera(markers, marker_info):
    """
    Function to get the pose the camera in the world (marker 0) frame.
    If multiple markers are detected, the average pose is returned.
    TODO: Improve this by using a more robust method. Like least squares.

    Parameters:
        markers: List of dictionaries containing a rotation vector and a translation vector
        marker_info: List containing the saved marker info from the scene

    Returns:
        rvecCam: Rotation vector from marker to camera
        tvecCam: Translation vector from marker to camera
    """
    if len(markers) == 0:
        return None, None
    
    cam_poses = []
    marker_poses = []

    for marker in markers:
        id = marker["id"]
        T_cam_to_marker = getTransformationMatrix(marker)
        T_marker_to_camera = np.linalg.inv(T_cam_to_marker)
        R_marker_to_camera = T_marker_to_camera[:3, :3]
        tvec_marker_to_camera = T_marker_to_camera[:3, 3]
        rotation_marker_to_camera = rot.from_matrix(R_marker_to_camera)
        quat_marker_to_camera = rotation_marker_to_camera.as_quat(scalar_first=True)
        if id == 0:
            cam_poses.append({"tvec": tvec_marker_to_camera, "quat": quat_marker_to_camera, "id": id})
        else:
            this_marker_info = None
            for info in marker_info:
                if info["id"] == id:
                    this_marker_info = info
                    break
            if this_marker_info is None:
                print(f"Unknown marker id: {id}. Cannot compute cam pose relative to marker")
                continue
            tvec_zero_to_marker = np.array(this_marker_info["tvec"])
            quat_zero_to_marker = np.array(this_marker_info["quat"])
            rotation_zero_to_marker = rot.from_quat(quat_zero_to_marker, scalar_first=True)
            rotation_zero_to_camera = rotation_zero_to_marker * rotation_marker_to_camera
            quat_zero_to_camera = rotation_zero_to_camera.as_quat(scalar_first=True)
            tvec_zero_to_camera = rotation_zero_to_marker.apply(tvec_marker_to_camera) + tvec_zero_to_marker
            cam_poses.append({"tvec": tvec_zero_to_camera, "quat": quat_zero_to_camera, "id": id})
            marker_poses.append({"tvec": tvec_zero_to_marker, "quat": quat_zero_to_marker, "id": id})

    if len(cam_poses) == 0:
        return None, None, marker_poses

    tvecCam = np.zeros(3)
    quatCam = average_quaternions([pose["quat"] for pose in cam_poses])
    for pose in cam_poses:
        tvecCam += pose["tvec"]
    tvecCam /= len(cam_poses)

    return quatCam, tvecCam, marker_poses

def processVideo(cap, detector, cam_matrix, dist_coeffs, marker_info, ros2Publisher):
    """
    Processes the video in cap, to paint in the axes of the marker and create a plot to visualize the 3D positions.
    
    Parameters:
        cap: cv2.VideoCapture containing the video to process
        detector: aruco marker detector
        cam_matrix: matrix containgn the intrinsic parameters of the camera
        dist_coeffs: Array containing the distortion coefficients of the camera
    """
    # Prepare figure
    if args.Visualization:
        fig_camera_frame = pylab.figure("Camera frame")
        ax_camera_frame = fig_camera_frame.add_subplot(111, projection='3d')
        ax_camera_frame.view_init(elev=90, azim=-90, roll=0)
        pylab.pause(0.01)

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
            # if args.Visualization:
                # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
                # cv2.resizeWindow("frame", 800, 500)
                # cv2.imshow('frame', frame)
            key = cv2.waitKey(1)
            if key == 27:
                break  # Exit on ESC
            continue

        for (corner, id) in zip(corners, ids):
            success, rvec, tvec = cv2.solvePnP(object_points, corner.astype(np.float32), cam_matrix, dist_coeffs, cv2.SOLVEPNP_IPPE_SQUARE)
            if success:
                marker = {"rvec": rvec,
                        "tvec": tvec, 
                        "id": id[0]}
                markers.append(marker)
                cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH*1.5, 2)

        quatCam, tvecCam, marker_poses = getPoseCamera(markers, marker_info)

        if quatCam is None or tvecCam is None:
            # if args.Visualization:
                # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
                # cv2.resizeWindow("frame", 800, 500)
                # cv2.imshow('frame', frame)
            key = cv2.waitKey(1)
            if key == 27:
                break  # Exit on ESC
            continue

        if args.Visualization:
            plotScene(marker_poses, quatCam, tvecCam, ax_camera_frame)

            # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("frame", 800, 500)
            # cv2.imshow('frame', frame)

        if ros2Publisher is not None:
            ros2Publisher.sendData(tvecCam.tolist(), quatCam.tolist())

        key = cv2.waitKey(1)
        if key == 27:
            break  # Exit on ESC

def readMarkerInfo(file="src/camera_pose/camera_pose/MarkerInfo/MarkerInfo.json"):
    """
    Reads the marker info from a json file.

    Parameters:
        file: Path to the json file
    Returns:
        marker_info: List containing the marker info
    """
    try:
        with open(file, 'r') as f:
            marker_info = json.load(f)
    except FileNotFoundError:
        marker_info = []
    return marker_info

def main(ros2Publisher=None):

    rclpy.init(args=None)

    minimal_publisher = MinimalPublisher()

    camera_matrix, dist_coeffs = load_camera_calibration()

    video_capture = cv2.VideoCapture(6) # Now hardcoded as 6, since this is the rgb cam from the rgbd camera

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250) #TODO: Change to 4x4 when changing input
    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    marker_info = readMarkerInfo()

    processVideo(video_capture, detector, camera_matrix, dist_coeffs, marker_info, minimal_publisher)

    video_capture.release()
    cv2.destroyAllWindows()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()