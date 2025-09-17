import cv2
import numpy as np
import json
import argparse
from scipy.spatial.transform import Rotation as rot


MARKER_LENGTH = 0.05

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-v", "--Visualization", help = "Show Output", default=True)

# Read arguments from command line
args = parser.parse_args()

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

def computeMarkerInfo(markers, save_markers, marker_info = []):
    """
    Function to plot the scene with the marker 0 at the origin.
    The camera and other markers are plotted relative to marker 0.
    This function only works if marker 0 is detected.
    
    Parameters:
        markers: List of dictionaries containing a rotation vector and a translation vector
        ax: figure to plot the scene in
        save_markers: Boolean to save the marker positions
        marker_info: List to save the marker positions
    """
    markerZero = None
    for marker in markers:
        if marker["id"] == 0:
            markerZero = marker
            break

    if markerZero is None:
        print("Marker 0 not found")
        return

    markers = [m for m in markers if m["id"] != 0] # Remove marker 0 as its pose is now the origin
    
    T_cam_to_0 = getTransformationMatrix(markerZero)
    T_0_to_cam = np.linalg.inv(T_cam_to_0)

    markers_from_0 = []

    for marker in markers:
        T_cam_to_marker = getTransformationMatrix(marker) 
        T_0_to_marker = T_0_to_cam @ T_cam_to_marker

        R_0_to_marker = T_0_to_marker[:3, :3]
        tvec_0_to_marker = T_0_to_marker[:3, 3]

        quat_0_to_marker = rot.from_matrix(R_0_to_marker).as_quat(scalar_first=True)
        if save_markers:
            print(f"Saving marker {marker['id']}")
            markers_from_0.append({"id": int(marker["id"]), "tvec": tvec_0_to_marker.tolist(), "quat": quat_0_to_marker.tolist()})

    if save_markers and len(markers_from_0) > 0:
        marker_info.append(markers_from_0)

def processVideo(cap, detector, cam_matrix, dist_coeffs, marker_info):
    """
    Processes the video in cap, to paint in the axes of the marker and create a plot to visualize the 3D positions.

    Parameters:
        cap: cv2.VideoCapture containing the video to process
        detector: aruco marker detector
        cam_matrix: matrix containgn the intrinsic parameters of the camera
        dist_coeffs: Array containing the distortion coefficients of the camera
    """

    save_markers = False

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
            if args.Visualization:
                cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("frame", 800, 500)
                cv2.imshow('frame', frame)
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

        if args.Visualization:
            computeMarkerInfo(markers, save_markers, marker_info)
            save_markers = False

            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("frame", 800, 500)
            cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break  # Exit on ESC
        elif key == 32:
            save_markers = True

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

def leastSquaresMarkers(marker_info):
    """
    Function to average the marker positions over multiple frames.
    Uses least squares for quaternion averaging.
    """
    marker_dict = {}
    for frame in marker_info:
        for marker in frame:
            id = marker["id"]
            if id not in marker_dict:
                marker_dict[id] = {"tvecs": [], "quats": []}
            marker_dict[id]["tvecs"].append(marker["tvec"])
            marker_dict[id]["quats"].append(marker["quat"])

    averaged_markers = []
    for id, data in marker_dict.items():
        tvecs = np.array(data["tvecs"])
        quats = np.array(data["quats"])
        avg_tvec = np.mean(tvecs, axis=0).tolist()
        avg_quat = average_quaternions(quats).tolist()
        averaged_markers.append({"id": id, "tvec": avg_tvec, "quat": avg_quat})

    return averaged_markers

def main():
    camera_matrix, dist_coeffs = load_camera_calibration()

    video_capture = cv2.VideoCapture(6) # Now hardcoded as 6, since this is the rgb cam from the rgbd camera

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    marker_info = []

    processVideo(video_capture, detector, camera_matrix, dist_coeffs, marker_info)

    video_capture.release()
    cv2.destroyAllWindows()

    marker_info = leastSquaresMarkers(marker_info)

    with open('src/camera_pose/camera_pose/MarkerInfo/MarkerInfo.json', 'w') as f:
        json.dump(marker_info, f, indent=4)


if __name__ == "__main__":
    main()