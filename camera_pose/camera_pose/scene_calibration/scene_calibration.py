import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import pylab
import time
import argparse
from scipy.spatial.transform import Rotation as rot


MARKER_LENGTH = 0.05

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-v", "--Visualization", help = "Show Output", default=True)

# Read arguments from command line
args = parser.parse_args()

def load_camera_calibration(calib_file="camera_pose/camera_pose/CamData/rgbd_with_metal_thing.json"):
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

def plotScene(markers, ax, save_markers, marker_info = []):
    """
    Function to plot the scene with the marker 0 at the origin.
    The camera and other markers are plotted relative to marker 0.
    This function only works if marker 0 is detected.
    #TODO: Make it work if marker 0 is not detected. By first recording the position of the markers relative to marker 0 in a way like this:
    Create a program to calieate the scene. For the calibration the user has to show marker 0 and then all other markers.
    The program records the position of all markers relative to marker 0. This program will then use the recordings to compute the positions of the markers
    Parameters:
        markers: List of dictionaries containing a rotation vector and a translation vector
        ax: figure to plot the scene in
    """
    clearFig(ax)
    markerZero = None
    for marker in markers:
        if marker["id"] == 0:
            markerZero = marker
            break

    if markerZero is None:
        print("Marker 0 not found")
        return

    markers = [m for m in markers if m["id"] != 0] # Remove all other markers with id 0
    plotZeroMarker(ax)
    
    T_cam_to_0 = getTransformationMatrix(markerZero)
    T_0_to_cam = np.linalg.inv(T_cam_to_0)

    R_cam = T_0_to_cam[:3, :3]
    tvec_cam = T_0_to_cam[:3, 3]

    xPoint = R_cam @ np.array([0.1, 0, 0]) + tvec_cam
    yPoint = R_cam @ np.array([0, 0.1, 0]) + tvec_cam
    zPoint = R_cam @ np.array([0, 0, 0.1]) + tvec_cam
    ax.scatter(tvec_cam[0], tvec_cam[1], tvec_cam[2], c='black')
    ax.plot([tvec_cam[0], xPoint[0]], [tvec_cam[1], xPoint[1]], [tvec_cam[2], xPoint[2]], c='r')
    ax.plot([tvec_cam[0], yPoint[0]], [tvec_cam[1], yPoint[1]], [tvec_cam[2], yPoint[2]], c='g')
    ax.plot([tvec_cam[0], zPoint[0]], [tvec_cam[1], zPoint[1]], [tvec_cam[2], zPoint[2]], c='b')

    markers_from_0 = []

    for marker in markers:
        T_cam_to_marker = getTransformationMatrix(marker) 
        T_0_to_marker = T_0_to_cam @ T_cam_to_marker

        R_0_to_marker = T_0_to_marker[:3, :3]
        tvec_0_to_marker = T_0_to_marker[:3, 3]

        quat_0_to_marker = rot.from_matrix(R_0_to_marker).as_quat()
        if save_markers: # Find some way to safe for only select frames
            print(f"Saving marker {marker['id']}")
            markers_from_0.append({"id": int(marker["id"]), "tvec": tvec_0_to_marker.tolist(), "quat": quat_0_to_marker.tolist()})

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

    if save_markers and len(markers_from_0) > 0:
        marker_info.append(markers_from_0)
    pylab.pause(0.01)

def processVideo(cap, detector, cam_matrix, dist_coeffs, marker_info):
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
                print("about to vis")
                cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
                print("created window")
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
            plotScene(markers, ax_camera_frame, save_markers, marker_info)

            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("frame", 800, 500)
            cv2.imshow('frame', frame)
        save_markers = False
        key = cv2.waitKey(1)
        if key == 27:
            break  # Exit on ESC
        elif key == 32:
            save_markers = True

def leastSquaresMarkers(marker_info):
    """
    Function to average the marker positions over multiple frames.
    This is a simple implementation, that just averages the positions.
    A more robust implementation could be done using a least squares approach.

    Parameters:
        marker_info: List of lists of dictionaries containing the marker positions
    Returns:
        averaged_markers: List of dictionaries containing the averaged marker positions
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
        avg_quat = np.mean(quats, axis=0)
        avg_quat /= np.linalg.norm(avg_quat)  # Normalize the quaternion
        avg_quat = avg_quat.tolist()
        averaged_markers.append({"id": id, "tvec": avg_tvec, "quat": avg_quat})

    return averaged_markers

def main():
    camera_matrix, dist_coeffs = load_camera_calibration()

    video_capture = cv2.VideoCapture(0) # Now hardcoded as 6, since this is the rgb cam from the rgbd camera

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250) #TODO: Change to 4x4 when changing input
    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    marker_info = []

    processVideo(video_capture, detector, camera_matrix, dist_coeffs, marker_info)

    video_capture.release()
    cv2.destroyAllWindows()

    marker_info = leastSquaresMarkers(marker_info)

    with open('MarkerInfo.json', 'w') as f:
        json.dump(marker_info, f, indent=4)


if __name__ == "__main__":
    main()