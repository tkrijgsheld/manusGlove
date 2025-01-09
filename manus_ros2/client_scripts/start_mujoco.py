import mujoco
import mujoco.viewer
import numpy as np
import time
from typing import Dict
from robot_descriptions import shadow_hand_mj_description


class ShadowHandSimulator:
    def __init__(self, model_path: str):
        # Load the model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Create visualization context
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Get all joint names from the model
        joint_names = []
        for i in range(self.model.njnt):
            name = self.model.names[self.model.name_jntadr[i] :].decode().split("\x00")[0]
            joint_names.append(name)

        # Create a mapping of joint names to their IDs
        self.joint_ids = {name: i for i, name in enumerate(joint_names)}

        # Print all joint names
        print("Available joints:")
        for joint_name in joint_names:
            print(f"- {joint_name}")

    def set_joint_positions(self, joint_positions: Dict[str, float]):
        """
        Set joint positions using a dictionary of joint names and their target positions

        Args:
            joint_positions: Dictionary with joint names as keys and target positions (in radians) as values
        """
        for joint_name, position in joint_positions.items():
            if joint_name in self.joint_ids:
                joint_id = self.joint_ids[joint_name]
                self.data.qpos[joint_id] = position
            else:
                print(f"Warning: Joint {joint_name} not found in model")

        mujoco.mj_forward(self.model, self.data)

    def simulate(self):
        """
        Run the simulation for the specified duration
        """
        while self.viewer.is_running():
            # Example: Set some joint positions
            test_positions = {
                # "lh_RFJ4": -0.2,
                # "lh_RFJ3": 5.0,
                # "lh_RFJ2": 5.0,
                # "lh_RFJ1": 5.0,
            }

            # Set the joint positions
            self.set_joint_positions(test_positions)
            mujoco.mj_step(self.model, self.data)

            ff_middle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "lh_ffmiddle")
            ff_proximal_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "lh_ffproximal")

            self.viewer.user_scn.ngeom = 0
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.005, 0, 0],
                pos=self.data.xpos[ff_middle_id],
                mat=np.eye(3).flatten(),
                rgba=[1, 0, 0, 1],
            )
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[1],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.005, 0, 0],
                pos=self.data.xpos[ff_proximal_id],
                mat=np.eye(3).flatten(),
                rgba=[0, 1, 0, 1],
            )
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[2],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.005, 0, 0],
                pos=self.data.site("lh_ffeffector").xpos,
                mat=np.eye(3).flatten(),
                rgba=[0.5, 0.2, 1, 1],
            )
            self.viewer.user_scn.ngeom = 3

            # self.model.body()

            ee_pos = np.array(self.data.site("lh_ffeffector").xpos)
            target_pos = np.array([0.37570832, 0.033, 0.03738391])
            print(f"ee_pos: {ee_pos}; target_pos: {target_pos}; diff: {ee_pos - target_pos}")

            self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            self.viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE

            self.viewer.sync()

    def close(self):
        """Close the viewer"""
        self.viewer.close()


def main():
    # Initialize the simulator
    simulator = ShadowHandSimulator("./mujoco_menagerie/shadow_hand/manus_demo.xml")

    # Run simulation
    try:
        simulator.simulate()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
