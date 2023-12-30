from neural_control.environments.drone_env import QuadRotorEnvBase
import airsim
import time
import numpy as np
from airsim.types import KinematicsState, Vector3r, Quaternionr

def euler_to_quaternion(euler_angles):
    """
    Transform euler angles into quaternion.
	# TODO: Check the rightness of this transformation
    Input: euler_angles: np.array[roll_x, pitch_y, yaw_z]
    Output: np.array[w, x, y, z]
    """
    roll_x, pitch_y, yaw_z = euler_angles

    cy = np.cos(yaw_z * 0.5)
    sy = np.sin(yaw_z * 0.5)
    cr = np.cos(roll_x * 0.5)
    sr = np.sin(roll_x * 0.5)
    cp = np.cos(pitch_y * 0.5)
    sp = np.sin(pitch_y * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return np.array([qw, qx, qy, qz])

def quaternion_to_euler(q):
    """
    Transform quaterion into euler
	# TODO: check if is right
    Input: np.array[w, x, y, z]
    Output: np.array[roll_x, pitch_y, yaw_z]
    """
    t0 = +2.0 * (q.w_val * q.x_val + q.y_val * q.z_val)
    t1 = +1.0 - 2.0 * (q.x_val * q.x_val + q.y_val * q.y_val)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (q.w_val * q.y_val - q.z_val * q.x_val)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
    t4 = +1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val)
    yaw_z = np.arctan2(t3, t4)

    return np.array([roll_x, pitch_y, yaw_z])

class AirsimWrapper(QuadRotorEnvBase):
	"""
	Similar to FlightmareWrapper
	# TODO: assert not unity
	# TODO: check quad config
	# TODO: is the scale of distance in airsim being meter?
	# TODO: What is the use of transform_border function in flightmare.py?
	"""
	def __init__(self, dt):
		super().__init__(None, dt)

		self.client = airsim.MultirotorClient()
		self.client.confirmConnection()
		self.client.enableApiControl(True)   # get control
		self.client.armDisarm(True)          # unlock

	def close(self):
		super().close()
		self.client.armDisarm(False)
		self.client.enableApiControl(False)

	def reset(self):
		"""
		Interface to Airsim reset
		Totally overwrite the reset from base clasee
		"""
		self.client.reset()
		self.client.enableApiControl(True)
		self.client.armDisarm(True)

		airsim_state = self.client.simGetGroundTruthKinematics(vehicle_name = '')

		np_state = self.airsim2apg_state(airsim_state)
		# set own state (current_np_state)
		self._state.from_np(np_state)
		return self._state
	
	def apg2airsim_action(self, action):
		"""
		Input: np.array[]
		Output: np.array[roll, pitch, yaw_rate, throttle]
		# TODO: The action transformation I'm using is totally wrong!!! This is temporary, find a right one!
		"""
		ang_momentum = action[1:]
		ang_momentum_deg_per_s = np.degrees(ang_momentum)

		total_thrust = action[0] * 15 - 7.5 + 9.81
		throttle = 2 * (total_thrust - 9.81) / 15
		return np.concatenate([ang_momentum_deg_per_s, np.array([throttle])])
	
	def airsim2apg_state(self, ori_state):
		"""
		Transform state from the Vector3r form into apg form.
		# TODO: Check the transformation of orientation
		# TODO: Check if the angular_velocity in airsim the same as obs in flightmare
		Input:
			class KinematicsState(MsgpackMixin):
				position = Vector3r()               # 位置
				orientation = Quaternionr()         # 姿态角
				linear_velocity = Vector3r()        # 速度
				angular_velocity = Vector3r()       # 机体角速率
				linear_acceleration = Vector3r()    # 加速度
				angular_acceleration = Vector3r()   # 机体角加速度

		Output:
			np.array[:12]
		"""
		tar_state = np.zeros(12)

		# add position
		tar_state[:3] = ori_state.position.to_numpy_array()
		# add velocity
		tar_state[6:9] = ori_state.linear_velocity.to_numpy_array()
		# add orientation
		tar_state[3:6] = quaternion_to_euler(ori_state.orientation.to_numpy_array())
		# add body rate
		tar_state[9:] = ori_state.angular_velocity.to_numpy_array()

		return tar_state
	
	def apg2airsim_state(self, np_state):
		"""
		Transform state from np array into AirSim KinematicsState form.

		Input:
			np_state: np.array of shape (12,) representing the state
					- np_state[:3]: position
					- np_state[3:6]: orientation in euler angles
					- np_state[6:9]: linear velocity
					- np_state[9:]: body rates (angular velocity)

		Output:
			KinematicsState instance
		"""
		airsim_state = KinematicsState()

		# set position
		airsim_state.position = Vector3r(*np_state[:3])
		# set linear velocity
		airsim_state.linear_velocity = Vector3r(*np_state[6:9])
		# set orientation
		airsim_state.orientation = Quaternionr(*euler_to_quaternion(np_state[3:6]))
		# set angular velocity (body rates)
		airsim_state.angular_velocity = Vector3r(*np_state[9:])

		return airsim_state
	
	def zero_reset(self, position_x=0, position_y=0, position_z=2):
		"""
		set state to given position and zero velocity
		"""
		super().zero_reset(position_x, position_y, position_z) # reset the state
		obs = self.env.zero_reset(position_x, position_y, position_z) # reset the position in airsim 
		airsim_state = self.apg2airsim_state(self._state.as_np)
		self.client.simSetKinematics(airsim_state)
		return self._state.as_np
	
	def step(self, action, thresh=.8, dynamics="airsim"):
		"""
		Overwrite step methods of drone_env
		Use dynamics model implementde in flightmare instead
		"""
		# convert action from model to airsim form
		action = self.apg2airsim_action(action)
		# take action in airsim, and wait until it's done
		self.client.moveByRollPitchYawrateThrottleAsync(*action, duration=self.dt).join()
		# get state and convert to apg form
		airsim_state = self.client.simGetGroundTruthKinematics(vehicle_name = '')

		np_state = self.airsim2apg_state(airsim_state)
		# set own state (current_np_state)
		self._state.from_np(np_state)
		stable = np.all(np.absolute(np_state[3:5]) < thresh)
		return np_state, stable

	
	
"""
 # connect to the AirSim simulator
 client = airsim.MultirotorClient()
 
 client.enableApiControl(True)   # get control
 client.armDisarm(True)          # unlock
 client.takeoffAsync().join()    # takeoff
 
 # square flight
 client.moveToZAsync(-3, 1).join()               # 上升到3m高度
 client.moveToPositionAsync(5, 0, -3, 1).join()  # 飞到（5,0）点坐标
 client.moveToPositionAsync(5, 5, -3, 1).join()  # 飞到（5,5）点坐标
 client.moveToPositionAsync(0, 5, -3, 1).join()  # 飞到（0,5）点坐标
 client.moveToPositionAsync(0, 0, -3, 1).join()  # 回到（0,0）点坐标
 
 client.landAsync().join()       # land
 client.armDisarm(False)         # lock
 client.enableApiControl(False)  # release control
"""