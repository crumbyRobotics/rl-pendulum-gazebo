import rospy
import ros_numpy
from std_srvs.srv import Empty
from sensor_msgs.msg import Image

from ur5_msgs.msg import Action
from ur5_msgs.msg import State

from gym.spaces import Box

import numpy as np
import torch
import torchvision.transforms as T


class UR5Env():
    def __init__(self, torch_device):
        rospy.init_node('ur5_train', anonymous=True)

        rospy.wait_for_service('/gazebo/reset_simulation')

        self.action_pub = rospy.Publisher('/ur5_action', Action, queue_size=10)
        self.obs_pub = rospy.Publisher('/ur5_obs', Image, queue_size=10)

        self.state_sub = rospy.Subscriber('/ur5_state', State, self.state_callback)
        self.image_sub = rospy.Subscriber('/ur5/camera1/image_raw', Image, self.image_callback)

        self.reset_srv = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.state = State()
        self.action = Action()

        self.obs = None
        self.reward = 0
        self.done = False

        self.timestep = 0

        lower_action = np.array([-5.])
        upper_action = np.array([5.])
        self.action_space = Box(np.float32(lower_action), np.float32(upper_action))

        # 状態が3つの時で上限と下限の設定と仮定
        lower_state = np.array([-1.9])
        upper_state = np.array([1.9])
        self.observation_space = Box(np.float32(lower_state), np.float32(upper_state))

        self.device = torch_device

    def state_callback(self, state):
        # stateの更新
        self.state = state

        # rewardの計算
        self.reward = - (self.state.angle1 ** 2 + 0.1 * self.state.angularvel1 ** 2 + 0.001 * self.action.torque1 ** 2)

        # doneの判定
        if abs(self.state.angle1) > 0.8:
            self.done = True

    def image_callback(self, image):
        # obsの計算
        screen = ros_numpy.numpify(image).transpose(2, 0, 1)  # PyTorch標準のCHWに変換 (ndarray, [0,255])
        screen_height = image.height
        screen_width = image.width

        # sliceでscreenをトリミング
        screen = screen[:, int(screen_height * 0.11):int(screen_height * 0.89), int(screen_width * 0.1):int(screen_width * 0.90)]

        # float32に変換し0-1に正規化
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        # Tensorに変換
        screen = torch.from_numpy(screen)

        resize = T.Compose([T.ToPILImage(), T.Resize(40, T.InterpolationMode.BICUBIC), T.ToTensor()])
        self.obs = resize(screen).unsqueeze(0).to(self.device)
        self.obs_pub.publish(ros_numpy.msgify(Image, np.ascontiguousarray(
            self.obs.cpu().numpy()[0].transpose(1, 2, 0) * 255, dtype=np.uint8), encoding='rgb8'))

    def reset(self):
        self.step(np.float32(0))
        self.reset_srv()
        self.done = False
        self.timestep = 0

    def step(self, action_):
        action = Action()
        action.torque1 = action_
        self.action_pub.publish(action)

        self.action = action

    def update(self):
        self.timestep = self.timestep + 1
        if self.timestep > 200:
            self.done = True
        return self.obs, self.reward, self.done

    def get_obs(self):
        while self.obs == None:
            pass
        return self.obs
