import rospy
from std_srvs.srv import Empty
from sensor_msgs import Image

from ur5_msgs import Action
from ur5_msgs import State

from gym.spaces import Box

import numpy as np
import torch
import torchvision.transforms as T


class UR5Env():
    def __init__(self, torch_device):
        rospy.init_node('ur5_train', anonymous=True)

        rospy.wait_for_service('/gazebo/reset_simulation')

        self.action_pub = rospy.Publisher('/ur5_action', Action, queue_size=10)
        self.state_sub = rospy.Subscriber('/ur5_state', State, self.state_callback)
        self.obs_sub = rospy.Subscriber('/ur5/camera1/image_raw', Image, self.image_callback)

        rospy.spin()  # rospyでは別スレッドで常にsubscribeしてくれる

        self.state = 0
        self.obs = 0
        self.reward = 0
        self.done = False

        lower_action = [-5]
        upper_action = [5]
        self.action_space = Box(low=lower_action, high=upper_action)

        # 状態が3つの時で上限と下限の設定と仮定
        lower_state = [-1.9]
        upper_state = [1.9]
        self.observation_space = Box(low=lower_state, high=upper_state)

        self.device = torch_device

    def state_callback(self, state):
        # stateの更新
        self.state = state

        # rewardの計算
        self.reward = 0

        # doneの判定
        self.done = False

    def image_callback(self, image):
        # obsの計算
        screen = image.data.transpose(2, 0, 1)  # PyTorch標準のCHWに変換
        screen_height = image.height
        screen_width = image.width

        # sliceでscreenをトリミング
        height_slice = slice(int(screen_height * 0.1), int(screen_height * 0.7))
        screen = screen[:, height_slice]
        width_slice = slice(int(screen_width * 0.11), int(screen_width * 0.89))
        screen = screen[:, :, width_slice]

        # float32に変換し0-1に正規化
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        # Tensorに変換
        screen = torch.from_numpy(screen)

        resize = T.Compose([T.ToPILImage(), T.Resize(40, T.InterpolationMode.BICUBIC), T.ToTensor()])
        self.obs = resize(screen).unsqueeze(0).to(self.device)

    def reset(self):
        rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

    def step(self, action_):
        action = Action()
        action.torque1 = action_
        self.action_pub.publish(action)

    def update(self):
        return self.obs, self.reward, self.done
