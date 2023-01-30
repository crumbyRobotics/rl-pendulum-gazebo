#!/usr/bin/env python3
from cmath import isnan
import time
import torch
import numpy as np

import rospy

from ur5_env import UR5Env
from sac import SAC


# 訓練用コントローラー
def train():
    # gpuが使用される場合の設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # gazebo上のur5用env
    env = UR5Env(device)

    # ハイパーパラメータ
    buffer_size = 1000  # Experienceのキュー容量
    warmup_size = 500  # 学習するかどうかのExperienceの最低限の容量
    train_interval = 10  # 学習する制御周期間隔
    batch_size = 32  # バッチサイズ
    gamma = 0.9  # 割引率
    soft_target_tau = 0.02  # Soft TargetでTargetに近づく割合
    encoder_soft_target_tau = 0.05  # Qのτよりも大きい数値
    hard_target_interval = 100  # Hard Targetで同期する間隔
    lr = 0.001
    target_entropy = -1 * env.action_space.shape[0]  # エントロピーαの目標値: -1xアクション数がいいらしい

    state_dim = 50  # 画像から抽出される特徴量の次元
    obs_shape = torch.stack([torch.cat([env.get_obs(), env.get_obs(), env.get_obs()])]).shape  # [1, 3, 3, 40, 92]
    num_layers = 2
    num_filters = 32
    decoder_latent_lambda = 0.0

    sac = SAC(device, state_dim, env.action_space, obs_shape, num_layers, num_filters, decoder_latent_lambda,
              buffer_size, gamma, soft_target_tau, encoder_soft_target_tau, target_entropy, lr, lr, lr)

    step_count = 0
    train_count = 0

    # 記録用
    history_rewards = []
    history_metrics = []
    history_metrics_y = []

    # 学習ループ
    for episode in range(5000):
        env.reset()  # gazeboをリセット
        obs, reward, done = env.update()   # 初期状態を取得
        prev_obs = obs
        prev_prev_obs = obs
        state = torch.stack([torch.cat([prev_prev_obs, prev_obs, obs])])
        action = 0
        total_reward = 0

        step = 0

        metrics_list = []

        r = rospy.Rate(10)  # 制御周期10Hz（ミニバッチによる訓練が100ms以下で終わる必要がある）

        # １エピソード
        while not done:
            obs, reward, done = env.update()  # 最新のセンサデータ、報酬等をsubscribe

            n_state = torch.stack([torch.cat([prev_prev_obs, prev_obs, obs])])
            step += 1
            total_reward += reward

            sac.replay_memory.push(
                state.clone().detach().to(device),
                torch.tensor(action, dtype=torch.float32).reshape(1, -1).to(device),
                n_state.clone().detach().to(device),
                torch.tensor(reward, dtype=torch.float32).reshape(1, -1).to(device),
                torch.tensor(done, dtype=torch.float32).reshape(1, -1).to(device))

            state = n_state

            # train_interval毎に, warmup貯まっていたら学習する
            if len(sac.replay_memory) >= warmup_size and step_count % train_interval == 0:
                q_net_sync = False
                if train_count % hard_target_interval == 0:
                    q_net_sync = True
                # モデルの更新
                metrics = sac.update(
                    batch_size,
                    q_net_sync)

                train_count += 1
                metrics_list.append(metrics)

            step_count += 1

            # アクションを決定・publish
            env_action, action = sac.sample_action_for_env(state)
            if isnan(env_action[0]):
                print("action is NaN. 学習失敗.")
                break

            env.step(env_action)

            # 前回・前々回のstateを保存
            prev_prev_obs = prev_obs
            prev_obs = obs

            r.sleep()

        # 1エピソードの合計報酬
        history_rewards.append(total_reward)

        # 1エピソードのメトリクス
        if len(metrics_list) > 0:
            history_metrics.append(np.mean(metrics_list, axis=0))  # 平均を保存
            history_metrics_y.append(episode)

        # --- print
        interval = 5
        if episode % interval == 0:
            print("{} (min,ave,max)reward {} {} {}, alpha={}".format(
                episode,
                np.min(history_rewards[-interval:]),
                np.mean(history_rewards[-interval:]),
                np.max(history_rewards[-interval:]),
                torch.exp(sac.log_alpha).cpu().detach().numpy()[0]
            ))


if __name__ == '__main__':
    try:
        train()
    except rospy.ROSInterruptException:
        pass
