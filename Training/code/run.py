from Training.code import hybrid3D_env
from baselines import logger
import tensorflow as tf
import baselines.common.tf_util as U

import os
import argparse
import tensorflow as tf

def train(num_timesteps, copter_model, model_directory, save_model_interval, save_model_with_prefix,
          restore_model_from_file, play, visualize):
    os.makedirs(model_directory, exist_ok=True)  # Create model directory if it doesn't exist

    from baselines.ppo1 import mlp_policy, pposgd_simple
    tf.compat.v1.disable_eager_execution()

    def policy_fn(name, ob_space, ac_space):
        ob_placeholder = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None] + list(ob_space.shape))
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=64, num_hid_layers=2, gaussian_fixed_var=False)

    train_env = hybrid3D_env.Hybrid3DEnv(data_folder="../data/", config_file=copter_model + ".xml", play=False)
    play_env = hybrid3D_env.Hybrid3DEnv(data_folder="../data/", config_file=copter_model + ".xml",
                                        play=visualize or play)

    pposgd_simple.learn(train_env, policy_fn,
                        max_timesteps=num_timesteps,
                        timesteps_per_actorbatch=4096,
                        clip_param=0.2, entcoeff=0.0,
                        optim_epochs=10,
                        optim_stepsize=3e-4,
                        optim_batchsize=64,
                        gamma=0.995, lam=0.95, schedule='linear',
                        )

    train_env.close()
    play_env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="QuadPlane")
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--controller', type=str, default="")
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()

    model = args.model
    play = args.play
    controller = args.controller
    visualize = args.visualize

    if controller == "":
        controller = None
    else:
        controller = controller + ".ckpt"

    logger.configure()

    train(num_timesteps=30000000,
          copter_model=model,
          model_directory="./results/",
          save_model_interval=5,
          save_model_with_prefix="model",
          restore_model_from_file=controller,
          play=play,
          visualize=visualize)


if __name__ == '__main__':
    main()
