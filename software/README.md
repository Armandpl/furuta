# Software Instructions

## 1. Install Jetpack on your Jetson Nano
Jetpack is an image with Deep Learning related libs preinstalled.
Follow the instructions at [https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit)

## 2. Setup a virtualenv
I recommend you use a virtual env to isolate this project dependencies from your system.  
```bash
sudo apt install -y python3-pip python3-venv
python3 -m venv ~/python-envs/frt
source ~/python-envs/frt/bin/activate
```

## 3. Install torch
Follow the instructions at [https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available)

## 4. Install Jetson.GPIO
Follow instructions at [https://pypi.org/project/Jetson.GPIO/](https://pypi.org/project/Jetson.GPIO/)

## 5. Install remaining dependencies
`pip install -r requirement.txt`

## 6. Activate the SPI bus and PWM pins
Use the [Jetson IO utility](https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/hw_setup_jetson_io.html) to activate the SPI bus and one PWM pin to control the motor.

## 7. Train/Deploy

You can deploy the train model with:  
`python robot_inference.py v204`  

Or train a new model with:  
`python sac_train.py --action_limiter True --continuity_cost True --history 2 --state_limits low --train_freq "1 episode" --gym_id FurutaReal-v0 --episode_length 1000 --fs 50 --fs_ctrl 50 --learning_starts 500 --sde_sample_freq 64`  

**Note**: If training on the physical robot, make sure the train_freq is "n episode". We can only turn off the robot between episodes, if you try training every n steps the robot while continue spinning while we update the policy.
