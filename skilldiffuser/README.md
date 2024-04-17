# SkillDiffuser

Zhixuan Liang, Yao Mu, Hengbo Ma, Masayoshi Tomizuka, Mingyu Ding, Ping Luo

## Usage
### Setup Python Environment
1. Install MuJoCo 200
```shell
unzip mujoco200_linux.zip
mv mujoco200_linux mujoco200
cp mjkey.txt ~/.mujoco
cp mjkey.txt ~/.mujoco/mujoco200/bin

# test the install
cd ~/.mujoco/mujoco200/bin
./simulate ../model/humanoid.xml

# add environment variables
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
export MUJOCO_KEY_PATH=~/.mujoco/${MUJOCO_KEY_PATH}
```
2. Install Pypi Packages
```shell
pip install -r requirements.txt
```
3. Install LOReL Environment
```shell
git clone https://github.com/suraj-nair-1/lorel.git

cd lorel/env
pip install -e .
```

## Instructions

Our code for running SkillDiffuser experiments is present in `hrl` folder.

To run the code, please use the following command:

`./train_lorel_compose.sh`

This is a sample command intended to show the usage of different flags available. The checkpoints can be downloaded from [here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/liangzx_connect_hku_hk/Em3qBc3AxWpOkYR7Pgd7lnUBH0bkLsILMpgUX2Xg5l3YGg?e=QslO6n).

## License

The code is made available for academic, non-commercial usage.

For any inquiry, contact: Zhixuan Liang (liangzx@connect.hku.hk)
