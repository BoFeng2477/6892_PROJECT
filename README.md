# 6892_PROJECT
Bo Feng [bf2477]  Yaning Wang [yw3606]

This is the main code for RL experiment implementation in our project.

We put this experiment tool box together thanks to:

> https://github.com/sfujim/TD3. 
>
> https://github.com/joe-siyuan-qiao/DetectoRS. 
>
> https://github.com/RITCHIEHuang/DeepRL_Algorithms.  

## Environment Preparation

```
Python version – 3.7.4
GCC version - 7.0
Gym version – 0.18.0
Mujoco - v200
mujoco-py - 2.0.2.8
Pytorch - 1.8.1 [ cpu version ]
MacOS Big Sur [1]
Ubuntu-20.04.2.0 [2]
```

Please install the versions of needed libraries as indicated above. 

There's a lot of compatiability issues when working with Mujoco Enviroment, the version combo listed here is acquired after a lot of failed trials, it works on our MacOS and Linux syste, and most likely it'll work on yours, too.

**Please NOTE** that if you have a GCC version higher than 8.0, you have to downgrade it to either 7.0 or 8.0.

## Running Experiments

Simply set the hypyerparameters of the RL algorithm you want to play with in the .py file, and run it by:

```
python SAC.py # Say if you want to experiment SAC
```

The reward data will then be saved in directory **log**

The default key hyper parameters we are using is listed down below:

##### SAC

```
Lr_p: 0.001   
[Policy Net Learning Rate]
Lr_v: 0.001 
[Value Net Learning Rate]
Lr_q: 0.001 
[QValue Net Learning Rate]
Gamma: 0.99 
[discount factor]
Polyak: 0.995 [Interpolation factor]
Min_update_step: 1000
[Min interacts for update]
Update_step:50
[Steps between A/C update]
Target_update_delay: 1
[Target Net update freq]
Explore_size: 10000
Buffer_size: 1000000
Batch_size: 256
```

##### TD3

```
Lr_p: 0.001   
[Policy Net Learning Rate]
Lr_v: 0.001 
[Value Net Learning Rate]
Gamma: 0.99 
[discount factor]
Polyak: 0.995 [Interpolation factor]
Action_noise: 0.2
T_a_N_s: 0.2
[Target action noise std]
T_a_N_c: 0.5
[~ noise clip ratio]
Min_update_step: 1000
[Min interacts for update]
Update_step:50
[Steps between A/C update]
Policy_update_delay: 2
Explore_size: 10000
Buffer_size: 1000000
Batch_size: 256
```

##### PPO

```
  Lr_p: 3e-4   
[Policy Net Learning Rate]
Lr_v: 3e-4 
[Value Net Learning Rate]
Gamma: 0.99 
[discount factor]
tau: 0.995 
[GAE factor]
Epsilon: 0.2 
[clip rate]
PPO epoch: 10 
[Inner loop Updating Actor]

batch_size:4000
```

 Have fun!