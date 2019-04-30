# IERG6130-project
# Overview
Due to the variation of the wave surface, the light propagation changes rapidly in the air-water interface and misalignment between the transmitter and receiver will degrade the communication performance severely.  In the air-water optical wireless communication (OWC) system, beam steering is crucial for establishing a reliable link in the air-water interface. In this project, a 2-dimensional (2D) microelectromechanical systems (MEMS) mirror is used at the transmitter side to adjust the light propagation path based on the power of the reflected signal. Reinforcement learning technique is applied to automatically adjust the tilt angles of x-axis and y-axis of the MEMS mirror, so that the light will transmit properly to the receiver side no matter how the wave surface changes. Since the wave surface varies rapidly and shows random behaviour, model free reinforcement learning based beam steering will be investigated to maintain a stable transmission link in the air-water OWC system.
Here, to verify the effectiveness of the on-policy and off-policy algorithms in the auto-alignment system, we apply a policy network and DDPG in the experiment for comparison.
# Dependencies
Python 3.6.4; Pytorch 0.2.0 
# Description
On_policy.py: An on-policy network is employed to control the MEMS mirror.     
DDPG_off_policy.py: DDPG is applied to control the MEMS mirror.      
Wave.py: The wave environment for DDPG. 
# Environment
For on-policy network, the environment is included in the On_policy.py.         
For the DDPG implementation, the environment is generated in Wave.py. 
# Reference
The DDPG implementation is developed based on the following algorithm.
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/9_Deep_Deterministic_Policy_Gradient_DDPG
