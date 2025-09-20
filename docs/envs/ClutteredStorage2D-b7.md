# prbench/ClutteredStorage2D-b7-v0
![random action GIF](assets/random_action_gifs/ClutteredStorage2D-b7.gif)

### Description
A 2D environment where the goal is to put all blocks inside a shelf.

There are always 7 blocks in this environment.

The robot has a movable circular base and a retractable arm with a rectangular vacuum end effector. Objects can be grasped and ungrasped when the end effector makes contact.

### Initial State Distribution
![initial state GIF](assets/initial_state_gifs/ClutteredStorage2D-b7.gif)

### Example Demonstration
![demo GIF](assets/demo_gifs/ClutteredStorage2D-b7/ClutteredStorage2D-b7_seed1_1752340825.gif)

### Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | robot | x |
| 1 | robot | y |
| 2 | robot | theta |
| 3 | robot | base_radius |
| 4 | robot | arm_joint |
| 5 | robot | arm_length |
| 6 | robot | vacuum |
| 7 | robot | gripper_height |
| 8 | robot | gripper_width |
| 9 | shelf | x |
| 10 | shelf | y |
| 11 | shelf | theta |
| 12 | shelf | static |
| 13 | shelf | color_r |
| 14 | shelf | color_g |
| 15 | shelf | color_b |
| 16 | shelf | z_order |
| 17 | shelf | width |
| 18 | shelf | height |
| 19 | shelf | x1 |
| 20 | shelf | y1 |
| 21 | shelf | theta1 |
| 22 | shelf | width1 |
| 23 | shelf | height1 |
| 24 | shelf | z_order1 |
| 25 | shelf | color_r1 |
| 26 | shelf | color_g1 |
| 27 | shelf | color_b1 |
| 28 | block0 | x |
| 29 | block0 | y |
| 30 | block0 | theta |
| 31 | block0 | static |
| 32 | block0 | color_r |
| 33 | block0 | color_g |
| 34 | block0 | color_b |
| 35 | block0 | z_order |
| 36 | block0 | width |
| 37 | block0 | height |
| 38 | block1 | x |
| 39 | block1 | y |
| 40 | block1 | theta |
| 41 | block1 | static |
| 42 | block1 | color_r |
| 43 | block1 | color_g |
| 44 | block1 | color_b |
| 45 | block1 | z_order |
| 46 | block1 | width |
| 47 | block1 | height |
| 48 | block2 | x |
| 49 | block2 | y |
| 50 | block2 | theta |
| 51 | block2 | static |
| 52 | block2 | color_r |
| 53 | block2 | color_g |
| 54 | block2 | color_b |
| 55 | block2 | z_order |
| 56 | block2 | width |
| 57 | block2 | height |
| 58 | block3 | x |
| 59 | block3 | y |
| 60 | block3 | theta |
| 61 | block3 | static |
| 62 | block3 | color_r |
| 63 | block3 | color_g |
| 64 | block3 | color_b |
| 65 | block3 | z_order |
| 66 | block3 | width |
| 67 | block3 | height |
| 68 | block4 | x |
| 69 | block4 | y |
| 70 | block4 | theta |
| 71 | block4 | static |
| 72 | block4 | color_r |
| 73 | block4 | color_g |
| 74 | block4 | color_b |
| 75 | block4 | z_order |
| 76 | block4 | width |
| 77 | block4 | height |
| 78 | block5 | x |
| 79 | block5 | y |
| 80 | block5 | theta |
| 81 | block5 | static |
| 82 | block5 | color_r |
| 83 | block5 | color_g |
| 84 | block5 | color_b |
| 85 | block5 | z_order |
| 86 | block5 | width |
| 87 | block5 | height |
| 88 | block6 | x |
| 89 | block6 | y |
| 90 | block6 | theta |
| 91 | block6 | static |
| 92 | block6 | color_r |
| 93 | block6 | color_g |
| 94 | block6 | color_b |
| 95 | block6 | z_order |
| 96 | block6 | width |
| 97 | block6 | height |


### Action Space
The entries of an array in this Box space correspond to the following action features:
| **Index** | **Feature** | **Description** | **Min** | **Max** |
| --- | --- | --- | --- | --- |
| 0 | dx | Change in robot x position (positive is right) | -0.050 | 0.050 |
| 1 | dy | Change in robot y position (positive is up) | -0.050 | 0.050 |
| 2 | dtheta | Change in robot angle in radians (positive is ccw) | -0.196 | 0.196 |
| 3 | darm | Change in robot arm length (positive is out) | -0.100 | 0.100 |
| 4 | vac | Directly sets the vacuum (0.0 is off, 1.0 is on) | 0.000 | 1.000 |


### Rewards
A penalty of -1.0 is given at every time step until termination, which occurs when all blocks are inside the shelf.


### References
Similar environments have been considered by many others, especially in the task and motion planning literature.
