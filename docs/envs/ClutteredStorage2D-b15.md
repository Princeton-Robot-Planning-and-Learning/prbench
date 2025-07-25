# prbench/ClutteredStorage2D-b15-v0
![random action GIF](assets/random_action_gifs/ClutteredStorage2D-b15.gif)

### Description
A 2D environment where the goal is to put all blocks inside a shelf.

There are always 15 blocks in this environment.

The robot has a movable circular base and a retractable arm with a rectangular vacuum end effector. Objects can be grasped and ungrasped when the end effector makes contact.

### Initial State Distribution
![initial state GIF](assets/initial_state_gifs/ClutteredStorage2D-b15.gif)

### Example Demonstration
![demo GIF](assets/demo_gifs/ClutteredStorage2D-b15.gif)

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
| 19 | block0 | x |
| 20 | block0 | y |
| 21 | block0 | theta |
| 22 | block0 | static |
| 23 | block0 | color_r |
| 24 | block0 | color_g |
| 25 | block0 | color_b |
| 26 | block0 | z_order |
| 27 | block0 | width |
| 28 | block0 | height |
| 29 | block1 | x |
| 30 | block1 | y |
| 31 | block1 | theta |
| 32 | block1 | static |
| 33 | block1 | color_r |
| 34 | block1 | color_g |
| 35 | block1 | color_b |
| 36 | block1 | z_order |
| 37 | block1 | width |
| 38 | block1 | height |
| 39 | block10 | x |
| 40 | block10 | y |
| 41 | block10 | theta |
| 42 | block10 | static |
| 43 | block10 | color_r |
| 44 | block10 | color_g |
| 45 | block10 | color_b |
| 46 | block10 | z_order |
| 47 | block10 | width |
| 48 | block10 | height |
| 49 | block11 | x |
| 50 | block11 | y |
| 51 | block11 | theta |
| 52 | block11 | static |
| 53 | block11 | color_r |
| 54 | block11 | color_g |
| 55 | block11 | color_b |
| 56 | block11 | z_order |
| 57 | block11 | width |
| 58 | block11 | height |
| 59 | block12 | x |
| 60 | block12 | y |
| 61 | block12 | theta |
| 62 | block12 | static |
| 63 | block12 | color_r |
| 64 | block12 | color_g |
| 65 | block12 | color_b |
| 66 | block12 | z_order |
| 67 | block12 | width |
| 68 | block12 | height |
| 69 | block13 | x |
| 70 | block13 | y |
| 71 | block13 | theta |
| 72 | block13 | static |
| 73 | block13 | color_r |
| 74 | block13 | color_g |
| 75 | block13 | color_b |
| 76 | block13 | z_order |
| 77 | block13 | width |
| 78 | block13 | height |
| 79 | block14 | x |
| 80 | block14 | y |
| 81 | block14 | theta |
| 82 | block14 | static |
| 83 | block14 | color_r |
| 84 | block14 | color_g |
| 85 | block14 | color_b |
| 86 | block14 | z_order |
| 87 | block14 | width |
| 88 | block14 | height |
| 89 | block2 | x |
| 90 | block2 | y |
| 91 | block2 | theta |
| 92 | block2 | static |
| 93 | block2 | color_r |
| 94 | block2 | color_g |
| 95 | block2 | color_b |
| 96 | block2 | z_order |
| 97 | block2 | width |
| 98 | block2 | height |
| 99 | block3 | x |
| 100 | block3 | y |
| 101 | block3 | theta |
| 102 | block3 | static |
| 103 | block3 | color_r |
| 104 | block3 | color_g |
| 105 | block3 | color_b |
| 106 | block3 | z_order |
| 107 | block3 | width |
| 108 | block3 | height |
| 109 | block4 | x |
| 110 | block4 | y |
| 111 | block4 | theta |
| 112 | block4 | static |
| 113 | block4 | color_r |
| 114 | block4 | color_g |
| 115 | block4 | color_b |
| 116 | block4 | z_order |
| 117 | block4 | width |
| 118 | block4 | height |
| 119 | block5 | x |
| 120 | block5 | y |
| 121 | block5 | theta |
| 122 | block5 | static |
| 123 | block5 | color_r |
| 124 | block5 | color_g |
| 125 | block5 | color_b |
| 126 | block5 | z_order |
| 127 | block5 | width |
| 128 | block5 | height |
| 129 | block6 | x |
| 130 | block6 | y |
| 131 | block6 | theta |
| 132 | block6 | static |
| 133 | block6 | color_r |
| 134 | block6 | color_g |
| 135 | block6 | color_b |
| 136 | block6 | z_order |
| 137 | block6 | width |
| 138 | block6 | height |
| 139 | block7 | x |
| 140 | block7 | y |
| 141 | block7 | theta |
| 142 | block7 | static |
| 143 | block7 | color_r |
| 144 | block7 | color_g |
| 145 | block7 | color_b |
| 146 | block7 | z_order |
| 147 | block7 | width |
| 148 | block7 | height |
| 149 | block8 | x |
| 150 | block8 | y |
| 151 | block8 | theta |
| 152 | block8 | static |
| 153 | block8 | color_r |
| 154 | block8 | color_g |
| 155 | block8 | color_b |
| 156 | block8 | z_order |
| 157 | block8 | width |
| 158 | block8 | height |
| 159 | block9 | x |
| 160 | block9 | y |
| 161 | block9 | theta |
| 162 | block9 | static |
| 163 | block9 | color_r |
| 164 | block9 | color_g |
| 165 | block9 | color_b |
| 166 | block9 | z_order |
| 167 | block9 | width |
| 168 | block9 | height |


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
