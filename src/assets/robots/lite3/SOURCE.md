Lite3 MJCF and STL assets originate from
https://github.com/DeepRoboticsLab/deep_robotics_model at commit
75824b516ecc8fa3f8f7ce5d6577e1e86d6b614f.
The complete upstream BSD 3-Clause license is retained in LICENSE.txt.

The vendor demo scene is adapted at runtime in src/tasks/pie/robots.py:
its floor, lights, demo cameras, actuators and sensors are removed; valid
foot inertia, collision masks, foot sites, PIE actuators and the depth camera
are added. The copied XML and STL files are retained as the source assets.
No assets are imported from the separate Isaac Gym project.
