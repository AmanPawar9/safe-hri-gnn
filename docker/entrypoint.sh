#!/bin/bash
set -e

# Source ROS 2 and the workspace
source /opt/ros/humble/setup.bash
if [ -f /ros2_ws/install/setup.bash ]; then
  source /ros2_ws/install/setup.bash
fi

# Source the Python virtual environment
if [ -f /ros2_ws/venv/bin/activate ]; then
  source /ros2_ws/venv/bin/activate
fi

exec "$@"