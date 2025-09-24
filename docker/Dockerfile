# Use the official ROS 2 Humble base image
FROM osrf/ros:humble-desktop

# Set shell and environment
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    cmake \
    build-essential \
    git \
    wget \
    curl \
    # ROS 2 packages
    ros-humble-moveit \
    ros-humble-moveit-resources \
    ros-humble-moveit-planners-ompl \
    ros-humble-moveit-servo \
    ros-humble-realsense2-camera \
    ros-humble-realsense2-description \
    ros-humble-joint-state-publisher \
    ros-humble-robot-state-publisher \
    ros-humble-xacro \
    ros-humble-tf2-tools \
    ros-humble-rviz2 \
    # Development tools
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

# Create a ROS 2 workspace
WORKDIR /ros2_ws

# Copy the project source code into the workspace's src directory
COPY ./src /ros2_ws/src

# Copy requirements file and install python dependencies into a venv
COPY requirements.txt /ros2_ws/requirements.txt
RUN python3 -m venv /ros2_ws/venv
RUN source /ros2_ws/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r /ros2_ws/requirements.txt

# Initialize rosdep if not already done
RUN if [ ! -f "/etc/ros/rosdep/sources.list.d/20-default.list" ]; then \
        rosdep init; \
    fi && \
    rosdep update

# Install ROS dependencies
RUN source /opt/ros/humble/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -y

# Set up entrypoint
COPY ./docker/entrypoint.sh /
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]