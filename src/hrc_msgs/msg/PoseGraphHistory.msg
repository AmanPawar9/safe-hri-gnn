# Represents a time-series of human poses for prediction
std_msgs/Header header
# A sequence of poses, where each pose is a list of joints
# The length of this array should correspond to the input sequence length (e.g., 30)
hrc_msgs/PoseGraph[] history