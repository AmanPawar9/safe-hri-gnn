# Represents the predicted future trajectory of all human joints
std_msgs/Header header
# A sequence of future poses
# The length of this array corresponds to the prediction horizon (e.g., 45 frames)
hrc_msgs/PoseGraph[] trajectory