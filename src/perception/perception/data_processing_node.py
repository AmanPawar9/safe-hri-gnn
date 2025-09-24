import rclpy
from rclpy.node import Node
from collections import deque
import numpy as np

# Assuming input from a body tracking system like NVIDIA Isaac ROS
# For this example, we'll use a placeholder message type.
# In a real system, you'd use the message from the actual body tracking pipeline.
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point

from hrc_msgs.msg import PoseGraph, PoseGraphHistory
from .kalman_filter import KalmanFilter

NUM_JOINTS = 34 # As per PRD
INPUT_SEQ_LEN = 30 # 1 second at 30 FPS

class DataProcessingNode(Node):
    def __init__(self):
        super().__init__('data_processing_node')
        self.get_logger().info("Data Processing Node started.")

        # --- Parameters ---
        self.input_topic = self.declare_parameter('input_topic', '/body_tracking_data').get_parameter_value().string_value

        # --- Data Structures ---
        self.pose_history_buffer = deque(maxlen=INPUT_SEQ_LEN)
        self.kalman_filters = [KalmanFilter() for _ in range(NUM_JOINTS)]

        # --- Subscribers & Publishers ---
        # NOTE: Subscribing to MarkerArray for demonstration purposes. Replace with your actual
        # skeleton message type (e.g., isaac_ros_body_track_msgs/BodyTrack).
        self.skeleton_sub = self.create_subscription(
            MarkerArray,
            self.input_topic,
            self.skeleton_callback,
            10
        )
        self.history_pub = self.create_publisher(PoseGraphHistory, '/pose_graph_history', 10)

    def skeleton_callback(self, msg: MarkerArray):
        """
        Processes raw skeleton data, applies filtering, and constructs pose graphs.
        This callback is a placeholder for a real implementation that would parse
        a message like `isaac_ros_body_track_msgs/BodyTrack`.
        """
        # --- 1. Skeleton Extraction (Placeholder) ---
        # This part simulates extracting joint data from the incoming message.
        # We assume `msg.markers` contains points representing the joints.
        # In a real scenario, you'd have a more direct way to get joint coordinates.
        
        # Check if we have enough markers
        if not msg.markers or len(msg.markers) == 0:
            self.get_logger().warn("Received empty MarkerArray. Skipping frame.")
            return
            
        # Extract joint positions from the first marker's points
        marker = msg.markers[0]
        if len(marker.points) < NUM_JOINTS:
            self.get_logger().warn(f"Received only {len(marker.points)} points, expected {NUM_JOINTS}. Skipping frame.")
            return

        current_joints_raw = np.array([[p.x, p.y, p.z] for p in marker.points[:NUM_JOINTS]])
        
        # --- 2. Data Filtering ---
        smoothed_joints = np.zeros_like(current_joints_raw)
        for i in range(NUM_JOINTS):
            measurement = current_joints_raw[i, :].reshape(3, 1)
            smoothed_joints[i, :] = self.kalman_filters[i].process(measurement)
        
        # --- 3. Graph Construction ---
        pose_graph = PoseGraph()
        pose_graph.header.stamp = self.get_clock().now().to_msg()
        pose_graph.header.frame_id = "camera_link" # Or your relevant frame
        pose_graph.joint_positions = [Point(x=p[0], y=p[1], z=p[2]) for p in smoothed_joints]

        # --- 4. Buffering ---
        self.pose_history_buffer.append(pose_graph)

        # --- 5. Output Publication ---
        if len(self.pose_history_buffer) == INPUT_SEQ_LEN:
            history_msg = PoseGraphHistory()
            history_msg.header.stamp = self.get_clock().now().to_msg()
            history_msg.history = list(self.pose_history_buffer)
            self.history_pub.publish(history_msg)
            # self.get_logger().info("Published pose graph history.")


def main(args=None):
    rclpy.init(args=args)
    node = DataProcessingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()