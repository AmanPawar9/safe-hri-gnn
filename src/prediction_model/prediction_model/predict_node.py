import rclpy
from rclpy.node import Node
import numpy as np
import torch
import os

from hrc_msgs.msg import PoseGraphHistory, PoseGraph, PredictedTrajectory
from geometry_msgs.msg import Point
from ament_index_python.packages import get_package_share_directory

from .model import STGNNModel, predict_future_poses

class PredictionNode(Node):
    def __init__(self):
        super().__init__('prediction_node')
        self.get_logger().info("Human Motion Prediction Node started.")

        # --- Parameters ---
        model_name = self.declare_parameter('model_name', 'stgnn_model.pth').get_parameter_value().string_value
        
        # --- Load Model ---
        model_path = os.path.join(
            get_package_share_directory('prediction_model'),
            'models',
            model_name
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = STGNNModel().to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.get_logger().info(f"Successfully loaded model from {model_path}")
        except FileNotFoundError:
            self.get_logger().error(f"Model file not found at {model_path}. Please train a model first.")
            raise

        # --- Subscribers & Publishers ---
        self.subscription = self.create_subscription(
            PoseGraphHistory,
            '/pose_graph_history',
            self.listener_callback,
            10
        )
        self.publisher = self.create_publisher(PredictedTrajectory, '/predicted_human_trajectory', 10)

    def listener_callback(self, msg: PoseGraphHistory):
        # 1. Convert ROS message to Python list format for the model
        pose_history = []
        for pose_graph in msg.history:
            frame = [(p.x, p.y, p.z) for p in pose_graph.joint_positions]
            pose_history.append(frame)

        # 2. Run prediction
        predicted_frames = predict_future_poses(self.model, pose_history)
        
        # 3. Convert prediction back to ROS message
        traj_msg = PredictedTrajectory()
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        traj_msg.header.frame_id = msg.history[-1].header.frame_id # Use last known frame_id

        for frame_data in predicted_frames:
            pose_graph = PoseGraph()
            pose_graph.header.stamp = traj_msg.header.stamp # Should increment this in a real scenario
            pose_graph.joint_positions = [Point(x=p[0], y=p[1], z=p[2]) for p in frame_data]
            traj_msg.trajectory.append(pose_graph)
            
        # 4. Publish the result
        self.publisher.publish(traj_msg)
        # self.get_logger().info("Published a predicted trajectory.")


def main(args=None):
    rclpy.init(args=args)
    node = PredictionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()