import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from threading import Lock

from moveit.planning import PlanningSceneMonitor
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from hrc_msgs.msg import PredictedTrajectory

class MotionPlannerNode(Node):
    def __init__(self):
        super().__init__('motion_planner_node')
        self.get_logger().info("Motion Planner Node started.")

        # --- Parameters ---
        self.robot_description = self.declare_parameter(
            'robot_description', 'robot_description'
        ).get_parameter_value().string_value

        # --- MoveIt Planning Scene ---
        self.planning_scene_monitor = PlanningSceneMonitor(self, self.robot_description)
        # Short delay to allow the planning scene monitor to connect
        self.get_clock().sleep_for(Duration(seconds=1.0).to_msg())
        self.planning_scene_publisher = self.create_publisher(CollisionObject, '/collision_object', 10)
        self.scene_lock = Lock()
        
        # Define key body parts to model as collision objects
        # The indices correspond to the joint indices from the skeleton model.
        self.body_parts = {
            "torso": ([0, 1, 2, 13, 10], 0.15),  # Joints and radius
            "left_upper_arm": ([4, 5], 0.06),
            "left_forearm": ([5, 6], 0.05),
            "right_upper_arm": ([7, 8], 0.06),
            "right_forearm": ([8, 9], 0.05),
        }

        # --- Subscribers ---
        self.trajectory_sub = self.create_subscription(
            PredictedTrajectory,
            '/predicted_human_trajectory',
            self.trajectory_callback,
            10
        )
        
        self.get_logger().info("Ready to receive human trajectories.")

    def trajectory_callback(self, msg: PredictedTrajectory):
        """
        Updates the MoveIt planning scene with dynamic collision objects based
        on the predicted human trajectory.
        """
        with self.scene_lock:
            if not self.planning_scene_monitor.get_planning_scene():
                self.get_logger().warn("Planning scene not yet available.")
                return

            # This is a simplified approach: update the collision objects to the
            # first predicted pose. A more advanced system would use the full
            # trajectory with MoveIt's time-parameterized collision checking.
            if not msg.trajectory:
                return
            
            first_future_pose = msg.trajectory[0]

            for part_name, (indices, radius) in self.body_parts.items():
                # Calculate the center of the body part
                points = [first_future_pose.joint_positions[i] for i in indices if i < len(first_future_pose.joint_positions)]
                if not points:
                    continue
                
                center_x = sum(p.x for p in points) / len(points)
                center_y = sum(p.y for p in points) / len(points)
                center_z = sum(p.z for p in points) / len(points)

                # Create and publish the collision object
                co = CollisionObject()
                co.header = msg.header
                co.id = f"human_{part_name}"
                co.operation = CollisionObject.MOVE # Add or move the object

                primitive = SolidPrimitive()
                primitive.type = SolidPrimitive.SPHERE
                primitive.dimensions = [radius]

                pose = Pose()
                pose.position.x = center_x
                pose.position.y = center_y
                pose.position.z = center_z
                pose.orientation.w = 1.0

                co.primitives.append(primitive)
                co.primitive_poses.append(pose)
                
                self.planning_scene_publisher.publish(co)

            # self.get_logger().info("Updated planning scene with human collision objects.")

            # --- Continuous Re-planning Logic (Placeholder) ---
            # In a full implementation, you would trigger re-planning here.
            # 1. Check if the current robot trajectory is in collision with the new scene.
            #    - This requires using the PlanningSceneMonitor's is_state_in_collision() method.
            # 2. If a collision is predicted, call MoveIt to compute a new plan.
            # 3. If no plan is found, execute a safety stop.
            # This logic is tightly coupled with the robot's action server (e.g., MoveIt Servo).

def main(args=None):
    rclpy.init(args=args)
    node = MotionPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()