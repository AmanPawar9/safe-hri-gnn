from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. Perception Module: Data Processing Node
        Node(
            package='perception',
            executable='data_processing_node',
            name='data_processing_node',
            output='screen',
            parameters=[
                # Use a remapping to connect to the actual sensor/body-tracker output
                # {'input_topic': '/isaac_ros_body_track'}
            ]
        ),

        # 2. Prediction Module: ST-GNN Prediction Node
        Node(
            package='prediction_model',
            executable='predict_node',
            name='prediction_node',
            output='screen',
            parameters=[
                # Specify the name of the trained model file
                {'model_name': 'stgnn_model.pth'}
            ]
        ),

        # 3. Robot Control Module: Motion Planner Node
        Node(
            package='robot_control',
            executable='motion_planner_node',
            name='motion_planner_node',
            output='screen',
            parameters=[
                # This should match the robot description parameter used by MoveIt
                {'robot_description': 'robot_description'}
            ]
        ),
    ])