import os

def create_project_structure():
    """
    Creates the specified directory and file structure for the project.
    """
    # List of all files to be created, including their full paths
    files_to_create = [
        ".github/workflows/ci.yml",
        "data/process_h36m.py",
        "docker/Dockerfile",
        "docker/docker-compose.yml",
        "docs/PRD.md",
        "notebooks/01_model_prototyping.ipynb",
        "src/hrc_msgs/msg/PoseGraph.msg",
        "src/hrc_msgs/msg/PoseGraphHistory.msg",
        "src/hrc_msgs/msg/PredictedTrajectory.msg",
        "src/hrc_msgs/CMakeLists.txt",
        "src/hrc_msgs/package.xml",
        "src/perception/perception/__init__.py",
        "src/perception/perception/data_processing_node.py",
        "src/perception/perception/kalman_filter.py",
        "src/perception/package.xml",
        "src/perception/resource/perception",
        "src/perception/setup.py",
        "src/prediction_model/prediction_model/__init__.py",
        "src/prediction_model/prediction_model/model.py",
        "src/prediction_model/prediction_model/predict_node.py",
        "src/prediction_model/prediction_model/utils.py",
        "src/prediction_model/package.xml",
        "src/prediction_model/resource/prediction_model",
        "src/prediction_model/setup.py",
        "src/prediction_model/train.py",
        "src/robot_control/robot_control/__init__.py",
        "src/robot_control/robot_control/motion_planner_node.py",
        "src/robot_control/launch/system.launch.py",
        "src/robot_control/package.xml",
        "src/robot_control/resource/robot_control",
        "src/robot_control/setup.py",
        "tests/test_prediction_model.py",
        ".gitignore",
        "LICENSE",
        "README.md",
        "requirements.txt"
    ]

    for filepath in files_to_create:
        try:
            # Get the directory part of the filepath
            directory = os.path.dirname(filepath)
            
            # If a directory is specified (i.e., not a root file)
            # create it, including any parent directories.
            # 'exist_ok=True' prevents an error if the directory already exists.
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            # Create the empty file. 'a' mode creates the file if it doesn't exist.
            with open(filepath, 'a'):
                pass
            
            print(f"✅ Created: {filepath}")
            
        except OSError as e:
            print(f"❌ Error creating {filepath}: {e}")

if __name__ == "__main__":
    print("🚀 Starting project setup...")
    create_project_structure()
    print("\n🎉 Project structure created successfully!")