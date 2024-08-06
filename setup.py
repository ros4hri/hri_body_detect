from setuptools import find_packages, setup

package_name = 'hri_body_detect'

setup(
    name=package_name,
    version='3.1.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/ament_index/resource_index/pal_system_module',
            ['module/' + package_name]),
        ('share/ament_index/resource_index/pal_configuration.hri_body_detect',
            ['config/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/hri_body_detect.launch.py']),
        ('share/' + package_name + '/launch', ['launch/hri_body_detect_with_args.launch.py']),
        ('share/' + package_name + '/module', ['module/hri_body_detect_module.yaml']),
        ('share/' + package_name + '/weights',
            ['hri_body_detect/weights/pose_landmarker_full.task']),
        ('share/' + package_name + '/config', ['config/00-defaults.yml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ferran Gebelli',
    maintainer_email='ferran.gebelli@pal-robotics.com',
    description='ROS node implementing multibody 2D/3D body pose estimation, using Google Mediapipe.\
                 Part of ROS4HRI.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'node_pose_detect = hri_body_detect.node_pose_detect:main'
        ],
    },
)
