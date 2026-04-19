from setuptools import setup

package_name = 'lapgym_ros2_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='SUBHASH',
    maintainer_email='subhashtronics@gmail.com',
    description='ROS 2 bridge for SOFA LapGym surgical-rl simulation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bridge_node = lapgym_ros2_bridge.bridge_node:main',
            'teleop_keyboard = lapgym_ros2_bridge.teleop_keyboard:main',
        ],
    },
)