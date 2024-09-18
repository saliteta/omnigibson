import numpy as np

cfg = {}

# Define scene
cfg["scene"] = {
    "type": "InteractiveTraversableScene",
    "floor_plane_visible": True,
    "scene_model":"Beechwood_0_garden"
}

# Define robots
cfg["robots"] = [
    {
        "type": "Turtlebot",
        "name": "whatever",
        "obs_modalities": ["rgb", "depth"],
        "action_type":'continuous',
        'sensor_config':{
            'VisionSensor':{
                'sensor_kwargs':{
                    'image_height':480,
                    'image_width':640,
                }
            }
        }
    },
]

move = ['ArrowRight', 'ArrowLeft', 'ArrowUp', 'ArrowDown', ' ', 'z']
rotation = ['w', 'a', 's', 'd', 'q', 'e']

start_pos=np.array([1.89897189, 23.27433968,  1.06539989])

fx=320
fy=320
cx=320
cy=240