# Omnigibson Semantic Reconstruction UI 
The following code is created for the following motivation
- Multiple user need to visualize the process
- The so called process is the agent (currently human) to explore the scene and collect the data for reconstruction


## How to Use
- First you need to install Omnigibson according to what [Behavior 1K](https://behavior.stanford.edu/behavior-1k) ask you do do
- Then directly call the following, it should work
```
    python app.py 
```
- If it does not, please raise an issue
- To modify the scene one wants to import, change this part is config.py
```
cfg["scene"] = {
    "type": "InteractiveTraversableScene",
    "floor_plane_visible": True,
    "scene_model":"Beechwood_0_garden"
}
```

## Simple result

To check our result, one can visit the following youtube website: 
[Result](https://youtu.be/_gSVPduxE20)