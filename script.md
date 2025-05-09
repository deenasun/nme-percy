# Required libraries
pip install torch opencv-python numpy matplotlib gymnasium timm scikit-image

## this converts the city.jpg to a depth map (as a png for visualization)
python scripts/estimate_depth.py --image data/city.jpg --model DPT_Large --output results/visualizations/city_depth_vis.png  

## this converts the city.jpg to a depth map (as a npy for grid mapping)
python scripts/batch_depth_estimation.py --input-dir data/ --extensions .jpg --model DPT_Large --save-raw --save-colored --save-vis  

## this converts the depth map to a grid map (the streets are 1 meaning it can be traveled, the city blocks are 0)
python scripts/convert_depth_to_grid.py --input results/raw_depth/city_depth.npy --grid-size 1024 --threshold 0.1 --save-visualization

## this tests the path planning (A*)
### I think instead of saying A* is our control, we can we use A* to make sure the map can be traversed. 
### so they dont realize how terrible the RL agent is. 
python -m scripts.test_path_planning --grid results/grid_maps/city_depth_grid.npy --max-grid-size 128

## this trains the RL agent
### it will take about 5 minutes for 64x64 grid
### as long as there is a bar that says Evaluation 100/100: 100%|██████████████████████████████| 100/100 [00:00<00:00, 219.56it/s, Reward=63.10, Steps=200, Success=0] it is working
python -m scripts.train_rl_agent --grid results/grid_maps/city_depth_grid.npy --invert-grid --max-grid-size 64 --episodes 50000 --learning-rate 0.2 --discount-factor 0.99 --exploration-rate 1 --exploration-decay 0.998
