# consider a grid. select one point. pass a wanted distance (e.g. 4). do bfs away from this point, find all other points 4 away. orient the start angle so that it face towards the goal along the longest axis. end angle = start angle. 
from collections import deque
import numpy as np
import os
import h5py

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def create_gif(image_sequence, output_filename="animation.gif", fps=1):
    """
    Creates a GIF from a sequence of images.
    
    Args:
        image_sequence: A list of numpy arrays, where each array is an image. 
                        Shape should be (H, W, 3) or (H, W).
        output_filename: Output path for the GIF.
        fps: Frames per second for the animation.
    """
    if not image_sequence:
        raise ValueError("Image sequence is empty!")

    fig, ax = plt.subplots()
    # Turn off axis for clean visualization
    ax.axis('off')
    
    # Initialize the plot with the first image
    im = ax.imshow(image_sequence[0], animated=True)

    def update(frame_idx):
        im.set_array(image_sequence[frame_idx])
        return [im]

    # Create the animation object
    anim = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(image_sequence), 
        interval=1000 / fps,  # Interval is in milliseconds
        blit=True
    )

    # Save as GIF using pillow writer
    # Note: matplotlib requires the 'pillow' package installed to save gifs!
    anim.save(output_filename, writer='pillow', fps=fps)
    
    # Alternatively close the plot so it doesn't try to show if running in a notebook/IDE
    plt.close(fig)
    print(f"Saved animation to {output_filename}")

def generate_start_end_pairs(grid, start_id, distance):
    for i, row in enumerate(grid):
        for j, entry in enumerate(row):
            if entry == start_id:
                start_pos = (i, j)
                break

    others = bfs_distance(grid, start_pos, distance)
    # orient the starting point towards the goal along the longest axis
    res = []
    for other_pos in others:
        axis_alignment = np.argmax(np.abs(np.array(other_pos) - np.array(start_pos)))
        # now determine if positive or negative
        direction = [0, 0]
        if other_pos[axis_alignment] - start_pos[axis_alignment] > 0:
            direction[axis_alignment] = 1
        else:
            direction[axis_alignment] = -1
        dir_to_angle = {
            (0, 1): 0,
            (1, 0): 90,
            (0, -1): 180,
            (-1, 0): 270
        }
        start_angle = dir_to_angle[tuple(direction)]
        end_id = grid[other_pos[0]][other_pos[1]]
        res.append([(start_id, start_angle), (end_id, start_angle)])
    return res

def get_traj_images(traj, images):
    result = []
    for step in traj:
        result.append(images[step[0]*4 + (step[1]//90)].reshape((512, 512, 3)))
    return result
            
def bfs_distance(grid, start_pos, distance):
    """
    Perform BFS from start_pos (i, j) to find all reachable points exactly `distance` away.
    
    Args:
        grid: 2D list or numpy array where -1 indicates an obstacle
        start_pos: Tuple (i, j) representing the starting coordinates
        distance: The exact distance to search for
        
    Returns:
        A list of tuples (r, c) representing all reachable points at the given distance.
    """
    rows = len(grid)
    cols = len(grid[0])
    
    # 4-way connectivity: Right, Down, Left, Up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    queue = deque([start_pos])
    distances = {start_pos: 0}
    
    points_at_distance = []
    
    while queue:
        curr = queue.popleft()
        dist = distances[curr]
        
        if dist == distance:
            points_at_distance.append(curr)
            # Assuming we only want points at exactly this distance, no need to explore further from here.
            # If distance can be smaller and you still want points, you can modify this.
            continue
            
        for dr, dc in directions:
            nr, nc = curr[0] + dr, curr[1] + dc
            
            # Check bounds and if it's not an obstacle (-1 grid value often denotes obstacle)
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != -1:
                neighbor = (nr, nc)
                
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    queue.append(neighbor)
                    
    return points_at_distance

def get_all_valid_paths(grid, size, horizon, scene_name):
    result = []
    for i in range(size):
        pairs = generate_start_end_pairs(grid, i, horizon)
        #traj_images = [get_traj_images(pair, images) for pair in pairs]
        result.extend([((scene_name, scene_name), horizon, pair) for pair in pairs])
    return result

def get_paths_for_scene(scene_name, horizons=[2, 4, 16]):
    processed_data_folder = "/home/alekseyvalouev/goalnav/AdobeIndoorNav/data"
    raw_data_folder = "/home/alekseyvalouev/goalnav/AdobeIndoorNav/datasets/adobeindoornav_dataset"
    
    h5_path = os.path.join(processed_data_folder, 'panorama_images_cropped_rgb_images', f'{scene_name}.h5')
    h5_file = h5py.File(h5_path, 'r')
    images = h5_file['observation']

    occ_input_file = os.path.join(raw_data_folder, scene_name, 'grid_occ.npy')
    connect_input_file = os.path.join(raw_data_folder, scene_name, 'grid_connect.npy')

    # Apply to existing files if available
    try:
        with open(occ_input_file, 'rb') as fin:
            grid_id = np.load(fin)
            
        with open(connect_input_file, 'rb') as fin:
            grid_connect = np.load(fin)

        print("Real grid_id shape:", grid_id.shape)
        print(grid_id)
        
        valid_nodes = grid_id[grid_id != -1]

        results = []
        for horizon in horizons:
            trajectories = get_all_valid_paths(grid_id, len(valid_nodes), horizon, scene_name)
            results.extend(trajectories)
        
        print("Total trajectories: ", len(results))
        print(results)

        #horizons, images = unzip(results)

        

        return results

    except FileNotFoundError:
        print("Real dataset files not found!")

if __name__ == "__main__":
    scene_name = "et07-imagination-lab"

    processed_data_folder = "/home/alekseyvalouev/goalnav/AdobeIndoorNav/data"
    raw_data_folder = "/home/alekseyvalouev/goalnav/AdobeIndoorNav/datasets/adobeindoornav_dataset"
    
    h5_path = os.path.join(processed_data_folder, 'panorama_images_cropped_rgb_images', f'{scene_name}.h5')
    h5_file = h5py.File(h5_path, 'r')
    images = h5_file['observation']
    
    paths = get_paths_for_scene(scene_name, horizons=[2, 4, 8, 16])
    for path in paths:
        if path[1] == 16:
            traj_images = get_traj_images(path[2], images)
            create_gif(traj_images)
        
                
