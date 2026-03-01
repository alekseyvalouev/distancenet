import os
import numpy as np
import heapq
import h5py

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def create_gif(image_sequence, output_filename="animation.gif", fps=10):
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

def nodes_path_to_commands(node_path, coords, start_dir=None):
    commands = []
    if len(node_path) <= 1:
        return commands
        
    dir_to_angle = {
        (0, 1): 0,
        (1, 0): 90,
        (0, -1): 180,
        (-1, 0): 270
    }
        
    turn_right = {
        (0, 1): (1, 0),
        (1, 0): (0, -1),
        (0, -1): (-1, 0),
        (-1, 0): (0, 1)
    }
    turn_left = {
        (0, 1): (-1, 0),
        (-1, 0): (0, -1),
        (0, -1): (1, 0),
        (1, 0): (0, 1)
    }
    
    first_node = node_path[0]
    second_node = node_path[1]
    
    first_pos = coords[first_node]
    second_pos = coords[second_node]
    
    if start_dir is not None:
        current_dir = start_dir
    else:
        # Assume starting orientation faces the first move
        current_dir = (second_pos[0] - first_pos[0], second_pos[1] - first_pos[1])
    
    current_pos = first_pos
    current_node = first_node
    
    for next_node in node_path[1:]:
        next_pos = coords[next_node]
        move_dir = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
        
        # Turn until move_dir == current_dir
        while current_dir != move_dir:
            if turn_right[current_dir] == move_dir:
                current_dir = move_dir
                commands.append((current_node, dir_to_angle[current_dir]))
            elif turn_left[current_dir] == move_dir:
                current_dir = move_dir
                commands.append((current_node, dir_to_angle[current_dir]))
            else: # 180 degree turn
                current_dir = turn_right[current_dir]
                commands.append((current_node, dir_to_angle[current_dir]))
                current_dir = turn_right[current_dir]
                commands.append((current_node, dir_to_angle[current_dir]))
                
        current_pos = next_pos
        current_node = next_node
        commands.append((current_node, dir_to_angle[current_dir]))

    return commands

def dijkstra_on_grid(grid, start_id, target_id, start_dir=None):
    """
    Finds the shortest path between start_id and target_id using Dijkstra's algorithm.
    Returns a tuple (path, commands) where path is a list of node IDs and 
    commands is a list of actions ('forward', 'turn left', 'turn right').
    """
    rows = len(grid)
    cols = len(grid[0])
    
    # Map all valid node IDs to their (row, col) coordinates
    coords = {}
    for r in range(rows):
        for c in range(cols):
            node_id = grid[r][c]
            if node_id != -1:
                coords[node_id] = (r, c)
                
    if start_id not in coords:
        raise ValueError(f"Start ID {start_id} not found in grid.")
    if target_id not in coords:
        raise ValueError(f"Target ID {target_id} not found in grid.")

    start_pos = coords[start_id]
    target_pos = coords[target_id]
    
    # Priority queue stores tuples of (cost, (r, c))
    pq = [(0, start_pos)]
    
    distances = {start_pos: 0}
    previous = {start_pos: None}
    
    # 4-way connectivity: Right, Down, Left, Up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] 
    
    path = []
    while pq:
        current_cost, (r, c) = heapq.heappop(pq)
        
        # If we reached the target, reconstruct the path
        if (r, c) == target_pos:
            curr = (r, c)
            while curr is not None:
                path.append(grid[curr[0]][curr[1]])
                curr = previous[curr]
            path = path[::-1] # Reverse to get start -> target
            break
            
        # Optimization: ignore if we've already found a better path to this node
        if current_cost > distances.get((r, c), float('inf')):
            continue
            
        # Check all valid neighbors
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            # Check bounds and make sure it's not an obstacle (-1)
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != -1:
                new_cost = current_cost + 1 # Cost is 1 per step
                neighbor_pos = (nr, nc)
                
                # If we found a shorter path to the neighbor, process it
                if new_cost < distances.get(neighbor_pos, float('inf')):
                    distances[neighbor_pos] = new_cost
                    previous[neighbor_pos] = (r, c)
                    heapq.heappush(pq, (new_cost, neighbor_pos))
                    
    if not path:
        return None, []

    # Reconstruct directional commands
    commands = nodes_path_to_commands(path, coords, start_dir)

    return path, commands


def generate_trajectories(grid, horizon=8):
    """
    Finds all valid shortest-path trajectories of exactly `horizon` length 
    without guess and check. Returns a list of tuples (path, commands), where 
    path is a list of node IDs and commands is a list of action strings.
    """
    rows = len(grid)
    cols = len(grid[0])
    
    # 4-way connectivity: Right, Down, Left, Up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] 
    
    # Precompute mapping of node IDs to (row, col) coordinates
    coords_map = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != -1:
                coords_map[grid[r][c]] = (r, c)
    
    trajectories = []
    
    # Run BFS from every valid starting position
    for sr in range(rows):
        for sc in range(cols):
            if grid[sr][sc] == -1:
                continue
                
            start_pos = (sr, sc)
            queue = [start_pos]
            distances = {start_pos: 0}
            
            idx = 0
            while idx < len(queue):
                curr = queue[idx]
                idx += 1
                
                dist = distances[curr]
                if dist == horizon:
                    target_pos = curr
                    
                    # 1. Traceback to find all nodes on any shortest path from start_pos to target_pos
                    valid_nodes_for_target = {target_pos}
                    queue_trace = [target_pos]
                    while queue_trace:
                        trace_curr = queue_trace.pop(0)
                        d_curr = distances[trace_curr]
                        if d_curr == 0:
                            continue
                        for dr_move, dc_move in directions:
                            nr, nc = trace_curr[0] + dr_move, trace_curr[1] + dc_move
                            if (nr, nc) in distances and distances[(nr, nc)] == d_curr - 1:
                                if (nr, nc) not in valid_nodes_for_target:
                                    valid_nodes_for_target.add((nr, nc))
                                    queue_trace.append((nr, nc))

                    # 2. Reconstruct forward path prioritizing furthest axis (minimize Chebyshev distance to target)
                    path_coords = [start_pos]
                    curr_fwd = start_pos
                    tr, tc = target_pos
                    
                    while curr_fwd != target_pos:
                        d_fwd = distances[curr_fwd]
                        best_next = None
                        best_chebyshev = float('inf')
                        
                        for dr_move, dc_move in directions:
                            nr, nc = curr_fwd[0] + dr_move, curr_fwd[1] + dc_move
                            # Check if valid next node on shortest path
                            if (nr, nc) in valid_nodes_for_target and distances[(nr, nc)] == d_fwd + 1:
                                cheb = max(abs(tr - nr), abs(tc - nc))
                                if cheb < best_chebyshev:
                                    best_chebyshev = cheb
                                    best_next = (nr, nc)
                                    
                        curr_fwd = best_next
                        path_coords.append(curr_fwd)
                        
                    node_path = [grid[r][c] for r, c in path_coords]
                    
                    # 3. Determine start_dir pointing along most distant axis
                    dr, dc = tr - sr, tc - sc
                    if abs(dr) >= abs(dc):
                        start_dir = (1 if dr > 0 else -1, 0)
                    else:
                        start_dir = (0, 1 if dc > 0 else -1)
                        
                    commands = nodes_path_to_commands(node_path, coords_map, start_dir=start_dir)
                    trajectories.append((node_path, commands))
                    continue # Do not explore past the horizon
                    
                # Explore neighbors
                for dr_move, dc_move in directions:
                    nr, nc = curr[0] + dr_move, curr[1] + dc_move
                    
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != -1:
                        neighbor = (nr, nc)
                        
                        if neighbor not in distances:
                            distances[neighbor] = dist + 1
                            queue.append(neighbor)
                            
    return trajectories

def subdivide_trajectory(traj, horizon=8):
    # if horizon is 8, 1-3, 3-5, 5-7, 7-9
    # length is horizon / 4 + 1
    
    sub_trajs = []

    H = horizon // 4
    points = traj[0]
    first_points = {} # First time a point is mentioned in our actual path (i.e. with turns)
    last_points = {} # Last time a point is mentioned in our actual path (i.e. with turns)
    for i, point in enumerate(traj[1]):
        if point[0] not in first_points:
            first_points[point[0]] = i
        last_points[point[0]] = i
    
    # Now generate all possible permutations. 
    valid_idx = list(range(1, 10, 2))
    for i, value_start in enumerate(valid_idx):
        for j, value_end in enumerate(valid_idx[i+1:]):
            start_id = last_points[traj[1][value_start][0]]
            end_id = first_points[traj[1][value_end][0]]
            
            sub_traj = traj[0][start_id:end_id+1]
            sub_traj_commands = traj[1][start_id:end_id+1]
            sub_trajs.append((sub_traj, sub_traj_commands))
    
    return sub_trajs

def get_sub_traj(trajectory, end_idx):
    output = []
    for i in range(0, len(trajectory[1])):
        output.append(trajectory[1][i])
        if (trajectory[1][i][0] == end_idx):
            break
    traj_len = trajectory[0].index(end_idx)
    return (traj_len, output)
        
def get_traj_images(traj, images):
    result = []
    for step in traj:
        result.append(images[step[0]*4 + (step[1]//90)].reshape((512, 512, 3)))
    return result

def dijkstra_sub_traj_hybrid(grid_id, n_entries, horizon=8):
    # run dijkstra
    start_node = np.random.randint(0, n_entries)
    end_node = np.random.randint(0, n_entries)
    trajectory = dijkstra_on_grid(grid_id, start_node, end_node)
    while len(trajectory[0]) >= horizon or len(trajectory[0]) <= 2:
        start_node = np.random.randint(0, n_entries)
        end_node = np.random.randint(0, n_entries)
        trajectory = dijkstra_on_grid(grid_id, start_node, end_node)

    # trajectory length should be at least 3. 
    # now that we got this trajectory, let's subdivide it into subgoals. 
    # subgoals are every 3. i.e. in a trajectory 1 2 3 4 5,
    # we hav subgoals 1-3, 1-5. 
    # if traj length is even, drop the final element :)
    sub_trajs = []
    for i in range(2, len(trajectory[0]), 2):
        sub_trajs.append(get_sub_traj(trajectory, trajectory[0][i]))
    return sub_trajs

if __name__ == "__main__":
    processed_data_folder = "/home/alekseyvalouev/goalnav/AdobeIndoorNav/data"
    raw_data_folder = "/home/alekseyvalouev/goalnav/AdobeIndoorNav/datasets/adobeindoornav_dataset"
    scene_name = "et07-cr-galgary"
    
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
        
        # Taking valid test points from real dataset
        valid_nodes = grid_id[grid_id != -1]
        np.random.seed(42)  # For deterministic outputs
        
        #trajectories = generate_trajectories(grid_id, horizon=8)
        #print(f"Found {len(trajectories)} trajectories of length 8.")
        trajectories = dijkstra_sub_traj_hybrid(grid_id, len(valid_nodes))
        print(trajectories)
        
        #print(subdivide_trajectory(trajectories[1]))
        
        images = get_traj_images(trajectories[1][1], images)
        create_gif(images, "animation.gif", fps=1)
        
                
    except FileNotFoundError:
        print("Real dataset files not found!")