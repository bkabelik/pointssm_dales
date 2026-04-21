import argparse
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import copy
from collections import OrderedDict

# PointSSM / Pointcept imports
from engines.defaults import default_config_parser, default_setup
from models import build_model
from utils.logger import get_root_logger
from datasets.transform import Compose

try:
    import laspy
except ImportError:
    laspy = None

from scipy.spatial import cKDTree
from noisefilter import run_interactive_filter, apply_headless_filter

def build_ground_model(points, return_num, total_returns, grid_size=5.0):
    """
    Build a coarse ground model using last returns to provide a stable vertical datum.
    """
    print(f"Building Ground Reference Model (DTM) at {grid_size}m resolution...")
    # Filter for last returns (most likely to be ground)
    mask = (return_num == total_returns)
    ground_pts = points[mask]
    
    if len(ground_pts) < 100:
        print("  ! Warning: Insufficient 'last return' data. Using all points for ground estimate.")
        ground_pts = points
        
    # Create grid
    min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
    max_x, max_y = np.max(points[:, 0]), np.max(points[:, 1])
    
    nx = int(np.ceil((max_x - min_x) / grid_size)) + 1
    ny = int(np.ceil((max_y - min_y) / grid_size)) + 1
    
    # Initialize with global 1st percentile to avoid underground noise
    global_floor = np.percentile(ground_pts[:, 2], 1)
    grid = np.full((nx, ny), global_floor, dtype=np.float32)
    
    # Grid indexing
    gx = ((ground_pts[:, 0] - min_x) / grid_size).astype(int)
    gy = ((ground_pts[:, 1] - min_y) / grid_size).astype(int)
    
    # Min-pooling for floor detection
    for i in range(len(ground_pts)):
        if ground_pts[i, 2] < grid[gx[i], gy[i]]:
            grid[gx[i], gy[i]] = ground_pts[i, 2]
            
    return grid, (min_x, min_y), grid_size

def get_floor_height(points, ground_model):
    """
    Lookup floor height from the pre-calculated DTM.
    """
    grid, (min_x, min_y), grid_size = ground_model
    gx = ((points[:, 0] - min_x) / grid_size).astype(int)
    gy = ((points[:, 1] - min_y) / grid_size).astype(int)
    
    # Clip to grid boundaries
    gx = np.clip(gx, 0, grid.shape[0] - 1)
    gy = np.clip(gy, 0, grid.shape[1] - 1)
    
    return grid[gx, gy]

def parse_args():
    parser = argparse.ArgumentParser("PointSSM Predictor for LAS files")
    parser.add_argument("--folder", type=str, required=True, help="Folder containing .las files")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint (e.g., model_best.pth)")
    parser.add_argument("--config_file", type=str, default="configs/dales/semseg-dales-12.py", help="Model config file")
    parser.add_argument("--noise_filter", choices=["yes", "no", "interactive"], default="interactive", help="Enable noise filtering")
    parser.add_argument("--smoothing", choices=["yes", "no"], default="yes", help="Enable majority voting smoothing")
    parser.add_argument("--options", nargs="+", action="append", help="override some settings in the used config")
    return parser.parse_args()


def smooth_predictions(points, preds, k=30, z_threshold=2.0):
    """
    Apply k-NN majority voting smoothing. 
    Improved: Edge-preserving by ignoring neighbors with large height differences (e.g. Roof vs Ground).
    """
    print(f"Applying edge-preserving k-NN (k={k}, dZ={z_threshold}m) smoothing...")
    tree = cKDTree(points)
    dists, indices = tree.query(points, k=k)
    
    smoothed_preds = np.zeros_like(preds)
    for i in range(len(preds)):
        neighbor_indices = indices[i]
        
        # Height-aware filtering: only consider neighbors within vertical range
        z_diff = np.abs(points[neighbor_indices, 2] - points[i, 2])
        valid_neighbors = neighbor_indices[z_diff < z_threshold]
        
        if len(valid_neighbors) > 0:
            neighbors_preds = preds[valid_neighbors]
            counts = np.bincount(neighbors_preds)
            smoothed_preds[i] = np.argmax(counts)
        else:
            # Fallback to original if no geometric neighbors found
            smoothed_preds[i] = preds[i]
    
    return smoothed_preds

# Global state to store persistent filter parameters across tiles
persistent_filter_config = {
    "params": None,
    "active": None,
    "apply_to_all": False
}

def predict_las(las_file_path, model, transform, cfg, args):
    if laspy is None:
        raise ImportError("laspy is not installed. Please install it (pip install laspy[lazrs,pylas]).")
        
    print(f"Loading {las_file_path}...")
    las = laspy.read(las_file_path)
    
    # Extract Coordinates and Intensity
    # Las files usually have coordinates scaled by header scale and added to offset. las.x gives actual float values.
    points = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    intensities = np.array(las.intensity).astype(np.float32)
    
    # Extract return info for Ground Model
    try:
        return_num = np.array(las.return_number)
        total_returns = np.array(las.number_of_returns)
    except AttributeError:
        print("Warning: Return number information missing. Ground model will be less accurate.")
        return_num = np.ones(len(points))
        total_returns = np.ones(len(points))

    # Pre-calculate Ground Model (DTM)
    ground_model = build_ground_model(points, return_num, total_returns)
    
    # Apply Noise Filter
    global persistent_filter_config
    if args.noise_filter in ["yes", "interactive"]:
        if persistent_filter_config["apply_to_all"]:
            print(f"Applying persistent filter settings to {os.path.basename(las_file_path)}...")
            keep_mask = apply_headless_filter(points, intensities, 
                                            persistent_filter_config["params"], 
                                            persistent_filter_config["active"])
        else:
            keep_mask, params, active, apply_to_all = run_interactive_filter(points, intensities)
            if apply_to_all:
                persistent_filter_config["params"] = params
                persistent_filter_config["active"] = active
                persistent_filter_config["apply_to_all"] = True
    else:
        keep_mask = np.ones(len(points), dtype=bool)
        
    print("Preparing data for PointSSM and splitting into overlapping blocks (Stride: 25m, Size: 50m)...")
    block_size = 50.0  # MUST be 50 to maintain depth=7 Hilbert curve fractal boundaries
    stride = 20.0      # 60% Overlap for maximum consensus at block edges
    
    # Calculate Global Intensity Scale (99th percentile to avoid high-intensity noise)
    global_intensity_max = np.percentile(intensities, 99.5) # Slightly lower for more wire contrast
    print(f"Global Intensity Scaling: Reference Max={global_intensity_max:.2f}")

    # Drop outliers from the active prediction set to avoid skewing geometric centers in 50m blocks
    valid_points = points[keep_mask].astype(np.float32)
    valid_intensities = intensities[keep_mask].astype(np.float32)
    valid_indices = np.where(keep_mask)[0]
    
    global_pred_logits = np.zeros((len(valid_points), cfg.data.num_classes), dtype=np.float32)
    
    model.eval()
    
    min_x, max_x = np.min(valid_points[:, 0]), np.max(valid_points[:, 0])
    min_y, max_y = np.min(valid_points[:, 1]), np.max(valid_points[:, 1])
    blocks_x = int(np.ceil((max_x - min_x) / stride))
    blocks_y = int(np.ceil((max_y - min_y) / stride))
    total_blocks = blocks_x * blocks_y
    processed_blocks = 0

    print(f"Splitting into {total_blocks} potential overlapping blocks of {block_size}x{block_size}m...")
    
    for bx in range(blocks_x):
        for by in range(blocks_y):
            # Define block boundaries using stride
            x_start = min_x + bx * stride
            y_start = min_y + by * stride
            x_end = x_start + block_size
            y_end = y_start + block_size
            
            x_mask = (valid_points[:, 0] >= x_start) & (valid_points[:, 0] < x_end)
            y_mask = (valid_points[:, 1] >= y_start) & (valid_points[:, 1] < y_end)
            block_mask = x_mask & y_mask
            
            block_idx = np.where(block_mask)[0]
            if len(block_idx) < 10:
                continue
                
            processed_blocks += 1
            if processed_blocks % 5 == 0:
                print(f"  -> Processing block {processed_blocks}/{total_blocks} (Points: {len(block_idx)})...")
            
            block_points = valid_points[block_idx].copy()
            block_intensities = valid_intensities[block_idx]
            
            # Pointcept / DALES Strict Normalization Geometry
            local_center_x = (x_start + x_end) / 2.0
            local_center_y = (y_start + y_end) / 2.0
            
            # Using stable DTM lookup for Consistent Z-Normalisation
            # This prevents class flickering in overlapping regions
            z_floors = get_floor_height(block_points, ground_model)

            # Normalization logic: 
            # Dividing by 25.0 maps the 50m block size (or radius 25m) to the [-1, 1] range.
            # This is critical for activating the pre-trained weights correctly.
            block_points[:, 0] = (block_points[:, 0] - local_center_x) / 25.0
            block_points[:, 1] = (block_points[:, 1] - local_center_y) / 25.0
            block_points[:, 2] = (block_points[:, 2] - z_floors) / 25.0
            
            # Spatial weighting for overlap resolving: Squared distance for sharper transition
            dist_x = np.abs(block_points[:, 0])
            dist_y = np.abs(block_points[:, 1])
            weights = (1.0 - np.maximum(dist_x, dist_y)) ** 2
            weights = np.clip(weights, 0.01, 1.0)
            
            # Intensity normalization: Global scaling to maintain material contrast
            norm_intensities = np.clip(block_intensities / (global_intensity_max + 1e-6), 0, 1)
            
            data_dict = {
                "coord": block_points,
                "strength": norm_intensities,
                "segment": np.zeros(len(block_points), dtype=np.int32),
                "name": f"b{bx}_{by}"
            }
            
            data_dict = transform(data_dict)
            fragment_list = data_dict["fragment_list"]
            segment = data_dict["segment"]
            
            # Aggregate probabilities across fragments back into full block length
            pred = torch.zeros((segment.size, cfg.data.num_classes)).cuda()
            
            for i in range(len(fragment_list)):
                input_dict = fragment_list[i]
                from datasets import collate_fn
                input_dict = collate_fn([input_dict])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                        
                idx_part = input_dict["index"]
                with torch.no_grad():
                    pred_part = model(input_dict)["seg_logits"]
                    pred_part = F.softmax(pred_part, -1)
                    bs = 0
                    for be in input_dict["offset"]:
                        pred[idx_part[bs:be], :] += pred_part[bs:be]
                        bs = be

            # Move from GPU back to block indices
            pred_part = pred.cpu().numpy()
            if "inverse" in data_dict.keys():
                pred_part = pred_part[data_dict["inverse"]]
                
            global_pred_logits[block_idx] += (pred_part * weights[:, np.newaxis])

    # Final Softmax Aggregation
    pred_classes = np.argmax(global_pred_logits, axis=1)
    
    # Smoothing
    if args.smoothing == "yes":
        pred_classes = smooth_predictions(valid_points, pred_classes)
        
    print(f"Prediction complete for {processed_blocks} blocks.")
    print("Reassembling LAS and writing output with ASPRS mapping...")
    # Remap DALES (0-7) to ASPRS codes:
    # 0->2, 1->5, 2->20, 3->20, 4->14, 5->19, 6->15, 7->6
    id2asprs = np.array([2, 5, 20, 20, 14, 19, 15, 6], dtype=np.uint8)
    asprs_preds = id2asprs[pred_classes]

    # Initialize with ASPRS 7 (Low Noise) as default for filtered points
    final_classes = np.full(len(points), 7, dtype=np.uint8)
    
    # Map predictions back to original indices
    final_classes[valid_indices] = asprs_preds
    
    las.classification = final_classes
    out_dir = os.path.join(args.folder, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(las_file_path))
    las.write(out_path)
    print(f"Saved: {out_path}")


def main():
    args = parse_args()
    
    logger = get_root_logger()
    cfg = default_config_parser(args.config_file, args.options)
    cfg = default_setup(cfg)
    
    logger.info("=> Building model ...")
    model = build_model(cfg.model).cuda()
    
    logger.info(f"Loading weight at: {args.model_path}")
    checkpoint = torch.load(args.model_path)
    weight = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        if key.startswith("module."):
            key = key[7:]
        weight[key] = value
    model.load_state_dict(weight, strict=True)
    
    transform = Compose(cfg.data.test.transform)
    # inject test_cfg to transform, to make fragment_list creation work
    class DummyDataset:
        pass
    
    class TestWrapper:
        def __init__(self, transform, test_cfg):
            self.transform = transform
            self.test_cfg = test_cfg
        def __call__(self, data_dict):
            data_dict = self.transform(data_dict)
            
            # Now we apply voxelize and post_transform
            if self.test_cfg is not None:
                voxelize_op = Compose([self.test_cfg.voxelize])
                post_transform = Compose(self.test_cfg.post_transform)
                
                # To create fragments for grid_sample based inference
                fragment_list = []
                data_dict_copy = copy.deepcopy(data_dict)
                data_part_list = voxelize_op(data_dict_copy)
                
                if isinstance(data_part_list, list):
                    for data_part in data_part_list:
                        fragment_list.append(post_transform(data_part))
                else:
                    fragment_list.append(post_transform(data_part_list))
                data_dict["fragment_list"] = fragment_list
            return data_dict
            
    test_transform = TestWrapper(transform, cfg.data.test.test_cfg)
    
    las_files = glob.glob(os.path.join(args.folder, "*.las")) + glob.glob(os.path.join(args.folder, "*.laz"))
    
    for las_file in las_files:
        predict_las(las_file, model, test_transform, cfg, args)
        
if __name__ == "__main__":
    main()
