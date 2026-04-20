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

try:
    import open3d as o3d
except ImportError:
    o3d = None

from scipy.spatial import cKDTree

def parse_args():
    parser = argparse.ArgumentParser("PointSSM Predictor for LAS files")
    parser.add_argument("--folder", type=str, required=True, help="Folder containing .las files")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint (e.g., model_best.pth)")
    parser.add_argument("--config_file", type=str, default="configs/dales/semseg-dales-12.py", help="Model config file")
    parser.add_argument("--noise_filter", choices=["yes", "no", "interactive"], default="interactive", help="Enable noise filtering")
    parser.add_argument("--smoothing", choices=["yes", "no"], default="yes", help="Enable majority voting smoothing")
    parser.add_argument("--options", nargs="+", action="append", help="override some settings in the used config")
    return parser.parse_args()

def interactive_noise_filter(points):
    """
    Apply interactive Statistical Outlier Removal (SOR) or Radius Outlier Removal (ROR).
    Returns boolean mask of points to keep.
    """
    if o3d is None:
        print("Warning: open3d is not installed. Run `pip install open3d` to enable noise filtering. Skipping...")
        return np.ones(len(points), dtype=bool)
    
    print("Converting to Open3D format for filtering...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Default parameters: Relaxed for LAS vegetation
    method = "SOR"
    nb_neighbors = 20
    std_ratio = 4.0   # Relaxed (standard is 2.0, which often kills trees)
    radius = 1.0     # Radius for ROR
    min_points = 5   # Min points for ROR
    
    print("\n--- Interactive Noise Filtering ---")
    print("TIP: If you have a lot of vegetation, SOR with StdRatio < 3.0 is too aggressive.")
    print("TIP: Try Radius Outlier Removal (ROR) for complex tree canopies.")
    
    while True:
        print(f"\nCurrent Method: {method}")
        if method == "SOR":
            print(f"  -> Statistical (SOR): Removing points with z-score > {std_ratio}")
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        else:
            print(f"  -> Radius (ROR): Keeping points if at least {min_points} neighbors within {radius} meters")
            cl, ind = pcd.remove_radius_outlier(nb_points=min_points, radius=radius)
            
        mask = np.zeros(len(points), dtype=bool)
        mask[ind] = True
        
        noise_pts = len(points) - np.sum(mask)
        print(f"Result: Filtered {noise_pts} noise points out of {len(points)} ({noise_pts / len(points) * 100:.2f}%).")
        
        print("\nOptions:")
        print(" [a] Accept these parameters and proceed")
        print(" [s] Switch method (SOR <-> ROR)")
        print(" [c] Change parameters for current method")
        choice = input("Your choice [a/s/c]: ").strip().lower()
        
        if choice == 'a':
            break
        elif choice == 's':
            method = "ROR" if method == "SOR" else "SOR"
        elif choice == 'c':
            if method == "SOR":
                nb_neighbors = int(input(f"Enter Neighbors (current {nb_neighbors}): ") or nb_neighbors)
                std_ratio = float(input(f"Enter StdRatio (current {std_ratio}): ") or std_ratio)
            else:
                radius = float(input(f"Enter Radius (current {radius}): ") or radius)
                min_points = int(input(f"Enter MinPoints (current {min_points}): ") or min_points)
        else:
            print("Invalid choice. Try again.")
            
    return mask

def smooth_predictions(points, preds, k=15):
    """
    Apply k-NN majority voting smoothing to predictions (good for overlapping flight strips).
    """
    print(f"Applying k-NN (k={k}) smoothing to predictions...")
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)
    
    smoothed_preds = np.zeros_like(preds)
    for i in range(len(preds)):
        neighbors_preds = preds[indices[i]]
        counts = np.bincount(neighbors_preds)
        smoothed_preds[i] = np.argmax(counts)
    
    return smoothed_preds

def predict_las(las_file_path, model, transform, cfg, args):
    if laspy is None:
        raise ImportError("laspy is not installed. Please install it (pip install laspy[lazrs,pylas]).")
        
    print(f"Loading {las_file_path}...")
    las = laspy.read(las_file_path)
    
    # Extract Coordinates and Intensity
    # Las files usually have coordinates scaled by header scale and added to offset. las.x gives actual float values.
    points = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    intensities = np.array(las.intensity).astype(np.float32)
    
    # Apply Noise Filter
    if args.noise_filter in ["yes", "interactive"]:
        # We can simulate an interactive loop, but for terminal simplicity we just run it once.
        # To make it fully interactive, we'd add input() logic.
        keep_mask = interactive_noise_filter(points)
    else:
        keep_mask = np.ones(len(points), dtype=bool)
        
    print("Preparing data for PointSSM and splitting into overlapping blocks (Stride: 25m, Size: 50m)...")
    block_size = 50.0  # MUST be 50 to maintain depth=7 Hilbert curve fractal boundaries
    stride = 25.0      # 50% Overlap to remove edge/tiling artifacts
    
    # Calculate Global Intensity Scale (99.9th percentile to avoid high-intensity noise)
    global_intensity_max = np.percentile(intensities, 99.9)
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
            local_center_z = (np.max(block_points[:, 2]) + np.min(block_points[:, 2])) / 2.0
 
            block_points[:, 0] = (block_points[:, 0] - local_center_x) / 25.0
            block_points[:, 1] = (block_points[:, 1] - local_center_y) / 25.0
            block_points[:, 2] = (block_points[:, 2] - local_center_z) / 25.0
            
            # Spatial weighting for overlap resolving
            dist_x = np.abs(block_points[:, 0])
            dist_y = np.abs(block_points[:, 1])
            weights = 1.0 - np.maximum(dist_x, dist_y)
            weights = np.clip(weights, 0.1, 1.0)
            
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
    print("Reassembling LAS and writing output...")
    # Initialize with 15 (Noise) as default for filtered points
    final_classes = np.full(len(points), 15, dtype=np.uint8)
    
    # Map predictions back to original indices
    final_classes[valid_indices] = pred_classes.astype(np.uint8)
    
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
