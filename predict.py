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
    
    # Default parameters
    method = "SOR"
    nb_neighbors = 20
    std_ratio = 2.0
    radius = 0.5
    min_points = 10
    
    while True:
        print(f"\nCurrent Filter: {method}")
        if method == "SOR":
            print(f"Parameters: Neighbors={nb_neighbors}, StdRatio={std_ratio}")
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        else:
            print(f"Parameters: Radius={radius}, MinPoints={min_points}")
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
        
    # Pre-processing coordinates and intensities
    target_points = points[keep_mask]
    target_intensities = intensities[keep_mask]
    
    # Usually coordinate systems shift the XY mean to 0 and Z min to 0.
    mean_x = np.mean(target_points[:, 0])
    mean_y = np.mean(target_points[:, 1])
    min_z = np.min(target_points[:, 2])
    
    norm_points = target_points.copy()
    norm_points[:, 0] -= mean_x
    norm_points[:, 1] -= mean_y
    norm_points[:, 2] -= min_z
    
    # DALES dataset scale for PointSSM: it expects `coord` and `strength`
    # Let's see if DALES model expects intensities 0-65535.
    # Often models work better with normalized 0-1 intensities or similar, but the DALES dataset 
    # uses 16bit intensities raw. We leave target_intensities as is, as `strength` passes directly.
    
    print("Preparing data for PointSSM...")
    data_dict = {
        "coord": norm_points,
        "strength": target_intensities,
        "segment": np.zeros(len(norm_points), dtype=np.int32),  # Dummy
        "name": os.path.basename(las_file_path)
    }
    
    # Apply test config transform (usually GridSample, ToTensor, Collect)
    data_dict = transform(data_dict)
    
    # Predict in chunks/fragments as done in PointSSM SemSegTester
    # The GridSample in test split creates grid components.
    # Actually, the test_cfg manages voxelization natively if specified.
    # Let's emulate SemSegTester fragment prediction logic.
    
    model.eval()
    
    fragment_list = data_dict["fragment_list"]
    segment = data_dict["segment"]
    pred = torch.zeros((segment.size, cfg.data.num_classes)).cuda()
    
    print("Running inference...")
    for i in range(len(fragment_list)):
        input_dict = fragment_list[i]
        # Wrap things into batch size 1
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

    pred_classes = pred.max(1)[1].data.cpu().numpy()
    
    if "inverse" in data_dict.keys():
        pred_classes = pred_classes[data_dict["inverse"]]
        
    print("Prediction complete.")
    
    # Smoothing
    if args.smoothing == "yes":
        pred_classes = smooth_predictions(norm_points, pred_classes)
        
    # Reassemble and Write LAS
    print("Reassembling LAS and writing output...")
    # Initialize all points with a default unclassified class (e.g. 0)
    final_classes = np.zeros(len(points), dtype=np.uint8)
    # The DALES dataset classes in config:
    # 0: Ground, 1: Veg, 2: Cars, 3: Trucks, 4: Power lines, 5: Fences, 6: Poles, 7: Buildings
    
    from datasets.dales import VALID_CLASS_IDS
    id2class = VALID_CLASS_IDS # maps contiguous id 0..7 to DALES native class (usually 0 is Ground, 1 is Veg, etc.)
    
    mapped_classes = id2class[pred_classes].astype(np.uint8)
    final_classes[keep_mask] = mapped_classes
    
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
                data_part = voxelize_op(data_dict_copy)
                
                # Assume no aug_transform for simple prediction
                data_part = post_transform(data_part)
                fragment_list.append(data_part)
                data_dict["fragment_list"] = fragment_list
            return data_dict
            
    test_transform = TestWrapper(transform, cfg.data.test.test_cfg)
    
    las_files = glob.glob(os.path.join(args.folder, "*.las")) + glob.glob(os.path.join(args.folder, "*.laz"))
    
    for las_file in las_files:
        predict_las(las_file, model, test_transform, cfg, args)
        
if __name__ == "__main__":
    main()
