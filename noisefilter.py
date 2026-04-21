import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
from scipy.spatial import cKDTree

def build_ground_model(points, return_num, total_returns, grid_size=2.0, fixed_min=None, fixed_max=None):
    """
    Build a Digital Terrain Model (DTM) from 'last return' points.
    Includes 1-cell padding and edge-aware interpolation.
    """
    # Use only last returns for ground detection
    ground_mask = (return_num == total_returns)
    ground_pts = points[ground_mask]
    
    if len(ground_pts) < 100:
        ground_pts = points # Fallback
        
    if fixed_min is not None and fixed_max is not None:
        min_x, min_y = fixed_min[0] - grid_size, fixed_min[1] - grid_size # 1-cell padding
        max_x, max_y = fixed_max[0] + grid_size, fixed_max[1] + grid_size
    else:
        min_x, min_y = np.min(ground_pts[:, 0]) - grid_size, np.min(ground_pts[:, 1]) - grid_size
        max_x, max_y = np.max(points[:, 0]) + grid_size, np.max(points[:, 1]) + grid_size
        
    nx = int(np.ceil((max_x - min_x) / grid_size)) + 1
    ny = int(np.ceil((max_y - min_y) / grid_size)) + 1
    
    # Initialize with NaN
    grid = np.full((nx, ny), np.nan, dtype=np.float32)
    gx = ((ground_pts[:, 0] - min_x) / grid_size).astype(int)
    gy = ((ground_pts[:, 1] - min_y) / grid_size).astype(int)
    gx = np.clip(gx, 0, nx - 1)
    gy = np.clip(gy, 0, ny - 1)
    
    # Fill cells with lowest point (Vectorized)
    temp_grid = np.full((nx, ny), 1e6, dtype=np.float32)
    np.minimum.at(temp_grid, (gx, gy), ground_pts[:, 2])
    mask = (temp_grid < 1e5)
    grid[mask] = temp_grid[mask]
    
    # --- Hole Filling (Interpolate with Boundary Protection) ---
    for _ in range(5): 
        valid = ~np.isnan(grid)
        if not np.any(~valid): break 
        
        shifted_sum = np.zeros_like(grid)
        count = np.zeros_like(grid)
        
        # Safe neighbor-averaging with boundary protection (No wrap-around)
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            # Create a shifted version that doesn't wrap
            s = np.full_like(grid, np.nan)
            if dx == 1: s[1:, :] = grid[:-1, :]
            elif dx == -1: s[:-1, :] = grid[1:, :]
            elif dy == 1: s[:, 1:] = grid[:, :-1]
            elif dy == -1: s[:, :-1] = grid[:, 1:]
            
            m = ~np.isnan(s)
            shifted_sum[m] += s[m]
            count[m] += 1
        
        update_mask = (~valid) & (count > 0)
        grid[update_mask] = shifted_sum[update_mask] / count[update_mask]
        
    # Final fallback for any remaining holes
    if np.any(np.isnan(grid)):
        grid[np.isnan(grid)] = np.nanpercentile(grid, 5.0)

    return grid, (min_x, min_y), grid_size

def get_heights_above_ground(points, ground_model):
    grid, (min_x, min_y), grid_size = ground_model
    gx = ((points[:, 0] - min_x) / grid_size).astype(int)
    gy = ((points[:, 1] - min_y) / grid_size).astype(int)
    gx = np.clip(gx, 0, grid.shape[0] - 1)
    gy = np.clip(gy, 0, grid.shape[1] - 1)
    return points[:, 2] - grid[gx, gy]

def calculate_cluster_linearity(points):
    """
    Calculate linearity for a set of points using eigenvalues of the covariance matrix.
    L = (e1 - e2) / e1. Near 1.0 for lines (wires), near 0.0 for blobs.
    """
    if len(points) < 5: # Too few points to reliably determine linearity
        return 0
    try:
        cov = np.cov(points, rowvar=False)
        evals = np.linalg.eigvalsh(cov) # Sorted ascending: e3, e2, e1
        e1, e2 = evals[2], evals[1]
        if e1 < 1e-9: return 0
        return (e1 - e2) / e1
    except:
        return 0
class NoiseFilterApp:
    def __init__(self, points, intensities=None, initial_params=None, initial_active=None, 
                 return_num=None, total_returns=None):
        self.points = points
        self.intensities = intensities
        self.return_num = return_num
        self.total_returns = total_returns
        self.mask = np.ones(len(points), dtype=bool)
        
        # Initial stats - Robust
        self.points_mean = np.mean(points, axis=0)
        self.points_local = points - self.points_mean # Local coords for rendering precision
        self.heights = None
        self.ground_pcd = None
        
        self.z_min = np.min(self.points_local[:, 2])
        self.z_max = np.max(self.points_local[:, 2])
        self.full_min = np.min(self.points_local, axis=0) # Lock DTM extents
        self.full_max = np.max(self.points_local, axis=0)
        
        # Default Filter Params
        self.params = {
            "sor_neighbors": 20,
            "sor_std_ratio": 4.0,
            "ror_radius": 0.5,           # Reduced default for speed
            "ror_min_points": 2,
            "dbscan_eps": 0.5,
            "dbscan_min_points": 10,
            "dbscan_base_alt": 5.0,
            "dbscan_linearity": 0.85,
            "floor_percentile": 1.0,
            "floor_buffer": 0.5,
            "intensity_min": 0.0,
            "max_gui_points": 25000000
        }
        
        # Override with initial params if provided
        if initial_params:
            self.params.update(initial_params)
        
        self.active_filters = {
            "sor": False,
            "ror": False, # OFF by default for instant launch
            "dbscan": False,
            "floor": False, # OFF by default for instant launch
            "intensity": False
        }
        
        if initial_active:
            self.active_filters.update(initial_active)
        self.apply_to_all = False

        # Setup GUI
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("Advanced Point Cloud Noise Filter", 1280, 800)
        
        # Theme
        em = self.window.theme.font_size
        
        # Scene Widget
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([0.05, 0.05, 0.05, 1.0]) # Dark background
        
        # Point Cloud
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.points_local[:, :3])
        self.original_colors = self._get_initial_colors()
        self.pcd.colors = o3d.utility.Vector3dVector(self.original_colors)
        
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit" # FLAT unlit points like CloudCompare
        self.mat.point_size = 3.0 # Thicker by default
        
        self.scene.scene.add_geometry("pcd", self.pcd, self.mat)
        
        if self.ground_pcd:
            self.scene.scene.add_geometry("ground", self.ground_pcd, rendering.MaterialRecord())
            self.scene.scene.show_geometry("ground", True) # ON BY DEFAULT
            
        self._reset_camera()
        
        # UI Panels
        self.panel = gui.Vert(0.5 * em, gui.Margins(0.5 * em))
        
        # Section: Stats
        self.label_stats = gui.Label(f"Points in GUI: {len(points):,}")
        self.panel.add_child(self.label_stats)
        
        self.label_outliers = gui.Label("Outliers: 0 (0.00%)")
        self.panel.add_child(self.label_outliers)
        
        # Section: Tools
        btn_reset = gui.Button("Reset Camera (Robust)")
        btn_reset.set_on_clicked(self._reset_camera)
        self.panel.add_child(btn_reset)
        
        # Point Size Slider
        hbox_ps = gui.Horiz(0.5 * em)
        hbox_ps.add_child(gui.Label("Point Size: "))
        slider_ps = gui.Slider(gui.Slider.INT)
        slider_ps.set_limits(1, 10)
        slider_ps.double_value = 3.0
        def on_ps_change(val):
            # Safe way to update point size without crashing: 
            # Modify and trigger a redraw/geometry update
            self.mat.point_size = int(val)
            self.scene.scene.remove_geometry("pcd")
            self.scene.scene.add_geometry("pcd", self.pcd, self.mat)
        slider_ps.set_on_value_changed(on_ps_change)
        hbox_ps.add_child(slider_ps)
        self.panel.add_child(hbox_ps)
        
        # Ground Visualization Toggle
        cb_ground = gui.Checkbox("Show Ground Terrain (Green)")
        cb_ground.checked = True # CHECKED BY DEFAULT
        def on_ground_toggle(val):
            if self.ground_pcd:
                self.scene.scene.show_geometry("ground", val)
        cb_ground.set_on_checked(on_ground_toggle)
        self.panel.add_child(cb_ground)
        
        # Section: Ground Filter
        self.panel.add_child(self._create_filter_section("Ground Elevation Filter", "floor", [
            ("Allowance below ground (m)", "floor_buffer", 0.0, 10.0, 0.5)
        ]))
        
        # Section: ROR
        self.panel.add_child(self._create_filter_section("Radius Outlier (ROR)", "ror", [
            ("Radius (m)", "ror_radius", 0.1, 10.0, 2.0),
            ("Min Points", "ror_min_points", 1, 50, 2)
        ]))
        
        # Section: SOR
        self.panel.add_child(self._create_filter_section("Statistical Outlier (SOR)", "sor", [
            ("Neighbors", "sor_neighbors", 5, 100, 20),
            ("Std Ratio", "sor_std_ratio", 0.1, 10.0, 4.0)
        ]))
        
        # Section: DBSCAN
        self.panel.add_child(self._create_filter_section("DBSCAN (Air Clusters)", "dbscan", [
            ("Eps distance", "dbscan_eps", 0.1, 5.0, 0.5),
            ("Min Cluster Size", "dbscan_min_points", 2, 500, 10),
            ("Base Altitude (m)", "dbscan_base_alt", 0.0, 50.0, 5.0),
            ("Linearity Protect", "dbscan_linearity", 0.0, 1.0, 0.85)
        ]))

        # Section: Sampling (Info only)
        section_samp = gui.CollapsableVert("Sampling Settings", 0.25 * em, gui.Margins(em, 0, 0, 0))
        hbox_samp = gui.Horiz(0.5 * em)
        hbox_samp.add_child(gui.Label("GUI Limit: "))
        slider_samp = gui.Slider(gui.Slider.INT)
        slider_samp.set_limits(1000000, 50000000) # Increased to 50M
        slider_samp.double_value = float(self.params["max_gui_points"])
        def on_samp_change(val):
            self.params["max_gui_points"] = int(val)
        slider_samp.set_on_value_changed(on_samp_change)
        hbox_samp.add_child(slider_samp)
        section_samp.add_child(hbox_samp)
        section_samp.add_child(gui.Label("(Apply on next file restart)"))
        self.panel.add_child(section_samp)

        # Section: Intensity
        self.panel.add_child(self._create_filter_section("Intensity Filter", "intensity", [
            ("Min Intensity", "intensity_min", 0.0, 1.0, 0.0)
        ]))

        # Confirm Buttons
        hbox_btns = gui.Horiz(0.5 * em)
        btn_tile = gui.Button("Accept this tile")
        btn_tile.set_on_clicked(self._on_confirm_tile)
        hbox_btns.add_child(btn_tile)
        
        btn_all = gui.Button("Use for ALL tiles")
        btn_all.set_on_clicked(self._on_confirm_all)
        hbox_btns.add_child(btn_all)
        
        self.panel.add_child(hbox_btns)
        
        self.window.add_child(self.scene)
        self.window.add_child(self.panel)
        
        # Set layout
        self.window.set_on_layout(self._on_layout)
        
        # Initial Update
        self._update_all()

    def _reset_camera(self):
        # Focus on the 'Real' data by calculating a robust bounding box (1st to 99th percentile)
        # This ignores extreme air/ground noise that zooms the camera out too far
        try:
            q_min = np.percentile(self.points_local, 1, axis=0)
            q_max = np.percentile(self.points_local, 99, axis=0)
            
            center = (q_min + q_max) / 2.0
            size = (q_max - q_min)
            # Create a robust box for camera focus
            bbox = o3d.geometry.AxisAlignedBoundingBox(q_min, q_max)
            self.scene.setup_camera(30, bbox, center) # 30 degree FOV for technical look
        except:
            self.scene.setup_camera(30, self.scene.scene.bounding_box, [0, 0, 0])

    def _get_initial_colors(self):
        # 1. Vibrant Turquoise/Blue Height Ramp (User Choice for Noise Filtering)
        z = self.points_local[:, 2]
        z_min = np.percentile(z, 1)
        z_max = np.percentile(z, 99)
        z_norm = np.clip((z - z_min) / (z_max - z_min + 1e-6), 0, 1)
        
        # Original-style vibrant Turquoise scheme
        colors = np.zeros((len(z), 3))
        colors[:, 0] = z_norm * 0.2 + 0.1 # Darker start
        colors[:, 1] = z_norm * 0.7 + 0.3 # Turquoise green-blue
        colors[:, 2] = 0.8               # Strong blue base
        return colors

    def _create_filter_section(self, title, key, sliders):
        em = self.window.theme.font_size
        section = gui.CollapsableVert(title, 0.25 * em, gui.Margins(em, 0, 0, 0))
        
        cb = gui.Checkbox("Enabled")
        cb.checked = self.active_filters[key]
        def on_toggle(checked):
            self.active_filters[key] = checked
            self._update_all()
        cb.set_on_checked(on_toggle)
        section.add_child(cb)
        
        for name, p_key, v_min, v_max, v_init in sliders:
            hbox = gui.Horiz(0.5 * em)
            hbox.add_child(gui.Label(f"{name}: "))
            
            slider = gui.Slider(gui.Slider.DOUBLE if isinstance(v_init, float) else gui.Slider.INT)
            slider.set_limits(v_min, v_max)
            slider.double_value = float(v_init)
            
            def on_change(val, k=p_key):
                self.params[k] = val
                self._update_all()
            slider.set_on_value_changed(on_change)
            
            hbox.add_child(slider)
            section.add_child(hbox)
            
        return section

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_width = 300 # Standard sidebar width
        self.panel.frame = gui.Rect(r.x, r.y, panel_width, r.height)
        self.scene.frame = gui.Rect(r.x + panel_width, r.y, r.width - panel_width, r.height)

    def _update_all(self):
        print("Re-calculating Multi-Pass Filters...")
        self.mask = np.ones(len(self.points), dtype=bool)
        
        # --- PASS 1: Noise Removal (Detect clusters/outliers to clean the DTM) ---
        
        # 1. Intensity Filter
        if self.active_filters["intensity"] and self.intensities is not None:
            max_i = np.max(self.intensities) + 1e-6
            self.mask &= (self.intensities / max_i >= self.params["intensity_min"])

        # 2. ROR Filter
        if self.active_filters["ror"]:
            _, ind = self.pcd.remove_radius_outlier(nb_points=int(self.params["ror_min_points"]), 
                                                   radius=self.params["ror_radius"])
            m = np.zeros(len(self.points), dtype=bool); m[ind] = True
            self.mask &= m
            
        # 3. SOR Filter
        if self.active_filters["sor"]:
            _, ind = self.pcd.remove_statistical_outlier(nb_neighbors=int(self.params["sor_neighbors"]), 
                                                        std_ratio=self.params["sor_std_ratio"])
            m = np.zeros(len(self.points), dtype=bool); m[ind] = True
            self.mask &= m

        # --- PASS 2: Dynamic DTM Refinement (Calculate Ground from Clean Points) ---
        
        valid_indices = np.where(self.mask)[0]
        if len(valid_indices) > 100 and self.return_num is not None:
            # Build DTM using ONLY points that survived Pass 1
            dtm_grid, dtm_meta, dtm_size = build_ground_model(
                self.points_local[valid_indices], 
                self.return_num[valid_indices], 
                self.total_returns[valid_indices],
                fixed_min=self.full_min,
                fixed_max=self.full_max # LOCK BOTH SIDES
            )
            # Apply DTM to FULL point set for the Elevation Filter
            gx = ((self.points_local[:, 0] - dtm_meta[0]) / dtm_size).astype(int)
            gy = ((self.points_local[:, 1] - dtm_meta[1]) / dtm_size).astype(int)
            gx = np.clip(gx, 0, dtm_grid.shape[0] - 1)
            gy = np.clip(gy, 0, dtm_grid.shape[1] - 1)
            self.heights = self.points_local[:, 2] - dtm_grid[gx, gy]
            
            # Update the Green Grid Visualization
            res_x, res_y = dtm_grid.shape
            grid_pts = []
            stride = max(1, res_x // 50) # Keep visualization sparse for speed
            for i in range(0, res_x, stride):
                for j in range(0, res_y, stride):
                    grid_pts.append([dtm_meta[0] + i*dtm_size, dtm_meta[1] + j*dtm_size, dtm_grid[i, j]])
            
            if self.ground_pcd is None:
                self.ground_pcd = o3d.geometry.PointCloud()

            self.ground_pcd.points = o3d.utility.Vector3dVector(np.array(grid_pts))
            self.ground_pcd.paint_uniform_color([0.0, 1.0, 0.0]) # Neon Green (Applied AFTER point update)
            self.scene.scene.remove_geometry("ground")
            self.scene.scene.add_geometry("ground", self.ground_pcd, rendering.MaterialRecord())

        # --- PASS 3: Ground Elevation & High-Altitude Clusters ---

        # 4. Ground Elevation Filter (Using the Refined DTM)
        if self.active_filters["floor"] and self.heights is not None:
            self.mask &= (self.heights >= -self.params["floor_buffer"])
            
        # 5. DBSCAN (Air Clusters) - Run on Ground-Filtered points to avoid ground confusion
        if self.active_filters["dbscan"] and self.heights is not None:
            # Detect noise clusters above the ground
            noise_cand_idx = np.where(self.mask & (self.heights > self.params["dbscan_base_alt"]))[0]
            if len(noise_cand_idx) > 0:
                pcd_temp = o3d.geometry.PointCloud()
                pcd_temp.points = o3d.utility.Vector3dVector(self.points_local[noise_cand_idx])
                labels = np.array(pcd_temp.cluster_dbscan(eps=self.params["dbscan_eps"], 
                                                        min_points=int(self.params["dbscan_min_points"])))
                
                subset_noise = (labels < 0)
                for i in range(labels.max() + 1):
                    cluster_indices = np.where(labels == i)[0]
                    # Power Line Protection (Linearity)
                    lin = calculate_cluster_linearity(self.points_local[noise_cand_idx[cluster_indices]])
                    if lin < self.params["dbscan_linearity"]:
                        subset_noise[cluster_indices] = True
                
                m = np.ones(len(self.points), dtype=bool)
                m[noise_cand_idx[subset_noise]] = False
                self.mask &= m

        # Update colors in viewer
        new_colors = self.original_colors.copy()
        new_colors[~self.mask] = [1.0, 0.0, 1.0] # Highlight outliers in Vibrant MAGENTA
        self.pcd.colors = o3d.utility.Vector3dVector(new_colors)
        
        # Robust update: remove and re-add to ensure the scene sees the new colors
        self.scene.scene.remove_geometry("pcd")
        self.scene.scene.add_geometry("pcd", self.pcd, self.mat)
        self.window.post_redraw()
        
        # Update stats
        outlier_count = np.sum(~self.mask)
        percent = (outlier_count / len(self.points)) * 100
        self.label_outliers.text = f"Outliers: {outlier_count:,} ({percent:.2f}%)"

    def _on_confirm_tile(self):
        self.apply_to_all = False
        gui.Application.instance.quit()

    def _on_confirm_all(self):
        self.apply_to_all = True
        gui.Application.instance.quit()

    def run(self):
        gui.Application.instance.run()
        return self.mask, self.params, self.active_filters, self.apply_to_all

def run_interactive_filter(points, intensities=None, initial_params=None, initial_active=None, return_num=None, total_returns=None):
    app = NoiseFilterApp(points, intensities, initial_params, initial_active, 
                         return_num=return_num, total_returns=total_returns)
    return app.run()

def apply_headless_filter(points, intensities, params, active_filters, return_num=None, total_returns=None):
    """
    Run the noise filters with Dynamic DTM Re-calculation.
    Sequence: Noise Removal -> Refine DTM -> Elevation Filter
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    combined_mask = np.ones(len(points), dtype=bool)
    
    # 1. Pre-DTM Noise Filtering
    if active_filters.get("intensity", False) and intensities is not None:
        max_i = np.max(intensities) + 1e-6
        combined_mask &= (intensities / max_i >= params["intensity_min"])

    if active_filters.get("ror", False):
        _, ind = pcd.remove_radius_outlier(nb_points=int(params["ror_min_points"]), 
                                            radius=params["ror_radius"])
        m = np.zeros(len(points), dtype=bool); m[ind] = True
        combined_mask &= m
        
    if active_filters.get("sor", False):
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=int(params["sor_neighbors"]), 
                                                std_ratio=params["sor_std_ratio"])
        m = np.zeros(len(points), dtype=bool); m[ind] = True
        combined_mask &= m
    
    # 2. Refined DTM Build from partially cleaned data
    valid_idx = np.where(combined_mask)[0]
    ground_model = build_ground_model(points[valid_idx], return_num[valid_idx], total_returns[valid_idx])
    heights = get_heights_above_ground(points, ground_model)
    
    # 3. Post-DTM Filtering
    if active_filters.get("floor", False):
        combined_mask &= (heights >= -params["floor_buffer"])
        
    if active_filters.get("dbscan", False):
        noise_cand_idx = np.where(combined_mask & (heights > params["dbscan_base_alt"]))[0]
        if len(noise_cand_idx) > 0:
            pcd_temp = o3d.geometry.PointCloud()
            pcd_temp.points = o3d.utility.Vector3dVector(points[noise_cand_idx])
            labels = np.array(pcd_temp.cluster_dbscan(eps=params["dbscan_eps"], 
                                                    min_points=int(params["dbscan_min_points"])))
            subset_noise = (labels < 0)
            for i in range(labels.max() + 1):
                c_idx = np.where(labels == i)[0]
                lin = calculate_cluster_linearity(points[noise_cand_idx[c_idx]])
                if lin < params["dbscan_linearity"]:
                    subset_noise[c_idx] = True
            combined_mask[noise_cand_idx[subset_noise]] = False
        
    return combined_mask
