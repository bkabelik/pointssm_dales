import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
from scipy.spatial import cKDTree

class NoiseFilterApp:
    def __init__(self, points, intensities=None):
        self.points = points
        self.intensities = intensities
        self.mask = np.ones(len(points), dtype=bool)
        
        # Initial stats
        self.z_min = np.min(points[:, 2])
        self.z_max = np.max(points[:, 2])
        self.z_mean = np.mean(points[:, 2])
        
        # Default Filter Params
        self.params = {
            "sor_neighbors": 20,
            "sor_std_ratio": 4.0,
            "ror_radius": 2.0,
            "ror_min_points": 2,
            "dbscan_eps": 0.5,
            "dbscan_min_points": 10,
            "floor_percentile": 1.0,
            "floor_buffer": 0.5,
            "intensity_min": 0.0
        }
        
        self.active_filters = {
            "sor": False,
            "ror": True,
            "dbscan": False,
            "floor": True,
            "intensity": False
        }
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
        self.pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        self.original_colors = self._get_initial_colors()
        self.pcd.colors = o3d.utility.Vector3dVector(self.original_colors)
        
        self.scene.scene.add_geometry("pcd", self.pcd, rendering.MaterialRecord())
        self.scene.setup_camera(60, self.scene.scene.bounding_box, [0, 0, 0])
        
        # UI Panels
        self.panel = gui.Vert(0.5 * em, gui.Margins(0.5 * em))
        
        # Section: Stats
        self.label_stats = gui.Label("Outliers: 0 (0.00%)")
        self.panel.add_child(self.label_stats)
        
        # Section: Tools
        btn_reset = gui.Button("Reset Camera")
        btn_reset.set_on_clicked(lambda: self.scene.setup_camera(60, self.scene.scene.bounding_box, [0, 0, 0]))
        self.panel.add_child(btn_reset)
        
        # Section: Ground Filter
        self.panel.add_child(self._create_filter_section("Ground Elevation Filter", "floor", [
            ("Floor Percentile", "floor_percentile", 0.1, 5.0, 1.0),
            ("Buffer below floor", "floor_buffer", 0.0, 5.0, 0.5)
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
            ("Min Cluster Size", "dbscan_min_points", 2, 500, 10)
        ]))

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

    def _get_initial_colors(self):
        # Default Z-based coloring for visibility
        z = self.points[:, 2]
        z_norm = (z - self.z_min) / (self.z_max - self.z_min + 1e-6)
        colors = np.zeros((len(z), 3))
        colors[:, 0] = 0.2 # Darker base
        colors[:, 1] = z_norm * 0.8 + 0.2
        colors[:, 2] = 0.8
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
        print("Re-calculating filters...")
        combined_mask = np.ones(len(self.points), dtype=bool)
        
        # 1. Floor Filter
        if self.active_filters["floor"]:
            z_floor = np.percentile(self.points[:, 2], self.params["floor_percentile"])
            combined_mask &= (self.points[:, 2] >= z_floor - self.params["floor_buffer"])
            
        # 2. ROR Filter
        if self.active_filters["ror"]:
            # Note: For large clouds, we might want to sample or use a more efficient ROR
            # We use Open3D's built-in ROR for speed
            _, ind = self.pcd.remove_radius_outlier(nb_points=int(self.params["ror_min_points"]), 
                                                   radius=self.params["ror_radius"])
            ror_mask = np.zeros(len(self.points), dtype=bool)
            ror_mask[ind] = True
            combined_mask &= ror_mask
            
        # 3. SOR Filter
        if self.active_filters["sor"]:
            _, ind = self.pcd.remove_statistical_outlier(nb_neighbors=int(self.params["sor_neighbors"]), 
                                                        std_ratio=self.params["sor_std_ratio"])
            sor_mask = np.zeros(len(self.points), dtype=bool)
            sor_mask[ind] = True
            combined_mask &= sor_mask
            
        # 4. DBSCAN Filter
        if self.active_filters["dbscan"]:
            labels = np.array(self.pcd.cluster_dbscan(eps=self.params["dbscan_eps"], 
                                                   min_points=int(self.params["dbscan_min_points"])))
            # Keep only points that are part of a cluster (label >= 0)
            dbscan_mask = (labels >= 0)
            combined_mask &= dbscan_mask

        # 5. Intensity Filter
        if self.active_filters["intensity"] and self.intensities is not None:
            # Normalize intensities if they are not 0-1
            max_i = np.max(self.intensities) + 1e-6
            norm_i = self.intensities / max_i
            combined_mask &= (norm_i >= self.params["intensity_min"])
            
        self.mask = combined_mask
        
        # Update colors in viewer
        new_colors = self.original_colors.copy()
        new_colors[~self.mask] = [1.0, 1.0, 1.0] # Highlight outliers in WHITE
        self.pcd.colors = o3d.utility.Vector3dVector(new_colors)
        
        # Robust update: remove and re-add to ensure the scene sees the new colors
        self.scene.scene.remove_geometry("pcd")
        self.scene.scene.add_geometry("pcd", self.pcd, rendering.MaterialRecord())
        self.window.post_redraw()
        
        # Update stats
        outlier_count = np.sum(~self.mask)
        percent = (outlier_count / len(self.points)) * 100
        self.label_stats.text = f"Outliers: {outlier_count:,} ({percent:.2f}%)"

    def _on_confirm_tile(self):
        self.apply_to_all = False
        gui.Application.instance.quit()

    def _on_confirm_all(self):
        self.apply_to_all = True
        gui.Application.instance.quit()

    def run(self):
        gui.Application.instance.run()
        return self.mask, self.params, self.active_filters, self.apply_to_all

def run_interactive_filter(points, intensities=None):
    app = NoiseFilterApp(points, intensities)
    return app.run()

def apply_headless_filter(points, intensities, params, active_filters):
    """
    Run the noise filters without a GUI using provided parameters.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    combined_mask = np.ones(len(points), dtype=bool)
    
    if active_filters.get("floor", False):
        z_floor = np.percentile(points[:, 2], params["floor_percentile"])
        combined_mask &= (points[:, 2] >= z_floor - params["floor_buffer"])
        
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
        
    if active_filters.get("dbscan", False):
        labels = np.array(pcd.cluster_dbscan(eps=params["dbscan_eps"], 
                                            min_points=int(params["dbscan_min_points"])))
        combined_mask &= (labels >= 0)
        
    if active_filters.get("intensity", False) and intensities is not None:
        max_i = np.max(intensities) + 1e-6
        combined_mask &= (intensities / max_i >= params["intensity_min"])
        
    return combined_mask
