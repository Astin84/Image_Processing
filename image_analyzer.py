# -*- coding: utf-8 -*-

import sys
import numpy as np
import cv2

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QLabel,
    QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox, QTextEdit,
    QMessageBox, QSizePolicy, QFrame,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PySide6.QtGui import QPixmap, QImage, QWheelEvent, QFont
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def bgr_to_qpixmap(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def compute_hsi(rgb):
    eps = 1e-8
    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]

    I = (R + G + B) / 3.0

    min_rgb = np.minimum(np.minimum(R, G), B)
    sum_rgb = R + G + B
    S = 1.0 - (3.0 * min_rgb / (sum_rgb + eps))
    S = np.clip(S, 0.0, 1.0)

    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + eps
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))
    H = np.degrees(theta)
    H = np.where(B > G, 360.0 - H, H)
    H = np.mod(H, 360.0)

    return H, S, I


def channel_stats(arr, name):
    return (
        f"{name}:\n"
        f"  min:  {np.min(arr):.4f}\n"
        f"  max:  {np.max(arr):.4f}\n"
        f"  mean: {np.mean(arr):.4f}\n"
        f"  std:  {np.std(arr):.4f}\n"
    )


class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(7, 4), dpi=100, facecolor='#ECE9D8')
        self.ax = self.fig.add_subplot(111, facecolor='#ECE9D8')
        super().__init__(self.fig)


class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def wheelEvent(self, event: QWheelEvent):
        zoom_in = 1.25
        zoom_out = 1 / zoom_in
        if event.angleDelta().y() > 0:
            self.scale(zoom_in, zoom_in)
        else:
            self.scale(zoom_out, zoom_out)


class ImageAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Analyzer - Windows XP Style")
        self.resize(1400, 800)

        self.original_bgr = None
        self.working_bgr = None
        self.visualization_mode = "combined"

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Sidebar - Windows XP Task Pane Style
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFrameShape(QFrame.StyledPanel)
        sidebar.setFrameShadow(QFrame.Raised)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setSpacing(6)
        sidebar_layout.setContentsMargins(4, 8, 4, 8)
        main_layout.addWidget(sidebar, 0)

        # Windows XP style title
        title_label = QLabel("Image Tasks")
        title_label.setObjectName("TaskTitle")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont("Tahoma", 10, QFont.Bold)
        title_label.setFont(title_font)
        sidebar_layout.addWidget(title_label)

        # File tasks group
        file_group = QGroupBox("File Tasks")
        file_group.setObjectName("FileGroup")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(4)
        file_layout.setContentsMargins(8, 15, 8, 8)

        self.btn_import = self.create_xp_button("📂 Open image...", self.load_image)
        self.btn_reset = self.create_xp_button("↩️ Restore original", self.reset_to_original)
        self.btn_reset.setEnabled(False)
        self.btn_export = self.create_xp_button("💾 Save image as...", self.export_image)
        self.btn_export.setEnabled(False)
        self.btn_clear = self.create_xp_button("🗑️ Clear workspace", self.clear_all)

        file_layout.addWidget(self.btn_import)
        file_layout.addWidget(self.btn_reset)
        file_layout.addWidget(self.btn_export)
        file_layout.addWidget(self.btn_clear)
        
        sidebar_layout.addWidget(file_group)

        # Visualization tasks group
        vis_group = QGroupBox("View Options")
        vis_group.setObjectName("VisGroup")
        vis_layout = QVBoxLayout(vis_group)
        vis_layout.setSpacing(4)
        vis_layout.setContentsMargins(8, 15, 8, 8)

        self.btn_visualization = self.create_xp_button("📊 Change view type", None)
        self.btn_visualization.setCheckable(True)
        self.btn_visualization.clicked.connect(self.toggle_visualization_panel)
        self.btn_visualization.setEnabled(False)
        vis_layout.addWidget(self.btn_visualization)

        self.visualization_panel = QFrame()
        self.visualization_panel.setObjectName("SubPanel")
        vis_sub_layout = QVBoxLayout(self.visualization_panel)
        vis_sub_layout.setSpacing(2)
        vis_sub_layout.setContentsMargins(20, 6, 6, 6)

        self.btn_show_hsi = self.create_sub_button("🎨 HSI Components", "hsi")
        self.btn_show_histogram = self.create_sub_button("📈 Histogram only", "histogram")
        self.btn_show_rgb = self.create_sub_button("🌈 RGB Channels", "rgb")
        self.btn_show_combined = self.create_sub_button("🔗 Combined view", "combined")

        vis_sub_layout.addWidget(self.btn_show_hsi)
        vis_sub_layout.addWidget(self.btn_show_histogram)
        vis_sub_layout.addWidget(self.btn_show_rgb)
        vis_sub_layout.addWidget(self.btn_show_combined)

        self.visualization_panel.setVisible(False)
        vis_layout.addWidget(self.visualization_panel)
        
        sidebar_layout.addWidget(vis_group)

        # Image processing tasks group
        proc_group = QGroupBox("Image Processing")
        proc_group.setObjectName("ProcGroup")
        proc_layout = QVBoxLayout(proc_group)
        proc_layout.setSpacing(4)
        proc_layout.setContentsMargins(8, 15, 8, 8)

        self.btn_ops = self.create_xp_button("⚙️ Basic operations", None)
        self.btn_ops.setCheckable(True)
        self.btn_ops.clicked.connect(self.toggle_ops_panel)
        self.btn_ops.setEnabled(False)
        proc_layout.addWidget(self.btn_ops)

        self.ops_panel = QFrame()
        self.ops_panel.setObjectName("SubPanel")
        ops_sub_layout = QVBoxLayout(self.ops_panel)
        ops_sub_layout.setSpacing(2)
        ops_sub_layout.setContentsMargins(20, 6, 6, 6)

        self.btn_smooth = self.create_sub_button("🔹 Smooth image", self.apply_smoothing)
        self.btn_sharp = self.create_sub_button("🔸 Sharpen image", self.apply_sharpening)
        self.btn_histeq = self.create_sub_button("📊 Equalize histogram", self.apply_hist_equalization)

        ops_sub_layout.addWidget(self.btn_smooth)
        ops_sub_layout.addWidget(self.btn_sharp)
        ops_sub_layout.addWidget(self.btn_histeq)

        self.ops_panel.setVisible(False)
        proc_layout.addWidget(self.ops_panel)
        
        sidebar_layout.addWidget(proc_group)

        # Advanced filters group
        filter_group = QGroupBox("Advanced Filters")
        filter_group.setObjectName("FilterGroup")
        filter_layout = QVBoxLayout(filter_group)
        filter_layout.setSpacing(4)
        filter_layout.setContentsMargins(8, 15, 8, 8)

        self.btn_filters = self.create_xp_button("🎛️ Frequency filters", None)
        self.btn_filters.setCheckable(True)
        self.btn_filters.clicked.connect(self.toggle_filters_panel)
        self.btn_filters.setEnabled(False)
        filter_layout.addWidget(self.btn_filters)

        self.filters_panel = QFrame()
        self.filters_panel.setObjectName("SubPanel")
        filters_sub_layout = QVBoxLayout(self.filters_panel)
        filters_sub_layout.setSpacing(2)
        filters_sub_layout.setContentsMargins(20, 6, 6, 6)

        self.btn_lowpass = self.create_sub_button("🔽 Low Pass filter", lambda: self.apply_filter("lowpass"))
        self.btn_highpass = self.create_sub_button("🔼 High Pass filter", lambda: self.apply_filter("highpass"))
        self.btn_notchpass = self.create_sub_button("🎯 Notch Pass filter", lambda: self.apply_filter("notchpass"))
        self.btn_notchreject = self.create_sub_button("🚫 Notch Reject filter", lambda: self.apply_filter("notchreject"))
        self.btn_gaussian = self.create_sub_button("⛰️ Gaussian filter", lambda: self.apply_filter("gaussian"))

        filters_sub_layout.addWidget(self.btn_lowpass)
        filters_sub_layout.addWidget(self.btn_highpass)
        filters_sub_layout.addWidget(self.btn_notchpass)
        filters_sub_layout.addWidget(self.btn_notchreject)
        filters_sub_layout.addWidget(self.btn_gaussian)

        self.filters_panel.setVisible(False)
        filter_layout.addWidget(self.filters_panel)
        
        sidebar_layout.addWidget(filter_group)

        # Status bar at bottom
        status_frame = QFrame()
        status_frame.setObjectName("StatusFrame")
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(6, 4, 6, 4)
        
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("StatusLabel")
        status_font = QFont("Tahoma", 8)
        self.status_label.setFont(status_font)
        status_layout.addWidget(self.status_label)
        
        sidebar_layout.addStretch(1)
        sidebar_layout.addWidget(status_frame)

        # Content area - Windows XP Work Area
        content = QFrame()
        content.setObjectName("Content")
        content.setFrameShape(QFrame.StyledPanel)
        content.setFrameShadow(QFrame.Sunken)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(8)
        main_layout.addWidget(content, 1)

        # Toolbar-like area
        toolbar = QFrame()
        toolbar.setObjectName("Toolbar")
        toolbar.setFixedHeight(36)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(6, 2, 6, 2)
        toolbar_layout.setSpacing(4)

        self.btn_fit = self.create_tool_button("🔍 Fit", self.fit_to_window)
        self.btn_fit.setEnabled(False)
        self.btn_reset_zoom = self.create_tool_button("🗺️ 1:1", self.reset_zoom)
        self.btn_reset_zoom.setEnabled(False)
        toolbar_layout.addWidget(self.btn_fit)
        toolbar_layout.addWidget(self.btn_reset_zoom)
        toolbar_layout.addStretch(1)
        
        view_label = QLabel("Image View")
        view_label.setFont(title_font)
        toolbar_layout.addWidget(view_label)
        toolbar_layout.addStretch(1)

        content_layout.addWidget(toolbar)

        # Main work area (image and stats)
        work_area = QHBoxLayout()
        work_area.setSpacing(8)
        content_layout.addLayout(work_area, 3)

        # Image preview area
        preview_frame = QFrame()
        preview_frame.setObjectName("PreviewFrame")
        preview_frame.setFrameShape(QFrame.StyledPanel)
        preview_frame.setFrameShadow(QFrame.Sunken)
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(4, 4, 4, 4)

        self.scene = QGraphicsScene(self)
        self.image_view = ZoomableGraphicsView(self)
        self.image_view.setObjectName("ImageView")
        self.image_view.setScene(self.scene)
        self.pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pix_item)

        preview_layout.addWidget(self.image_view)
        work_area.addWidget(preview_frame, 3)

        # Statistics panel
        stats_frame = QFrame()
        stats_frame.setObjectName("StatsFrame")
        stats_frame.setFrameShape(QFrame.StyledPanel)
        stats_frame.setFrameShadow(QFrame.Raised)
        stats_layout = QVBoxLayout(stats_frame)
        stats_layout.setContentsMargins(8, 8, 8, 8)

        stats_title = QLabel("Image Statistics")
        stats_title.setObjectName("StatsTitle")
        stats_title.setFont(title_font)
        stats_title.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(stats_title)

        self.stats_box = QTextEdit()
        self.stats_box.setObjectName("StatsBox")
        self.stats_box.setReadOnly(True)
        stats_layout.addWidget(self.stats_box)
        
        work_area.addWidget(stats_frame, 2)

        # Histogram/Visualization area
        vis_frame = QFrame()
        vis_frame.setObjectName("VisFrame")
        vis_frame.setFrameShape(QFrame.StyledPanel)
        vis_frame.setFrameShadow(QFrame.Raised)
        vis_layout = QVBoxLayout(vis_frame)
        vis_layout.setContentsMargins(8, 8, 8, 8)

        vis_title = QLabel("Visualization")
        vis_title.setObjectName("VisTitle")
        vis_title.setFont(title_font)
        vis_title.setAlignment(Qt.AlignCenter)
        vis_layout.addWidget(vis_title)

        self.canvas = MplCanvas()
        vis_layout.addWidget(self.canvas)
        
        content_layout.addWidget(vis_frame, 2)

        # Windows XP StyleSheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #D4D0C8;
            }
            
            QFrame#Sidebar {
                background-color: #ECE9D8;
                border: 1px solid #808080;
                border-radius: 4px;
            }
            
            QFrame#Content {
                background-color: #FFFFFF;
                border: 2px solid #808080;
            }
            
            QFrame#PreviewFrame, QFrame#StatsFrame, QFrame#VisFrame {
                background-color: #FFFFFF;
            }
            
            QFrame#Toolbar {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ECE9D8, stop:1 #D4D0C8);
                border: 1px solid #808080;
                border-radius: 3px;
            }
            
            QFrame#StatusFrame {
                background-color: #ECE9D8;
                border: 1px solid #808080;
                border-radius: 3px;
            }
            
            QFrame#SubPanel {
                background-color: #F8F8F8;
                border: 1px solid #C0C0C0;
                border-radius: 2px;
            }
            
            QGroupBox {
                color: #000080;
                font-weight: bold;
                border: 1px solid #808080;
                border-radius: 5px;
                margin-top: 10px;
                background-color: #ECE9D8;
            }
            
            QGroupBox#FileGroup, QGroupBox#VisGroup, QGroupBox#ProcGroup, QGroupBox#FilterGroup {
                background-color: #ECE9D8;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                background-color: #ECE9D8;
            }
            
            /* Windows XP Style Buttons */
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F9F9F9, stop:0.1 #E8E8E8, stop:0.5 #E0E0E0, stop:0.9 #D8D8D8, stop:1 #C8C8C8);
                border: 1px solid #808080;
                border-radius: 3px;
                padding: 4px 12px;
                font-family: Tahoma;
                font-size: 9pt;
                color: #000000;
                text-align: left;
                min-height: 24px;
            }
            
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FFFFFF, stop:0.1 #F0F0F0, stop:0.5 #E8E8E8, stop:0.9 #E0E0E0, stop:1 #D0D0D0);
                border: 1px solid #000080;
            }
            
            QPushButton:pressed {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #C8C8C8, stop:0.1 #D8D8D8, stop:0.5 #E0E0E0, stop:0.9 #E8E8E8, stop:1 #F9F9F9);
                padding: 5px 11px 3px 13px;
            }
            
            QPushButton:disabled {
                background-color: #E0E0E0;
                color: #808080;
                border: 1px solid #C0C0C0;
            }
            
            /* Toolbar buttons */
            QPushButton#ToolButton {
                padding: 3px 8px;
                min-width: 60px;
                text-align: center;
            }
            
            /* Sub-panel buttons */
            QPushButton#SubButton {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 2px;
                padding: 2px 8px;
                font-size: 8.5pt;
                min-height: 20px;
            }
            
            QPushButton#SubButton:hover {
                background-color: #E8F0FF;
                border: 1px solid #316AC5;
            }
            
            QPushButton#SubButton:pressed {
                background-color: #C8D8F8;
                padding: 3px 7px 1px 9px;
            }
            
            /* Text areas */
            QTextEdit {
                background-color: #FFFFFF;
                border: 1px solid #808080;
                border-radius: 2px;
                font-family: Consolas;
                font-size: 9pt;
                color: #000000;
            }
            
            QTextEdit#StatsBox {
                background-color: #F8F8F8;
            }
            
            /* Labels */
            QLabel#TaskTitle {
                color: #000080;
                background-color: transparent;
                padding: 2px;
            }
            
            QLabel#StatusLabel {
                color: #000000;
                background-color: transparent;
                font-style: italic;
            }
            
            QLabel#StatsTitle, QLabel#VisTitle {
                color: #000080;
                background-color: transparent;
            }
            
            /* Graphics View */
            QGraphicsView#ImageView {
                background-color: #F0F0F0;
                border: 1px solid #808080;
                border-radius: 2px;
            }
        """)

        self.enable_image_actions(False)

    def create_xp_button(self, text, callback):
        """Create Windows XP style main button"""
        btn = QPushButton(text)
        btn.setFont(QFont("Tahoma", 9))
        if callback:
            btn.clicked.connect(callback)
        return btn

    def create_sub_button(self, text, callback):
        """Create Windows XP style sub-panel button"""
        btn = QPushButton(text)
        btn.setObjectName("SubButton")
        btn.setFont(QFont("Tahoma", 8))
        if isinstance(callback, str):
            btn.clicked.connect(lambda: self.set_visualization_mode(callback))
        elif callback:
            btn.clicked.connect(callback)
        return btn

    def create_tool_button(self, text, callback):
        """Create Windows XP style toolbar button"""
        btn = QPushButton(text)
        btn.setObjectName("ToolButton")
        btn.setFont(QFont("Tahoma", 9))
        btn.clicked.connect(callback)
        return btn

    # ---------- UI helpers ----------

    def set_status(self, text):
        self.status_label.setText(f" {text}")

    def toggle_visualization_panel(self):
        self.visualization_panel.setVisible(self.btn_visualization.isChecked())

    def toggle_ops_panel(self):
        self.ops_panel.setVisible(self.btn_ops.isChecked())

    def toggle_filters_panel(self):
        self.filters_panel.setVisible(self.btn_filters.isChecked())

    def set_visualization_mode(self, mode):
        self.visualization_mode = mode
        self.btn_visualization.setChecked(False)
        self.visualization_panel.setVisible(False)
        self.update_analysis()
        mode_names = {
            "hsi": "HSI Components",
            "histogram": "Histogram View",
            "rgb": "RGB Channels",
            "combined": "Combined View"
        }
        self.set_status(f"View changed to: {mode_names[mode]}")

    def enable_image_actions(self, enabled):
        self.btn_reset.setEnabled(enabled)
        self.btn_export.setEnabled(enabled)
        self.btn_visualization.setEnabled(enabled)
        self.btn_ops.setEnabled(enabled)
        self.btn_filters.setEnabled(enabled)
        
        self.btn_visualization.setChecked(False)
        self.btn_ops.setChecked(False)
        self.btn_filters.setChecked(False)
        
        self.visualization_panel.setVisible(False)
        self.ops_panel.setVisible(False)
        self.filters_panel.setVisible(False)

        self.btn_fit.setEnabled(enabled)
        self.btn_reset_zoom.setEnabled(enabled)

    # ---------- File operations ----------

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.critical(self, "Error", "Failed to load image.")
            return

        self.original_bgr = img.copy()
        self.working_bgr = img.copy()

        self.enable_image_actions(True)
        self.set_status("Image loaded successfully")

        self.update_display()
        self.fit_to_window()
        self.update_analysis()

    def reset_to_original(self):
        if self.original_bgr is None:
            return
        self.working_bgr = self.original_bgr.copy()
        self.set_status("Image restored to original")
        self.update_display()
        self.fit_to_window()
        self.update_analysis()

    def clear_all(self):
        reply = QMessageBox.question(self, 'Clear Workspace', 
                                   'Are you sure you want to clear the workspace?',
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.original_bgr = None
            self.working_bgr = None

            self.pix_item.setPixmap(QPixmap())
            self.scene.setSceneRect(0, 0, 1, 1)

            self.stats_box.clear()
            self.canvas.ax.clear()
            self.canvas.draw()

            self.enable_image_actions(False)
            self.set_status("Workspace cleared")

    # ---------- Display & analysis ----------

    def update_display(self):
        if self.working_bgr is None:
            return
        pix = bgr_to_qpixmap(self.working_bgr)
        self.pix_item.setPixmap(pix)
        self.scene.setSceneRect(pix.rect())

    def fit_to_window(self):
        if self.pix_item.pixmap().isNull():
            return
        self.image_view.fitInView(self.pix_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.set_status("Image fitted to window")

    def reset_zoom(self):
        self.image_view.resetTransform()
        self.set_status("Zoom reset to 1:1")

    def export_image(self):
        if self.working_bgr is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image As",
            "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif *.tiff)"
        )
        if not path:
            return

        ok = cv2.imwrite(path, self.working_bgr)
        if not ok:
            QMessageBox.critical(self, "Error", "Failed to save image.")
        else:
            self.set_status(f"Image saved to: {path}")

    def update_analysis(self):
        if self.working_bgr is None:
            return

        rgb = cv2.cvtColor(self.working_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        R = rgb[:, :, 0]
        G = rgb[:, :, 1]
        B = rgb[:, :, 2]

        H, S, I = compute_hsi(rgb)

        text = "RGB (0 to 1)\n\n"
        text += channel_stats(R, "R")
        text += channel_stats(G, "G")
        text += channel_stats(B, "B")
        text += "\nHSI\n\n"
        text += channel_stats(H, "H (deg)")
        text += channel_stats(S, "S")
        text += channel_stats(I, "I")

        self.stats_box.setPlainText(text)
        self.plot_visualizations(R, G, B, H, S, I)

    def plot_visualizations(self, R, G, B, H, S, I):
        ax = self.canvas.ax
        ax.clear()

        if self.visualization_mode == "combined":
            ax.hist((R * 255).ravel(), bins=256, alpha=0.35, label="R", range=(0, 255), color='red')
            ax.hist((G * 255).ravel(), bins=256, alpha=0.35, label="G", range=(0, 255), color='green')
            ax.hist((B * 255).ravel(), bins=256, alpha=0.35, label="B", range=(0, 255), color='blue')
            ax.set_title("RGB Histograms", fontsize=10, color='#000080')
            ax.set_xlabel("Value (0-255)", fontsize=9)
            
        elif self.visualization_mode == "hsi":
            ax.hist(H.ravel(), bins=180, alpha=0.7, label="Hue", range=(0, 360), color='purple')
            ax.hist(S.ravel(), bins=100, alpha=0.7, label="Saturation", range=(0, 1), color='orange')
            ax.hist(I.ravel(), bins=100, alpha=0.7, label="Intensity", range=(0, 1), color='cyan')
            ax.set_title("HSI Components", fontsize=10, color='#000080')
            ax.set_xlabel("Value", fontsize=9)
            
        elif self.visualization_mode == "histogram":
            ax.hist((R * 255).ravel(), bins=256, alpha=0.6, label="Red", range=(0, 255), color='red', histtype='stepfilled')
            ax.hist((G * 255).ravel(), bins=256, alpha=0.6, label="Green", range=(0, 255), color='green', histtype='stepfilled')
            ax.hist((B * 255).ravel(), bins=256, alpha=0.6, label="Blue", range=(0, 255), color='blue', histtype='stepfilled')
            ax.hist(I.ravel() * 255, bins=256, alpha=0.3, label="Intensity", range=(0, 255), color='gray', histtype='step')
            ax.set_title("Enhanced Histograms", fontsize=10, color='#000080')
            ax.set_xlabel("Value (0-255)", fontsize=9)
            
        elif self.visualization_mode == "rgb":
            colors = ['red', 'green', 'blue']
            channels = [R*255, G*255, B*255]
            labels = ['Red Channel', 'Green Channel', 'Blue Channel']
            
            for i, (channel, color, label) in enumerate(zip(channels, colors, labels)):
                ax.hist(channel.ravel(), bins=256, alpha=0.7, label=label, 
                       range=(0, 255), color=color, histtype='stepfilled' if i==0 else 'step')
            ax.set_title("RGB Channel Distributions", fontsize=10, color='#000080')
            ax.set_xlabel("Value (0-255)", fontsize=9)

        ax.set_ylabel("Pixel Count", fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#F8F8F8')
        self.canvas.fig.tight_layout()
        self.canvas.draw()

    # ---------- Image operations ----------

    def apply_smoothing(self):
        if self.working_bgr is None:
            return
        self.working_bgr = cv2.GaussianBlur(self.working_bgr, (7, 7), 0)
        self.set_status("Applied smoothing (Gaussian blur)")
        self.update_display()
        self.update_analysis()

    def apply_sharpening(self):
        if self.working_bgr is None:
            return
        blur = cv2.GaussianBlur(self.working_bgr, (0, 0), 1.5)
        a = 1.2
        sharp = cv2.addWeighted(self.working_bgr, 1.0 + a, blur, -a, 0)
        self.working_bgr = np.clip(sharp, 0, 255).astype(np.uint8)
        self.set_status("Applied sharpening (unsharp mask)")
        self.update_display()
        self.update_analysis()

    def apply_hist_equalization(self):
        if self.working_bgr is None:
            return
        ycrcb = cv2.cvtColor(self.working_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = cv2.equalizeHist(y)
        ycrcb_eq = cv2.merge([y_eq, cr, cb])
        self.working_bgr = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
        self.set_status("Applied histogram equalization (luminance)")
        self.update_display()
        self.update_analysis()

    # ---------- Frequency Domain Filters ----------

    def create_filter(self, shape, filter_type, cutoff=30, order=2):
        """Create frequency domain filter"""
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
        
        if filter_type == "lowpass":
            filter_mask = np.zeros((rows, cols))
            filter_mask[distance <= cutoff] = 1
            
        elif filter_type == "highpass":
            filter_mask = np.ones((rows, cols))
            filter_mask[distance <= cutoff] = 0
            
        elif filter_type == "gaussian":
            filter_mask = np.exp(-(distance**2) / (2 * (cutoff**2)))
            
        elif filter_type == "notchpass":
            filter_mask = np.ones((rows, cols))
            notch_points = [(crow + 30, ccol + 30), (crow - 30, ccol - 30),
                           (crow + 30, ccol - 30), (crow - 30, ccol + 30)]
            for point in notch_points:
                y, x = point
                ry, rx = np.ogrid[-y:rows-y, -x:cols-x]
                mask = rx**2 + ry**2 <= cutoff**2
                filter_mask[mask] = 0
                
        elif filter_type == "notchreject":
            filter_mask = np.zeros((rows, cols))
            notch_points = [(crow + 30, ccol + 30), (crow - 30, ccol - 30),
                           (crow + 30, ccol - 30), (crow - 30, ccol + 30)]
            for point in notch_points:
                y, x = point
                ry, rx = np.ogrid[-y:rows-y, -x:cols-x]
                mask = rx**2 + ry**2 <= cutoff**2
                filter_mask[mask] = 1
        
        return filter_mask

    def apply_filter(self, filter_type):
        if self.working_bgr is None:
            return
            
        filtered_channels = []
        for i in range(3):
            channel = self.working_bgr[:, :, i].astype(np.float32)
            
            f = np.fft.fft2(channel)
            fshift = np.fft.fftshift(f)
            
            rows, cols = channel.shape
            filter_mask = self.create_filter((rows, cols), filter_type, cutoff=30)
            
            fshift_filtered = fshift * filter_mask
            
            f_ishift = np.fft.ifftshift(fshift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            
            filtered_channels.append(img_back)
        
        self.working_bgr = np.stack(filtered_channels, axis=2).astype(np.uint8)
        
        filter_names = {
            "lowpass": "Low Pass",
            "highpass": "High Pass", 
            "gaussian": "Gaussian",
            "notchpass": "Notch Pass",
            "notchreject": "Notch Reject"
        }
        
        self.set_status(f"Applied {filter_names[filter_type]} Filter")
        self.update_display()
        self.update_analysis()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Tahoma", 9))
    win = ImageAnalyzer()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()