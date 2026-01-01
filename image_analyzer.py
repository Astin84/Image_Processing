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
from PySide6.QtGui import QPixmap, QImage, QWheelEvent
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
        self.fig = Figure(figsize=(7, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
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

        self.setWindowTitle("Advanced Image Analyzer")
        self.resize(1400, 800)

        self.original_bgr = None
        self.working_bgr = None
        self.visualization_mode = "combined"  # combined, hsi, histogram, rgb

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(14)

        # Sidebar
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setSpacing(10)
        sidebar_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.addWidget(sidebar, 0)

        self.btn_import = QPushButton("📁 Import Image")
        self.btn_import.clicked.connect(self.load_image)
        sidebar_layout.addWidget(self.btn_import)

        self.btn_reset = QPushButton("🔄 Reset to Original")
        self.btn_reset.clicked.connect(self.reset_to_original)
        self.btn_reset.setEnabled(False)
        sidebar_layout.addWidget(self.btn_reset)

        self.btn_clear = QPushButton("🗑️ Clear")
        self.btn_clear.clicked.connect(self.clear_all)
        sidebar_layout.addWidget(self.btn_clear)

        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setFrameShadow(QFrame.Sunken)
        sidebar_layout.addWidget(line1)

        # Visualization modes
        self.btn_visualization = QPushButton("📊 Visualization Modes")
        self.btn_visualization.setCheckable(True)
        self.btn_visualization.clicked.connect(self.toggle_visualization_panel)
        self.btn_visualization.setEnabled(False)
        sidebar_layout.addWidget(self.btn_visualization)

        self.visualization_panel = QFrame()
        self.visualization_panel.setObjectName("VisPanel")
        vis_layout = QVBoxLayout(self.visualization_panel)
        vis_layout.setContentsMargins(0, 0, 0, 0)
        vis_layout.setSpacing(8)

        self.btn_show_hsi = QPushButton("🎨 HSI Components")
        self.btn_show_hsi.clicked.connect(lambda: self.set_visualization_mode("hsi"))
        vis_layout.addWidget(self.btn_show_hsi)

        self.btn_show_histogram = QPushButton("📈 Histogram")
        self.btn_show_histogram.clicked.connect(lambda: self.set_visualization_mode("histogram"))
        vis_layout.addWidget(self.btn_show_histogram)

        self.btn_show_rgb = QPushButton("🌈 RGB Channels")
        self.btn_show_rgb.clicked.connect(lambda: self.set_visualization_mode("rgb"))
        vis_layout.addWidget(self.btn_show_rgb)

        self.btn_show_combined = QPushButton("🔗 Combined View")
        self.btn_show_combined.clicked.connect(lambda: self.set_visualization_mode("combined"))
        vis_layout.addWidget(self.btn_show_combined)

        self.visualization_panel.setVisible(False)
        sidebar_layout.addWidget(self.visualization_panel)

        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        sidebar_layout.addWidget(line2)

        # Operations
        self.btn_ops = QPushButton("⚙️ Operations")
        self.btn_ops.setCheckable(True)
        self.btn_ops.clicked.connect(self.toggle_ops_panel)
        self.btn_ops.setEnabled(False)
        sidebar_layout.addWidget(self.btn_ops)

        self.ops_panel = QFrame()
        self.ops_panel.setObjectName("OpsPanel")
        ops_layout = QVBoxLayout(self.ops_panel)
        ops_layout.setContentsMargins(0, 0, 0, 0)
        ops_layout.setSpacing(8)

        self.btn_smooth = QPushButton("🔹 Smoothing")
        self.btn_smooth.clicked.connect(self.apply_smoothing)
        ops_layout.addWidget(self.btn_smooth)

        self.btn_sharp = QPushButton("🔸 Sharpening")
        self.btn_sharp.clicked.connect(self.apply_sharpening)
        ops_layout.addWidget(self.btn_sharp)

        self.btn_histeq = QPushButton("📊 Histogram Equalization")
        self.btn_histeq.clicked.connect(self.apply_hist_equalization)
        ops_layout.addWidget(self.btn_histeq)

        self.ops_panel.setVisible(False)
        sidebar_layout.addWidget(self.ops_panel)

        # Advanced Filters
        self.btn_filters = QPushButton("🎛️ Frequency Domain Filters")
        self.btn_filters.setCheckable(True)
        self.btn_filters.clicked.connect(self.toggle_filters_panel)
        self.btn_filters.setEnabled(False)
        sidebar_layout.addWidget(self.btn_filters)

        self.filters_panel = QFrame()
        self.filters_panel.setObjectName("FiltersPanel")
        filters_layout = QVBoxLayout(self.filters_panel)
        filters_layout.setContentsMargins(0, 0, 0, 0)
        filters_layout.setSpacing(8)

        self.btn_lowpass = QPushButton("🔽 Low Pass Filter")
        self.btn_lowpass.clicked.connect(lambda: self.apply_filter("lowpass"))
        filters_layout.addWidget(self.btn_lowpass)

        self.btn_highpass = QPushButton("🔼 High Pass Filter")
        self.btn_highpass.clicked.connect(lambda: self.apply_filter("highpass"))
        filters_layout.addWidget(self.btn_highpass)

        self.btn_notchpass = QPushButton("🎯 Notch Pass Filter")
        self.btn_notchpass.clicked.connect(lambda: self.apply_filter("notchpass"))
        filters_layout.addWidget(self.btn_notchpass)

        self.btn_notchreject = QPushButton("🚫 Notch Reject Filter")
        self.btn_notchreject.clicked.connect(lambda: self.apply_filter("notchreject"))
        filters_layout.addWidget(self.btn_notchreject)

        self.btn_gaussian = QPushButton("⛰️ Gaussian Filter")
        self.btn_gaussian.clicked.connect(lambda: self.apply_filter("gaussian"))
        filters_layout.addWidget(self.btn_gaussian)

        self.filters_panel.setVisible(False)
        sidebar_layout.addWidget(self.filters_panel)

        self.status_label = QLabel("Status: No image loaded")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("StatusLabel")
        sidebar_layout.addWidget(self.status_label)
        sidebar_layout.addStretch(1)

        # Content
        content = QFrame()
        content.setObjectName("Content")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(12)
        main_layout.addWidget(content, 1)

        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        content_layout.addLayout(top_row, 3)

        # Preview group
        image_group = QGroupBox("Preview")
        image_layout = QVBoxLayout(image_group)

        self.scene = QGraphicsScene(self)
        self.image_view = ZoomableGraphicsView(self)
        self.image_view.setScene(self.scene)
        self.pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pix_item)

        image_layout.addWidget(self.image_view)

        zoom_btn_row = QHBoxLayout()

        self.btn_fit = QPushButton("🔍 Fit to Window")
        self.btn_fit.clicked.connect(self.fit_to_window)
        self.btn_fit.setEnabled(False)
        zoom_btn_row.addWidget(self.btn_fit)

        self.btn_reset_zoom = QPushButton("🗺️ Reset Zoom")
        self.btn_reset_zoom.clicked.connect(self.reset_zoom)
        self.btn_reset_zoom.setEnabled(False)
        zoom_btn_row.addWidget(self.btn_reset_zoom)

        self.btn_export = QPushButton("💾 Export Image")
        self.btn_export.clicked.connect(self.export_image)
        self.btn_export.setEnabled(False)
        zoom_btn_row.addWidget(self.btn_export)

        image_layout.addLayout(zoom_btn_row)
        top_row.addWidget(image_group, 3)

        # Stats group
        stats_group = QGroupBox("RGB and HSI Indexes")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_box = QTextEdit()
        self.stats_box.setReadOnly(True)
        self.stats_box.setObjectName("StatsBox")
        stats_layout.addWidget(self.stats_box)
        top_row.addWidget(stats_group, 2)

        # Histogram group
        hist_group = QGroupBox("Visualizations")
        hist_layout = QVBoxLayout(hist_group)
        self.canvas = MplCanvas()
        hist_layout.addWidget(self.canvas)
        content_layout.addWidget(hist_group, 2)

        # Style
        self.setStyleSheet("""
            QMainWindow { background: #af6a00; }
            QGroupBox {
                color: #e8eefc;
                border: 1px solid #26324a;
                border-radius: 12px;
                margin-top: 10px;
                padding: 10px;
                background: rgba(255,255,255,0.02);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                font-weight: 600;
            }

            #Sidebar {
                background: #0f1a30;
                border: 1px solid #26324a;
                border-radius: 14px;
                min-width: 280px;
                max-width: 300px;
            }

            #Content {
                background: rgba(255,255,255,0.01);
                border: 1px solid #26324a;
                border-radius: 14px;
            }

            QPushButton {
                background: #1f6feb;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 12px;
                font-weight: 600;
                text-align: left;
            }
            QPushButton:hover { background: #2b7bff; }
            QPushButton:pressed { background: #1a5bd1; }
            QPushButton:disabled {
                background: #2a3550;
                color: #9fb1d8;
            }

            #VisPanel QPushButton,
            #OpsPanel QPushButton,
            #FiltersPanel QPushButton {
                background: #162a52;
                border: 1px solid #26324a;
                padding-left: 14px;
                font-size: 12px;
            }
            #VisPanel QPushButton:hover { background: #1a3263; }
            #OpsPanel QPushButton:hover { background: #1a3263; }
            #FiltersPanel QPushButton:hover { background: #1a3263; }

            #VisPanel QPushButton {
                background: #2a1a52;
            }
            #FiltersPanel QPushButton {
                background: #521a2a;
            }

            QTextEdit {
                background: #0f1626;
                color: #e8eefc;
                border: 1px solid #26324a;
                border-radius: 10px;
                font-family: Consolas, monospace;
                font-size: 12px;
            }

            #StatusLabel {
                color: #cbd6f1;
                padding-top: 6px;
                font-size: 11px;
            }
        """)

        self.enable_image_actions(False)

    # ---------- UI helpers ----------

    def set_status(self, text):
        self.status_label.setText(f"Status: {text}")

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
        self.set_status(f"Visualization mode: {mode_names[mode]}")

    def enable_image_actions(self, enabled):
        self.btn_reset.setEnabled(enabled)
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
        self.btn_export.setEnabled(enabled)

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
        self.set_status("Image loaded")

        self.update_display()
        self.fit_to_window()
        self.update_analysis()

    def reset_to_original(self):
        if self.original_bgr is None:
            return
        self.working_bgr = self.original_bgr.copy()
        self.set_status("Reset to original")
        self.update_display()
        self.fit_to_window()
        self.update_analysis()

    def clear_all(self):
        self.original_bgr = None
        self.working_bgr = None

        self.pix_item.setPixmap(QPixmap())
        self.scene.setSceneRect(0, 0, 1, 1)

        self.stats_box.clear()
        self.canvas.ax.clear()
        self.canvas.draw()

        self.enable_image_actions(False)
        self.set_status("No image loaded")

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

    def reset_zoom(self):
        self.image_view.resetTransform()

    def export_image(self):
        if self.working_bgr is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Image",
            "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif *.tiff)"
        )
        if not path:
            return

        ok = cv2.imwrite(path, self.working_bgr)
        if not ok:
            QMessageBox.critical(self, "Error", "Failed to export image.")
        else:
            QMessageBox.information(self, "Export", "Image exported successfully.")

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
            # Original combined view
            ax.hist((R * 255).ravel(), bins=256, alpha=0.35, label="R", range=(0, 255), color='red')
            ax.hist((G * 255).ravel(), bins=256, alpha=0.35, label="G", range=(0, 255), color='green')
            ax.hist((B * 255).ravel(), bins=256, alpha=0.35, label="B", range=(0, 255), color='blue')
            ax.set_title("RGB Histograms")
            ax.set_xlabel("Value (0-255)")
            
        elif self.visualization_mode == "hsi":
            # HSI components
            ax.hist(H.ravel(), bins=180, alpha=0.7, label="Hue", range=(0, 360), color='purple')
            ax.hist(S.ravel(), bins=100, alpha=0.7, label="Saturation", range=(0, 1), color='orange')
            ax.hist(I.ravel(), bins=100, alpha=0.7, label="Intensity", range=(0, 1), color='cyan')
            ax.set_title("HSI Components")
            ax.set_xlabel("Value")
            
        elif self.visualization_mode == "histogram":
            # Enhanced histogram view
            ax.hist((R * 255).ravel(), bins=256, alpha=0.6, label="Red", range=(0, 255), color='red', histtype='stepfilled')
            ax.hist((G * 255).ravel(), bins=256, alpha=0.6, label="Green", range=(0, 255), color='green', histtype='stepfilled')
            ax.hist((B * 255).ravel(), bins=256, alpha=0.6, label="Blue", range=(0, 255), color='blue', histtype='stepfilled')
            ax.hist(I.ravel() * 255, bins=256, alpha=0.3, label="Intensity", range=(0, 255), color='gray', histtype='step')
            ax.set_title("Enhanced Histograms")
            ax.set_xlabel("Value (0-255)")
            
        elif self.visualization_mode == "rgb":
            # RGB channels separately
            colors = ['red', 'green', 'blue']
            channels = [R*255, G*255, B*255]
            labels = ['Red Channel', 'Green Channel', 'Blue Channel']
            
            for i, (channel, color, label) in enumerate(zip(channels, colors, labels)):
                ax.hist(channel.ravel(), bins=256, alpha=0.7, label=label, 
                       range=(0, 255), color=color, histtype='stepfilled' if i==0 else 'step')
            ax.set_title("RGB Channel Distributions")
            ax.set_xlabel("Value (0-255)")

        ax.set_ylabel("Pixel Count")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.2)
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
        
        # Create distance matrix
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
        
        if filter_type == "lowpass":
            # Ideal low-pass filter
            filter_mask = np.zeros((rows, cols))
            filter_mask[distance <= cutoff] = 1
            
        elif filter_type == "highpass":
            # Ideal high-pass filter
            filter_mask = np.ones((rows, cols))
            filter_mask[distance <= cutoff] = 0
            
        elif filter_type == "gaussian":
            # Gaussian filter
            filter_mask = np.exp(-(distance**2) / (2 * (cutoff**2)))
            
        elif filter_type == "notchpass":
            # Notch pass filter (removes specific frequencies)
            filter_mask = np.ones((rows, cols))
            # Create multiple notch points
            notch_points = [(crow + 30, ccol + 30), (crow - 30, ccol - 30),
                           (crow + 30, ccol - 30), (crow - 30, ccol + 30)]
            for point in notch_points:
                y, x = point
                ry, rx = np.ogrid[-y:rows-y, -x:cols-x]
                mask = rx**2 + ry**2 <= cutoff**2
                filter_mask[mask] = 0
                
        elif filter_type == "notchreject":
            # Notch reject filter (opposite of notch pass)
            filter_mask = np.zeros((rows, cols))
            # Create multiple notch points
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
            
        # Apply filter to each channel
        filtered_channels = []
        for i in range(3):
            channel = self.working_bgr[:, :, i].astype(np.float32)
            
            # Perform FFT
            f = np.fft.fft2(channel)
            fshift = np.fft.fftshift(f)
            
            # Create filter
            rows, cols = channel.shape
            filter_mask = self.create_filter((rows, cols), filter_type, cutoff=30)
            
            # Apply filter
            fshift_filtered = fshift * filter_mask
            
            # Inverse FFT
            f_ishift = np.fft.ifftshift(fshift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            
            filtered_channels.append(img_back)
        
        # Combine channels
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
    win = ImageAnalyzer()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()