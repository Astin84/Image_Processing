# -*- coding: utf-8 -*-
import sys
import numpy as np
import cv2

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGroupBox, QTextEdit, QMessageBox, QSizePolicy
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """Convert OpenCV BGR image to QPixmap for display."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def compute_hsi_from_rgb(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute HSI from RGB (float in [0,1]).
    Returns:
      H in degrees [0, 360)
      S in [0,1]
      I in [0,1]
    """
    eps = 1e-8
    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]

    I = (R + G + B) / 3.0

    # Saturation: 1 - 3 * min(R,G,B) / (R+G+B)
    min_rgb = np.minimum(np.minimum(R, G), B)
    sum_rgb = R + G + B
    S = 1.0 - (3.0 * min_rgb / (sum_rgb + eps))
    S = np.clip(S, 0.0, 1.0)

    # Hue:
    # theta = arccos( 0.5*((R-G)+(R-B)) / sqrt((R-G)^2 + (R-B)*(G-B)) )
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + eps
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))  # [0, pi]

    H = np.degrees(theta)  # [0, 180]
    # If B > G then H = 360 - H
    H = np.where(B > G, 360.0 - H, H)
    H = np.mod(H, 360.0)

    return H, S, I


def channel_stats(arr: np.ndarray, name: str) -> str:
    """Compute min/max/mean/std for a channel and return formatted text."""
    a = arr.astype(np.float64)
    return (
        f"{name}:\n"
        f"  min:  {np.min(a):.4f}\n"
        f"  max:  {np.max(a):.4f}\n"
        f"  mean: {np.mean(a):.4f}\n"
        f"  std:  {np.std(a):.4f}\n"
    )


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(7, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


class ImageAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Analyzer — Histogram + RGB & HSI Indexes")
        self.resize(1200, 720)

        self.image_bgr: np.ndarray | None = None

        # Main layout
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)

        # Left panel (controls + stats)
        left_panel = QVBoxLayout()
        main_layout.addLayout(left_panel, 1)

        btn_row = QHBoxLayout()
        self.btn_open = QPushButton("Import Image")
        self.btn_open.clicked.connect(self.open_image)
        btn_row.addWidget(self.btn_open)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear_all)
        btn_row.addWidget(self.btn_clear)

        left_panel.addLayout(btn_row)

        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(300)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setStyleSheet(
            "QLabel { border: 1px solid #444; border-radius: 10px; background: #111; color: #bbb; }"
        )
        left_panel.addWidget(self.image_label, 4)

        stats_group = QGroupBox("RGB & HSI Indexes (Channel Stats)")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setStyleSheet("QTextEdit { font-family: Consolas, monospace; }")
        stats_layout.addWidget(self.stats_text)
        left_panel.addWidget(stats_group, 3)

        # Right panel (histograms)
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel, 1)

        hist_group = QGroupBox("Histograms")
        hist_layout = QVBoxLayout(hist_group)
        self.canvas = MplCanvas()
        hist_layout.addWidget(self.canvas)
        right_panel.addWidget(hist_group)

        # Modern dark-ish styling
        self.setStyleSheet("""
            QMainWindow { background: #0b0f14; }
            QGroupBox {
                color: #e6e6e6;
                border: 1px solid #2a2f3a;
                border-radius: 12px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QPushButton {
                background: #1f6feb;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 14px;
                font-weight: 600;
            }
            QPushButton:hover { background: #2b7bff; }
            QPushButton:pressed { background: #1a5bd1; }
            QTextEdit {
                background: #0f1620;
                color: #e6e6e6;
                border: 1px solid #2a2f3a;
                border-radius: 10px;
            }
        """)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select an Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"
        )
        if not file_path:
            return

        bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if bgr is None:
            QMessageBox.critical(self, "Error", "Could not load the selected image.")
            return

        self.image_bgr = bgr
        self.update_preview()
        self.update_analysis()

    def clear_all(self):
        self.image_bgr = None
        self.image_label.setText("No image loaded")
        self.stats_text.clear()
        self.canvas.ax.clear()
        self.canvas.draw()

    def update_preview(self):
        assert self.image_bgr is not None
        pix = bgr_to_qpixmap(self.image_bgr)

        # Fit to label size while preserving aspect ratio
        scaled = pix.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.image_bgr is not None:
            self.update_preview()

    def update_analysis(self):
        assert self.image_bgr is not None

        # Prepare RGB in [0,1]
        rgb_u8 = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        rgb = rgb_u8.astype(np.float32) / 255.0

        R = rgb[..., 0]
        G = rgb[..., 1]
        B = rgb[..., 2]

        H, S, I = compute_hsi_from_rgb(rgb)

        # Stats output
        text = []
        text.append("RGB (normalized 0..1)\n")
        text.append(channel_stats(R, "R"))
        text.append(channel_stats(G, "G"))
        text.append(channel_stats(B, "B"))

        text.append("\nHSI\n")
        text.append("H in degrees [0..360), S and I in [0..1]\n\n")
        text.append(channel_stats(H, "H"))
        text.append(channel_stats(S, "S"))
        text.append(channel_stats(I, "I"))

        self.stats_text.setPlainText("".join(text))

        # Histograms
        self.plot_histograms(rgb_u8, H, S, I)

    def plot_histograms(self, rgb_u8: np.ndarray, H: np.ndarray, S: np.ndarray, I: np.ndarray):
        ax = self.canvas.ax
        ax.clear()

        # RGB histograms (0..255)
        r = rgb_u8[..., 0].ravel()
        g = rgb_u8[..., 1].ravel()
        b = rgb_u8[..., 2].ravel()

        ax.hist(r, bins=256, alpha=0.35, label="R", range=(0, 255))
        ax.hist(g, bins=256, alpha=0.35, label="G", range=(0, 255))
        ax.hist(b, bins=256, alpha=0.35, label="B", range=(0, 255))

        # HSI histograms
        # H: 0..360
        ax.hist(H.ravel(), bins=180, alpha=0.25, label="H (deg)", range=(0, 360))
        # S, I: 0..1
        ax.hist(S.ravel(), bins=100, alpha=0.25, label="S", range=(0, 1))
        ax.hist(I.ravel(), bins=100, alpha=0.25, label="I", range=(0, 1))

        ax.set_title("Histograms: RGB and HSI")
        ax.set_xlabel("Value")
        ax.set_ylabel("Pixel Count")
        ax.legend()
        ax.grid(True, alpha=0.2)

        self.canvas.fig.tight_layout()
        self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    w = ImageAnalyzer()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
