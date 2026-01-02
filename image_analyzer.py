# -*- coding: utf-8 -*-

import sys
import numpy as np
import cv2

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QLabel,
    QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox, QTextEdit,
    QMessageBox, QSizePolicy, QFrame,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QComboBox, QToolBar, QStatusBar, QMenuBar, QMenu, QProgressBar
)
from PySide6.QtGui import QPixmap, QImage, QWheelEvent, QFont, QIcon, QAction, QPalette, QColor
from PySide6.QtCore import Qt, QTimer

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
    def __init__(self, theme="windows_xp"):
        self.fig = Figure(figsize=(7, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.current_theme = theme
        self.apply_theme()
        super().__init__(self.fig)
        
    def apply_theme(self):
        if self.current_theme == "windows_xp":
            self.fig.patch.set_facecolor('#ECE9D8')
            self.ax.set_facecolor('#ECE9D8')
            self.ax.title.set_color('#000080')
        elif self.current_theme == "modern":
            self.fig.patch.set_facecolor('#F5F5F5')
            self.ax.set_facecolor('#FFFFFF')
            self.ax.title.set_color('#2C3E50')
        elif self.current_theme == "light":
            self.fig.patch.set_facecolor('#FFFFFF')
            self.ax.set_facecolor('#F9F9F9')
            self.ax.title.set_color('#000000')
        elif self.current_theme == "dark":
            self.fig.patch.set_facecolor('#2D2D2D')
            self.ax.set_facecolor('#1E1E1E')
            self.ax.title.set_color('#FFFFFF')
            self.ax.tick_params(colors='#CCCCCC')
            self.ax.xaxis.label.set_color('#CCCCCC')
            self.ax.yaxis.label.set_color('#CCCCCC')
            
        if self.current_theme == "dark":
            for spine in self.ax.spines.values():
                spine.set_color('#666666')


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
        
        self.current_theme = "windows_xp"
        self.original_bgr = None
        self.working_bgr = None
        self.visualization_mode = "combined"
        
        self.setup_ui()
        self.apply_theme(self.current_theme)

    def setup_ui(self):
        self.setWindowTitle("Image Analyzer Pro")
        self.resize(1400, 850)
        
        # Create central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
        
        # Main content area
        content_widget = QWidget()
        content_widget.setObjectName("ContentWidget")
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(10)
        
        # Left sidebar
        self.create_sidebar(content_layout)
        
        # Right content area
        self.create_content_area(content_layout)
        
        main_layout.addWidget(content_widget)
        
        # Status bar
        self.create_status_bar()
        
        self.enable_image_actions(False)

    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open Image...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.load_image)
        file_menu.addAction(open_action)
        
        save_action = QAction("&Save Image As...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.export_image)
        save_action.setEnabled(False)
        self.save_action = save_action
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        reset_action = QAction("&Reset to Original", self)
        reset_action.setShortcut("Ctrl+R")
        reset_action.triggered.connect(self.reset_to_original)
        reset_action.setEnabled(False)
        self.reset_action = reset_action
        file_menu.addAction(reset_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        self.theme_menu = view_menu.addMenu("&Theme")
        
        xp_theme = QAction("Windows &XP", self)
        xp_theme.triggered.connect(lambda: self.apply_theme("windows_xp"))
        self.theme_menu.addAction(xp_theme)
        
        modern_theme = QAction("&Modern", self)
        modern_theme.triggered.connect(lambda: self.apply_theme("modern"))
        self.theme_menu.addAction(modern_theme)
        
        light_theme = QAction("&Light", self)
        light_theme.triggered.connect(lambda: self.apply_theme("light"))
        self.theme_menu.addAction(light_theme)
        
        dark_theme = QAction("&Dark", self)
        dark_theme.triggered.connect(lambda: self.apply_theme("dark"))
        self.theme_menu.addAction(dark_theme)
        
        view_menu.addSeparator()
        
        fit_action = QAction("&Fit to Window", self)
        fit_action.setShortcut("F")
        fit_action.triggered.connect(self.fit_to_window)
        fit_action.setEnabled(False)
        self.fit_action = fit_action
        view_menu.addAction(fit_action)
        
        reset_zoom_action = QAction("Reset &Zoom", self)
        reset_zoom_action.setShortcut("Ctrl+0")
        reset_zoom_action.triggered.connect(self.reset_zoom)
        reset_zoom_action.setEnabled(False)
        self.reset_zoom_action = reset_zoom_action
        view_menu.addAction(reset_zoom_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        smooth_action = QAction("&Smooth Image", self)
        smooth_action.triggered.connect(self.apply_smoothing)
        smooth_action.setEnabled(False)
        self.smooth_action = smooth_action
        tools_menu.addAction(smooth_action)
        
        sharpen_action = QAction("S&harpen Image", self)
        sharpen_action.triggered.connect(self.apply_sharpening)
        sharpen_action.setEnabled(False)
        self.sharpen_action = sharpen_action
        tools_menu.addAction(sharpen_action)
        
        histeq_action = QAction("&Equalize Histogram", self)
        histeq_action.triggered.connect(self.apply_hist_equalization)
        histeq_action.setEnabled(False)
        self.histeq_action = histeq_action
        tools_menu.addAction(histeq_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Theme selector
        toolbar.addWidget(QLabel(" Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Windows XP", "Modern", "Light", "Dark"])
        self.theme_combo.currentTextChanged.connect(self.change_theme_from_combo)
        toolbar.addWidget(self.theme_combo)
        
        toolbar.addSeparator()
        
        # Quick access buttons
        self.btn_open = self.create_toolbar_button("📁 Open", self.load_image)
        self.btn_save = self.create_toolbar_button("💾 Save", self.export_image)
        self.btn_save.setEnabled(False)
        self.btn_reset = self.create_toolbar_button("↩️ Reset", self.reset_to_original)
        self.btn_reset.setEnabled(False)
        
        toolbar.addWidget(self.btn_open)
        toolbar.addWidget(self.btn_save)
        toolbar.addWidget(self.btn_reset)
        
        toolbar.addSeparator()
        
        self.btn_fit = self.create_toolbar_button("🔍 Fit", self.fit_to_window)
        self.btn_fit.setEnabled(False)
        self.btn_zoom_reset = self.create_toolbar_button("🗺️ 1:1", self.reset_zoom)
        self.btn_zoom_reset.setEnabled(False)
        
        toolbar.addWidget(self.btn_fit)
        toolbar.addWidget(self.btn_zoom_reset)

    def create_toolbar_button(self, text, callback):
        btn = QPushButton(text)
        btn.setMinimumHeight(28)
        btn.clicked.connect(callback)
        return btn

    def create_sidebar(self, parent_layout):
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setMinimumWidth(280)
        sidebar.setMaximumWidth(320)
        
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(12, 12, 12, 12)
        sidebar_layout.setSpacing(15)
        
        # File operations section
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(8)
        file_layout.setContentsMargins(12, 15, 12, 12)
        
        self.btn_import = self.create_action_button("📂 Open Image", self.load_image)
        self.btn_export = self.create_action_button("💾 Save Image", self.export_image)
        self.btn_export.setEnabled(False)
        self.btn_clear = self.create_action_button("🗑️ Clear All", self.clear_all)
        
        file_layout.addWidget(self.btn_import)
        file_layout.addWidget(self.btn_export)
        file_layout.addWidget(self.btn_clear)
        
        sidebar_layout.addWidget(file_group)
        
        # Image processing section
        process_group = QGroupBox("Image Processing")
        process_layout = QVBoxLayout(process_group)
        process_layout.setSpacing(8)
        process_layout.setContentsMargins(12, 15, 12, 12)
        
        self.btn_smooth = self.create_action_button("🔹 Smooth", self.apply_smoothing)
        self.btn_sharpen = self.create_action_button("🔸 Sharpen", self.apply_sharpening)
        self.btn_histeq = self.create_action_button("📊 Equalize", self.apply_hist_equalization)
        
        process_layout.addWidget(self.btn_smooth)
        process_layout.addWidget(self.btn_sharpen)
        process_layout.addWidget(self.btn_histeq)
        
        sidebar_layout.addWidget(process_group)
        
        # Filters section
        filter_group = QGroupBox("Frequency Filters")
        filter_layout = QVBoxLayout(filter_group)
        filter_layout.setSpacing(8)
        filter_layout.setContentsMargins(12, 15, 12, 12)
        
        self.btn_lowpass = self.create_action_button("🔽 Low Pass", lambda: self.apply_filter("lowpass"))
        self.btn_highpass = self.create_action_button("🔼 High Pass", lambda: self.apply_filter("highpass"))
        self.btn_gaussian = self.create_action_button("⛰️ Gaussian", lambda: self.apply_filter("gaussian"))
        
        filter_layout.addWidget(self.btn_lowpass)
        filter_layout.addWidget(self.btn_highpass)
        filter_layout.addWidget(self.btn_gaussian)
        
        sidebar_layout.addWidget(filter_group)
        
        # View options section
        view_group = QGroupBox("View Options")
        view_layout = QVBoxLayout(view_group)
        view_layout.setSpacing(8)
        view_layout.setContentsMargins(12, 15, 12, 12)
        
        self.btn_view_hsi = self.create_action_button("🎨 HSI View", lambda: self.set_visualization_mode("hsi"))
        self.btn_view_rgb = self.create_action_button("🌈 RGB View", lambda: self.set_visualization_mode("rgb"))
        self.btn_view_hist = self.create_action_button("📈 Histogram", lambda: self.set_visualization_mode("histogram"))
        self.btn_view_combined = self.create_action_button("🔗 Combined", lambda: self.set_visualization_mode("combined"))
        
        view_layout.addWidget(self.btn_view_hsi)
        view_layout.addWidget(self.btn_view_rgb)
        view_layout.addWidget(self.btn_view_hist)
        view_layout.addWidget(self.btn_view_combined)
        
        sidebar_layout.addWidget(view_group)
        
        sidebar_layout.addStretch()
        
        # Progress indicator
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        sidebar_layout.addWidget(self.progress_bar)
        
        parent_layout.addWidget(sidebar)

    def create_action_button(self, text, callback):
        btn = QPushButton(text)
        btn.setMinimumHeight(36)
        btn.clicked.connect(callback)
        return btn

    def create_content_area(self, parent_layout):
        content_area = QWidget()
        content_layout = QVBoxLayout(content_area)
        content_layout.setSpacing(10)
        
        # Image display area
        image_frame = QFrame()
        image_frame.setObjectName("ImageFrame")
        image_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        image_layout = QVBoxLayout(image_frame)
        image_layout.setContentsMargins(4, 4, 4, 4)
        
        self.scene = QGraphicsScene(self)
        self.image_view = ZoomableGraphicsView(self)
        self.image_view.setObjectName("ImageView")
        self.image_view.setScene(self.scene)
        self.pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pix_item)
        
        # Image info bar
        info_bar = QFrame()
        info_layout = QHBoxLayout(info_bar)
        info_layout.setContentsMargins(8, 4, 8, 4)
        
        self.image_info = QLabel("No image loaded")
        info_layout.addWidget(self.image_info)
        info_layout.addStretch()
        
        image_layout.addWidget(self.image_view)
        image_layout.addWidget(info_bar)
        
        content_layout.addWidget(image_frame, 3)
        
        # Bottom panels
        bottom_panel = QHBoxLayout()
        bottom_panel.setSpacing(10)
        
        # Statistics panel
        stats_frame = QFrame()
        stats_frame.setObjectName("StatsFrame")
        stats_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        stats_layout = QVBoxLayout(stats_frame)
        stats_layout.setContentsMargins(10, 10, 10, 10)
        
        stats_title = QLabel("Image Statistics")
        stats_title.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(stats_title)
        
        self.stats_box = QTextEdit()
        self.stats_box.setReadOnly(True)
        stats_layout.addWidget(self.stats_box)
        
        bottom_panel.addWidget(stats_frame, 2)
        
        # Visualization panel
        vis_frame = QFrame()
        vis_frame.setObjectName("VisFrame")
        vis_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        vis_layout = QVBoxLayout(vis_frame)
        vis_layout.setContentsMargins(10, 10, 10, 10)
        
        vis_title = QLabel("Visualization")
        vis_title.setAlignment(Qt.AlignCenter)
        vis_layout.addWidget(vis_title)
        
        self.canvas = MplCanvas(self.current_theme)
        vis_layout.addWidget(self.canvas)
        
        bottom_panel.addWidget(vis_frame, 3)
        
        content_layout.addLayout(bottom_panel, 2)
        
        parent_layout.addWidget(content_area, 1)

    def create_status_bar(self):
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        
        self.status_label = QLabel("Ready")
        status_bar.addWidget(self.status_label, 1)
        
        self.size_label = QLabel("")
        status_bar.addPermanentWidget(self.size_label)

    def change_theme_from_combo(self, theme_name):
        theme_map = {
            "Windows XP": "windows_xp",
            "Modern": "modern",
            "Light": "light",
            "Dark": "dark"
        }
        self.apply_theme(theme_map.get(theme_name, "windows_xp"))

    def apply_theme(self, theme):
        self.current_theme = theme
        self.theme_combo.blockSignals(True)
        
        # Update combo box selection
        if theme == "windows_xp":
            self.theme_combo.setCurrentIndex(0)
        elif theme == "modern":
            self.theme_combo.setCurrentIndex(1)
        elif theme == "light":
            self.theme_combo.setCurrentIndex(2)
        elif theme == "dark":
            self.theme_combo.setCurrentIndex(3)
            
        self.theme_combo.blockSignals(False)
        
        # Apply stylesheet
        self.setStyleSheet(self.get_stylesheet(theme))
        
        # Update matplotlib canvas theme
        if hasattr(self, 'canvas'):
            self.canvas.current_theme = theme
            self.canvas.apply_theme()
            if self.working_bgr is not None:
                self.update_analysis()
        
        # Update application palette for certain themes
        if theme == "dark":
            dark_palette = QPalette()
            dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.WindowText, Qt.white)
            dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
            dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
            dark_palette.setColor(QPalette.ToolTipText, Qt.white)
            dark_palette.setColor(QPalette.Text, Qt.white)
            dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ButtonText, Qt.white)
            dark_palette.setColor(QPalette.BrightText, Qt.red)
            dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.HighlightedText, Qt.black)
            QApplication.setPalette(dark_palette)
        else:
            QApplication.setPalette(QApplication.style().standardPalette())
        
        self.status_label.setText(f"Theme changed to {theme.replace('_', ' ').title()}")

    def get_stylesheet(self, theme):
        if theme == "windows_xp":
            return """
                QMainWindow {
                    background-color: #D4D0C8;
                }
                
                QFrame#Sidebar {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #ECE9D8, stop:1 #D4D0C8);
                    border: 1px solid #808080;
                    border-radius: 6px;
                }
                
                QFrame#ContentWidget {
                    background-color: #FFFFFF;
                }
                
                QFrame#ImageFrame, QFrame#StatsFrame, QFrame#VisFrame {
                    background-color: #FFFFFF;
                    border-radius: 4px;
                }
                
                QGroupBox {
                    font-weight: bold;
                    font-size: 11pt;
                    color: #000080;
                    border: 2px solid #808080;
                    border-radius: 6px;
                    margin-top: 12px;
                    padding-top: 12px;
                    background-color: #ECE9D8;
                }
                
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 12px;
                    padding: 0 8px 0 8px;
                }
                
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #F9F9F9, stop:0.1 #E8E8E8, stop:0.5 #E0E0E0, stop:0.9 #D8D8D8, stop:1 #C8C8C8);
                    border: 1px solid #808080;
                    border-radius: 4px;
                    padding: 8px 16px;
                    font-family: Tahoma;
                    font-size: 10pt;
                    color: #000000;
                    min-height: 36px;
                }
                
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #FFFFFF, stop:0.1 #F0F0F0, stop:0.5 #E8E8E8, stop:0.9 #E0E0E0, stop:1 #D0D0D0);
                    border: 1px solid #000080;
                }
                
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #C8C8C8, stop:0.1 #D8D8D8, stop:0.5 #E0E0E0, stop:0.9 #E8E8E8, stop:1 #F9F9F9);
                }
                
                QPushButton:disabled {
                    background-color: #E0E0E0;
                    color: #808080;
                }
                
                QTextEdit {
                    background-color: #F8F8F8;
                    border: 1px solid #808080;
                    border-radius: 3px;
                    font-family: Consolas;
                    font-size: 10pt;
                }
                
                QLabel {
                    font-family: Tahoma;
                }
                
                QStatusBar {
                    background-color: #ECE9D8;
                    border-top: 1px solid #808080;
                }
                
                QComboBox {
                    border: 1px solid #808080;
                    border-radius: 3px;
                    padding: 4px;
                    background-color: white;
                    min-width: 120px;
                }
                
                QComboBox:hover {
                    border: 1px solid #000080;
                }
                
                QToolBar {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #ECE9D8, stop:1 #D4D0C8);
                    border-bottom: 1px solid #808080;
                    spacing: 6px;
                    padding: 4px;
                }
                
                QMenuBar {
                    background-color: #ECE9D8;
                    border-bottom: 1px solid #808080;
                }
                
                QGraphicsView#ImageView {
                    background-color: #F0F0F0;
                    border: 1px solid #808080;
                    border-radius: 3px;
                }
            """
            
        elif theme == "modern":
            return """
                QMainWindow {
                    background-color: #F5F5F5;
                }
                
                QFrame#Sidebar {
                    background-color: #FFFFFF;
                    border: 1px solid #E0E0E0;
                    border-radius: 8px;
                }
                
                QFrame#ContentWidget {
                    background-color: #FFFFFF;
                }
                
                QFrame#ImageFrame, QFrame#StatsFrame, QFrame#VisFrame {
                    background-color: #FFFFFF;
                    border: 1px solid #E0E0E0;
                    border-radius: 8px;
                }
                
                QGroupBox {
                    font-weight: bold;
                    font-size: 11pt;
                    color: #2C3E50;
                    border: 2px solid #3498DB;
                    border-radius: 8px;
                    margin-top: 12px;
                    padding-top: 12px;
                    background-color: #F8F9FA;
                }
                
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 12px;
                    padding: 0 8px 0 8px;
                    color: #2980B9;
                }
                
                QPushButton {
                    background-color: #3498DB;
                    border: none;
                    border-radius: 6px;
                    padding: 10px 18px;
                    font-family: Segoe UI;
                    font-size: 10pt;
                    font-weight: 500;
                    color: white;
                    min-height: 40px;
                }
                
                QPushButton:hover {
                    background-color: #2980B9;
                }
                
                QPushButton:pressed {
                    background-color: #21618C;
                }
                
                QPushButton:disabled {
                    background-color: #BDC3C7;
                    color: #7F8C8D;
                }
                
                QTextEdit {
                    background-color: #FFFFFF;
                    border: 1px solid #E0E0E0;
                    border-radius: 6px;
                    font-family: Consolas;
                    font-size: 10pt;
                    padding: 8px;
                }
                
                QLabel {
                    font-family: Segoe UI;
                }
                
                QStatusBar {
                    background-color: #FFFFFF;
                    border-top: 1px solid #E0E0E0;
                    color: #7F8C8D;
                }
                
                QComboBox {
                    border: 2px solid #3498DB;
                    border-radius: 6px;
                    padding: 6px;
                    background-color: white;
                    min-width: 120px;
                    font-family: Segoe UI;
                }
                
                QComboBox:hover {
                    border-color: #2980B9;
                }
                
                QToolBar {
                    background-color: #FFFFFF;
                    border-bottom: 1px solid #E0E0E0;
                    spacing: 8px;
                    padding: 6px;
                }
                
                QMenuBar {
                    background-color: #FFFFFF;
                    border-bottom: 1px solid #E0E0E0;
                }
                
                QGraphicsView#ImageView {
                    background-color: #F8F9FA;
                    border: 1px solid #E0E0E0;
                    border-radius: 6px;
                }
            """
            
        elif theme == "light":
            return """
                QMainWindow {
                    background-color: #FFFFFF;
                }
                
                QFrame#Sidebar {
                    background-color: #FAFAFA;
                    border-right: 1px solid #E0E0E0;
                }
                
                QFrame#ContentWidget {
                    background-color: #FFFFFF;
                }
                
                QFrame#ImageFrame, QFrame#StatsFrame, QFrame#VisFrame {
                    background-color: #FFFFFF;
                    border: 1px solid #E0E0E0;
                    border-radius: 4px;
                }
                
                QGroupBox {
                    font-weight: 600;
                    font-size: 11pt;
                    color: #424242;
                    border: 1px solid #BDBDBD;
                    border-radius: 6px;
                    margin-top: 12px;
                    padding-top: 12px;
                    background-color: #FFFFFF;
                }
                
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 12px;
                    padding: 0 8px 0 8px;
                    color: #616161;
                }
                
                QPushButton {
                    background-color: #FFFFFF;
                    border: 1px solid #E0E0E0;
                    border-radius: 4px;
                    padding: 8px 16px;
                    font-family: Arial;
                    font-size: 10pt;
                    color: #424242;
                    min-height: 36px;
                }
                
                QPushButton:hover {
                    background-color: #F5F5F5;
                    border-color: #BDBDBD;
                }
                
                QPushButton:pressed {
                    background-color: #EEEEEE;
                }
                
                QPushButton:disabled {
                    background-color: #FAFAFA;
                    color: #BDBDBD;
                    border-color: #EEEEEE;
                }
                
                QTextEdit {
                    background-color: #FFFFFF;
                    border: 1px solid #E0E0E0;
                    border-radius: 4px;
                    font-family: Consolas;
                    font-size: 10pt;
                    padding: 8px;
                }
                
                QLabel {
                    font-family: Arial;
                }
                
                QStatusBar {
                    background-color: #FAFAFA;
                    border-top: 1px solid #E0E0E0;
                    color: #757575;
                }
                
                QComboBox {
                    border: 1px solid #E0E0E0;
                    border-radius: 4px;
                    padding: 6px;
                    background-color: white;
                    min-width: 120px;
                    font-family: Arial;
                }
                
                QComboBox:hover {
                    border-color: #BDBDBD;
                }
                
                QToolBar {
                    background-color: #FFFFFF;
                    border-bottom: 1px solid #E0E0E0;
                    spacing: 6px;
                    padding: 4px;
                }
                
                QMenuBar {
                    background-color: #FFFFFF;
                    border-bottom: 1px solid #E0E0E0;
                }
                
                QGraphicsView#ImageView {
                    background-color: #FAFAFA;
                    border: 1px solid #E0E0E0;
                    border-radius: 4px;
                }
            """
            
        elif theme == "dark":
            return """
                QMainWindow {
                    background-color: #2D2D2D;
                }
                
                QFrame#Sidebar {
                    background-color: #3C3C3C;
                    border-right: 1px solid #555555;
                }
                
                QFrame#ContentWidget {
                    background-color: #2D2D2D;
                }
                
                QFrame#ImageFrame, QFrame#StatsFrame, QFrame#VisFrame {
                    background-color: #3C3C3C;
                    border: 1px solid #555555;
                    border-radius: 6px;
                }
                
                QGroupBox {
                    font-weight: 600;
                    font-size: 11pt;
                    color: #FFFFFF;
                    border: 1px solid #555555;
                    border-radius: 6px;
                    margin-top: 12px;
                    padding-top: 12px;
                    background-color: #3C3C3C;
                }
                
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 12px;
                    padding: 0 8px 0 8px;
                    color: #BB86FC;
                }
                
                QPushButton {
                    background-color: #555555;
                    border: 1px solid #666666;
                    border-radius: 6px;
                    padding: 10px 18px;
                    font-family: Segoe UI;
                    font-size: 10pt;
                    color: #FFFFFF;
                    min-height: 40px;
                }
                
                QPushButton:hover {
                    background-color: #666666;
                    border-color: #777777;
                }
                
                QPushButton:pressed {
                    background-color: #444444;
                }
                
                QPushButton:disabled {
                    background-color: #3C3C3C;
                    color: #777777;
                    border-color: #555555;
                }
                
                QTextEdit {
                    background-color: #2D2D2D;
                    border: 1px solid #555555;
                    border-radius: 6px;
                    font-family: Consolas;
                    font-size: 10pt;
                    padding: 8px;
                    color: #FFFFFF;
                }
                
                QLabel {
                    font-family: Segoe UI;
                    color: #FFFFFF;
                }
                
                QStatusBar {
                    background-color: #3C3C3C;
                    border-top: 1px solid #555555;
                    color: #CCCCCC;
                }
                
                QComboBox {
                    border: 1px solid #555555;
                    border-radius: 6px;
                    padding: 6px;
                    background-color: #3C3C3C;
                    min-width: 120px;
                    font-family: Segoe UI;
                    color: #FFFFFF;
                }
                
                QComboBox:hover {
                    border-color: #666666;
                }
                
                QToolBar {
                    background-color: #3C3C3C;
                    border-bottom: 1px solid #555555;
                    spacing: 8px;
                    padding: 6px;
                }
                
                QMenuBar {
                    background-color: #3C3C3C;
                    border-bottom: 1px solid #555555;
                    color: #FFFFFF;
                }
                
                QMenuBar::item:selected {
                    background-color: #555555;
                }
                
                QGraphicsView#ImageView {
                    background-color: #2D2D2D;
                    border: 1px solid #555555;
                    border-radius: 6px;
                }
                
                QMenu {
                    background-color: #3C3C3C;
                    color: #FFFFFF;
                    border: 1px solid #555555;
                }
                
                QMenu::item:selected {
                    background-color: #555555;
                }
            """
            
        return ""

    # ---------- UI helpers ----------

    def show_loading(self, show=True, message="Processing..."):
        self.progress_bar.setVisible(show)
        if show:
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            self.status_label.setText(message)

    def set_status(self, text):
        self.status_label.setText(text)
        self.statusBar().showMessage(text, 3000)

    def set_visualization_mode(self, mode):
        self.visualization_mode = mode
        if self.working_bgr is not None:
            self.update_analysis()
        
        mode_names = {
            "hsi": "HSI Components",
            "histogram": "Histogram View",
            "rgb": "RGB Channels",
            "combined": "Combined View"
        }
        self.set_status(f"View changed to: {mode_names[mode]}")

    def enable_image_actions(self, enabled):
        self.btn_export.setEnabled(enabled)
        self.btn_reset.setEnabled(enabled)
        self.btn_save.setEnabled(enabled)
        self.btn_smooth.setEnabled(enabled)
        self.btn_sharpen.setEnabled(enabled)
        self.btn_histeq.setEnabled(enabled)
        self.btn_lowpass.setEnabled(enabled)
        self.btn_highpass.setEnabled(enabled)
        self.btn_gaussian.setEnabled(enabled)
        
        self.btn_fit.setEnabled(enabled)
        self.btn_zoom_reset.setEnabled(enabled)
        
        # Menu actions
        self.save_action.setEnabled(enabled)
        self.reset_action.setEnabled(enabled)
        self.fit_action.setEnabled(enabled)
        self.reset_zoom_action.setEnabled(enabled)
        self.smooth_action.setEnabled(enabled)
        self.sharpen_action.setEnabled(enabled)
        self.histeq_action.setEnabled(enabled)

    # ---------- File operations ----------

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*.*)"
        )
        if not path:
            return

        self.show_loading(True, "Loading image...")
        QApplication.processEvents()
        
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.critical(self, "Error", "Failed to load image.")
            self.show_loading(False)
            return

        self.original_bgr = img.copy()
        self.working_bgr = img.copy()

        self.enable_image_actions(True)
        self.set_status(f"Image loaded: {path}")
        self.image_info.setText(f"Image: {img.shape[1]} × {img.shape[0]} pixels, {img.shape[2]} channels")
        
        self.update_display()
        self.fit_to_window()
        self.update_analysis()
        
        self.show_loading(False)

    def reset_to_original(self):
        if self.original_bgr is None:
            return
        self.working_bgr = self.original_bgr.copy()
        self.set_status("Image restored to original")
        self.update_display()
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
            
            self.image_info.setText("No image loaded")
            self.size_label.setText("")

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

        self.show_loading(True, "Saving image...")
        QApplication.processEvents()
        
        ok = cv2.imwrite(path, self.working_bgr)
        if not ok:
            QMessageBox.critical(self, "Error", "Failed to save image.")
        else:
            self.set_status(f"Image saved: {path}")
        
        self.show_loading(False)

    def update_analysis(self):
        if self.working_bgr is None:
            return

        rgb = cv2.cvtColor(self.working_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        R = rgb[:, :, 0]
        G = rgb[:, :, 1]
        B = rgb[:, :, 2]

        H, S, I = compute_hsi(rgb)

        text = "RGB Statistics (0 to 1)\n"
        text += "─" * 30 + "\n"
        text += channel_stats(R, "Red Channel")
        text += channel_stats(G, "Green Channel")
        text += channel_stats(B, "Blue Channel")
        text += "\nHSI Statistics\n"
        text += "─" * 30 + "\n"
        text += channel_stats(H, "Hue (degrees)")
        text += channel_stats(S, "Saturation")
        text += channel_stats(I, "Intensity")
        
        # Add image info
        text += "\nImage Information\n"
        text += "─" * 30 + "\n"
        h, w, c = self.working_bgr.shape
        text += f"Dimensions: {w} × {h} pixels\n"
        text += f"Channels: {c}\n"
        text += f"Total Pixels: {h * w:,}\n"
        text += f"Memory: {(h * w * c) / 1024:.1f} KB"

        self.stats_box.setPlainText(text)
        
        # Update size label in status bar
        self.size_label.setText(f"Size: {w}×{h}")
        
        self.plot_visualizations(R, G, B, H, S, I)

    def plot_visualizations(self, R, G, B, H, S, I):
        ax = self.canvas.ax
        ax.clear()

        if self.visualization_mode == "combined":
            ax.hist((R * 255).ravel(), bins=256, alpha=0.35, label="R", range=(0, 255), color='red')
            ax.hist((G * 255).ravel(), bins=256, alpha=0.35, label="G", range=(0, 255), color='green')
            ax.hist((B * 255).ravel(), bins=256, alpha=0.35, label="B", range=(0, 255), color='blue')
            ax.set_title("RGB Histograms", fontsize=11, fontweight='bold')
            ax.set_xlabel("Pixel Value (0-255)", fontsize=10)
            
        elif self.visualization_mode == "hsi":
            ax.hist(H.ravel(), bins=180, alpha=0.7, label="Hue", range=(0, 360), color='purple')
            ax.hist(S.ravel(), bins=100, alpha=0.7, label="Saturation", range=(0, 1), color='orange')
            ax.hist(I.ravel(), bins=100, alpha=0.7, label="Intensity", range=(0, 1), color='cyan')
            ax.set_title("HSI Components", fontsize=11, fontweight='bold')
            ax.set_xlabel("Value", fontsize=10)
            
        elif self.visualization_mode == "histogram":
            ax.hist((R * 255).ravel(), bins=256, alpha=0.6, label="Red", range=(0, 255), color='red', histtype='stepfilled')
            ax.hist((G * 255).ravel(), bins=256, alpha=0.6, label="Green", range=(0, 255), color='green', histtype='stepfilled')
            ax.hist((B * 255).ravel(), bins=256, alpha=0.6, label="Blue", range=(0, 255), color='blue', histtype='stepfilled')
            ax.hist(I.ravel() * 255, bins=256, alpha=0.3, label="Intensity", range=(0, 255), color='gray', histtype='step')
            ax.set_title("Enhanced Histograms", fontsize=11, fontweight='bold')
            ax.set_xlabel("Pixel Value (0-255)", fontsize=10)
            
        elif self.visualization_mode == "rgb":
            colors = ['red', 'green', 'blue']
            channels = [R*255, G*255, B*255]
            labels = ['Red Channel', 'Green Channel', 'Blue Channel']
            
            for i, (channel, color, label) in enumerate(zip(channels, colors, labels)):
                ax.hist(channel.ravel(), bins=256, alpha=0.7, label=label, 
                       range=(0, 255), color=color, histtype='stepfilled' if i==0 else 'step')
            ax.set_title("RGB Channel Distributions", fontsize=11, fontweight='bold')
            ax.set_xlabel("Pixel Value (0-255)", fontsize=10)

        ax.set_ylabel("Pixel Count", fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Apply theme-specific grid colors
        if self.current_theme == "dark":
            ax.grid(True, alpha=0.2, linestyle='--', color='#666666')
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()

    # ---------- Image operations ----------

    def apply_smoothing(self):
        if self.working_bgr is None:
            return
        
        self.show_loading(True, "Applying smoothing filter...")
        QTimer.singleShot(100, self._do_smoothing)  # Non-blocking

    def _do_smoothing(self):
        try:
            self.working_bgr = cv2.GaussianBlur(self.working_bgr, (7, 7), 0)
            self.set_status("Applied smoothing (Gaussian blur)")
            self.update_display()
            self.update_analysis()
        finally:
            self.show_loading(False)

    def apply_sharpening(self):
        if self.working_bgr is None:
            return
        
        self.show_loading(True, "Applying sharpening filter...")
        QTimer.singleShot(100, self._do_sharpening)

    def _do_sharpening(self):
        try:
            blur = cv2.GaussianBlur(self.working_bgr, (0, 0), 1.5)
            a = 1.2
            sharp = cv2.addWeighted(self.working_bgr, 1.0 + a, blur, -a, 0)
            self.working_bgr = np.clip(sharp, 0, 255).astype(np.uint8)
            self.set_status("Applied sharpening (unsharp mask)")
            self.update_display()
            self.update_analysis()
        finally:
            self.show_loading(False)

    def apply_hist_equalization(self):
        if self.working_bgr is None:
            return
        
        self.show_loading(True, "Applying histogram equalization...")
        QTimer.singleShot(100, self._do_hist_equalization)

    def _do_hist_equalization(self):
        try:
            ycrcb = cv2.cvtColor(self.working_bgr, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            y_eq = cv2.equalizeHist(y)
            ycrcb_eq = cv2.merge([y_eq, cr, cb])
            self.working_bgr = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
            self.set_status("Applied histogram equalization (luminance)")
            self.update_display()
            self.update_analysis()
        finally:
            self.show_loading(False)

    # ---------- Frequency Domain Filters ----------

    def create_filter(self, shape, filter_type, cutoff=30, order=2):
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
            
        return filter_mask

    def apply_filter(self, filter_type):
        if self.working_bgr is None:
            return
        
        self.show_loading(True, f"Applying {filter_type.replace('_', ' ')} filter...")
        QTimer.singleShot(100, lambda: self._do_apply_filter(filter_type))

    def _do_apply_filter(self, filter_type):
        try:
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
                "gaussian": "Gaussian"
            }
            
            self.set_status(f"Applied {filter_names[filter_type]} Filter")
            self.update_display()
            self.update_analysis()
        finally:
            self.show_loading(False)

    def show_about(self):
        about_text = """
        <h3>Image Analyzer Pro</h3>
        <p>Version 2.0</p>
        <p>A comprehensive image analysis and processing tool with multiple themes.</p>
        <p>Features:</p>
        <ul>
            <li>Multiple theme support (Windows XP, Modern, Light, Dark)</li>
            <li>Image statistics and analysis</li>
            <li>Color space transformations (RGB, HSI)</li>
            <li>Image filtering and enhancement</li>
            <li>Frequency domain filters</li>
            <li>Interactive visualization</li>
        </ul>
        <p>© 2024 Image Analyzer Pro</p>
        """
        QMessageBox.about(self, "About Image Analyzer Pro", about_text)


def main():
    app = QApplication(sys.argv)
    
    # Set default font
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    
    win = ImageAnalyzer()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()