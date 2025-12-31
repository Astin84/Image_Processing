
# Image Analyzer

A modern desktop image analysis and processing application built with **Python**, **PySide6**, and **OpenCV**.
It provides interactive image viewing, RGB & HSI color analysis, histogram visualization, and common image enhancement operations.

---

## ✨ Features

### 🖼 Image Viewer

* Import images (`PNG`, `JPG`, `BMP`, `TIFF`)
* Smooth **mouse-wheel zoom**
* **Click & drag** panning
* Fit image to window
* Reset zoom

### 🎨 Color Analysis

* RGB channel statistics:

  * Min, Max, Mean, Standard Deviation
* HSI color space analysis:

  * Hue (degrees)
  * Saturation
  * Intensity
* Combined **RGB + HSI histograms**

### 🛠 Image Operations

Accessible from an expandable **Operations** panel:

* **Smoothing** (Gaussian Blur)
* **Sharpening** (Unsharp Mask)
* **Histogram Equalization** (luminance-based)

### 💾 Export

* Export processed images to:

  * PNG
  * JPG / JPEG
  * BMP
  * TIFF

### 🧩 UI & Workflow

* Professional dark theme
* Sidebar-based layout
* Non-destructive editing (original image preserved)
* Live histogram and statistics updates

---

## 📦 Requirements

* Python **3.9+**
* Windows / Linux / macOS

### Dependencies

```bash
pip install pyside6 opencv-python numpy matplotlib
```

---

## 🚀 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/image-analyzer.git
cd image-analyzer
```

2. Install dependencies:

```bash
python -m pip install pyside6 opencv-python numpy matplotlib
```

3. Run the application:

```bash
python image_analyzer.py
```

---

## 🧭 Usage

1. Click **Import Image** to load an image.
2. Use the mouse wheel to zoom and drag to pan.
3. Open **Operations** to apply image processing filters.
4. Inspect RGB & HSI statistics and histograms.
5. Export the processed image using **Export Image**.

---

## 🧠 Technical Notes

* **HSI Color Space**

  * Intensity = average of RGB channels
  * Saturation based on minimum RGB value
  * Hue computed using arccosine with quadrant correction

* **Histogram Equalization**

  * Applied only to the **luminance channel (Y)** in YCrCb color space
  * Preserves original color balance

---

## 📁 Project Structure

```
image-analyzer/
│
├── image_analyzer.py    # Main application
├── README.md            # Documentation
├── requirements.txt     # Dependencies (optional)
└── screenshots/         # UI screenshots (optional)
```

---

## 🔮 Planned Enhancements

* Undo / Redo support
* Slider-based filter controls
* Additional color spaces (HSV, LAB, YCbCr)
* Webcam input
* Batch processing
* Standalone EXE packaging (PyInstaller)

---

## 📜 License

Released under the **MIT License**.
Free to use, modify, and distribute.

---

## 👤 Author

**Mohammad Yasin Firoozi**
GitHub: [@firooziyasin90](https://github.com/firooziyasin90)

---

If you want, I can also:

* Generate a `requirements.txt`
* Add GitHub badges
* Write a CONTRIBUTING.md
* Prepare screenshots for the README
* Create a release-ready Windows EXE

Just tell me.
