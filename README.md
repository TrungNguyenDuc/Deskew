# Oblique Plane Microscopy (OPM) Image Processing & Analysis Suite

** Oblique Plane Microscopy (OPM) Image Processing & Analysis Suite** is a comprehensive Python-based application designed for efficient processing and quantitative analysis of data from Light-Sheet Fluorescence Microscopes (LSFMs), particularly those employing scanned oblique plane illumination techniques like Open-Top Light-Sheet (OTLS) microscopy.

This suite provides a user-friendly Graphical User Interface (GUI) and a powerful backend that leverages GPU acceleration to streamline your workflow, enabling rapid on-the-fly feedback crucial for system alignment, optimization, and routine data processing.
![image](https://github.com/user-attachments/assets/00b1e0bc-3edb-4035-a0ad-527be8ca5978)


## The Challenge Addressed

Aligning and optimizing custom LSFM systems, especially when testing different objectives or optical configurations, requires immediate quantitative feedback on image quality. Standard microscope control software often lacks integrated, real-time deskewing and precise resolution/SNR analysis capabilities, leading to time-consuming offline processing. **Oblique Plane Microscopy (OPM) Image Processing & Analysis Suite** fills this gap by providing these essential tools directly at your fingertips.

## Core Modules & Key Features

**Oblique Plane Microscopy (OPM) Image Processing & Analysis Suite** is built around three primary analysis modules:

### 1. Deskewing Module
*   **Geometric Correction:** Accurately corrects skewed raw LSFM data to produce geometrically correct volumetric representations.
*   **High-Fidelity Options:**
    *   **Interpolated Affine Shear:** Provides superior image quality by using sub-pixel interpolation during the shearing process, minimizing artifacts.
    *   Fast Integer-Shift Shear: Available for quicker processing if highest fidelity is not paramount.
*   **GPU Acceleration:** Utilizes NVIDIA CUDA (via CuPy) for significant speed-ups in shearing, 3D Z-scaling (zoom), and rotation operations.
*   **Intelligent Chunking:** Enables processing of large datasets that exceed GPU memory by breaking them into manageable chunks.
*   **CPU Fallback:** Ensures robust processing even if GPU resources are unavailable.
*   **Optional Post-Shear Smoothing:** GPU-accelerated Gaussian filtering can be applied to the final deskewed volume to mitigate potential minor artifacts from the shearing process.
*   **Configurable Parameters:** Allows user input for scan angle, pixel sizes (dx, dz), shear direction, etc.

### 2. Decorrelation Analysis Module
*   **Quantitative Resolution & SNR:** Objectively measures image resolution and Signal-to-Noise Ratio using Fourier Ring Correlation (FRC) / Decorrelation Analysis principles.
*   **Flexible Analysis Modes:**
    *   **MIP-Based:** Analyzes Maximum Intensity Projections (XY, XZ, YZ) for a rapid overview of system performance across different orientations.
    *   **Sampled Slice-wise:** Provides a more detailed volumetric assessment by analyzing a user-defined number of equally spaced 2D slices along the X, Y, and Z stack orientations, reporting min/max/average metrics.
*   **CPU-Based Core Algorithm:** Ensures stability and reliability of the decorrelation metrics.

### 3. PSF Fitting Module
*   **Resolution from Bead Images:** Analyzes images of sub-resolution beads (PSFs) to determine experimental resolution.
*   **Gaussian Fitting:** Fits 1D Gaussian profiles to line profiles taken through identified beads along X, Y, and Z axes.
*   **FWHM Calculation:** Calculates Full-Width at Half-Maximum (FWHM) from the Gaussian fits to report resolution in nm.
*   **GPU-Accelerated Data Preparation:** Initial steps like image padding, thresholding, and bead candidate identification (labeling, ROI extraction) can be GPU-accelerated.
*   **CPU-Based Curve Fitting:** The core `scipy.optimize.curve_fit` remains on the CPU for robust iterative optimization.
*   **Filtering & Reporting:** Filters bead fits based on R² quality and reports mean/std FWHM values and number of good beads.
*   **Diagnostic Plots:** Generates plots of FWHM vs. position and thresholded bead images.

### General Features
*   **User-Friendly GUI (Tkinter):**
    *   Intuitive interface with dedicated tabs for Deskew, Decorrelation, and PSF Fitting.
    *   Easy parameter input and process control.
    *   Real-time logging window for monitoring progress and diagnostics.
    *   Options to enable/disable specific steps (e.g., saving intermediates, smoothing, skipping analyses).
*   **Flexible Outputs:**
    *   Processed images saved as ImageJ/Fiji-compatible TIFF files.
    *   Detailed reports (`deskew_note.txt`, logged summaries for decorrelation and PSF).
    *   Publication-quality plots for MIPs and analysis results.
*   **Memory Management:** Optimized for handling large datasets on both CPU and GPU.

## Installation


**Prerequisites:**
*   Python (version 3.12 or higher)
*   NVIDIA GPU with CUDA Toolkit (version 12.9) installed (for GPU acceleration)
*   CuPy (matching your CUDA version)
*   NumPy, SciPy, Tifffile, Matplotlib, Scikit-image, Pillow

**Steps:**
1.  **Clone the repository:**
   2.  **Create a Conda Environment (Recommended):**
    ```bash
    conda create -n [your_env_name] python=3.9 # Or your preferred Python version
    conda activate [your_env_name]
    ```
3.  **Install Dependencies:**
    *   **CuPy:** Follow official CuPy installation instructions for your CUDA version (e.g., `pip install cupy-cuda11x`).
    *   Install other packages:
        ```bash
        pip install numpy scipy tifffile matplotlib scikit-image Pillow
        ```
## Usage

1.  **Standalone Application:**
    *   Run the main Python script: `python main_gui.py` (or whatever your entry point is).
    *   Use the GUI to load data, set parameters for Deskew, Decorrelation, or PSF Fitting, and run the analyses.



### Example Workflow (during system alignment):
1.  Acquire a test image/stack (e.g., beads or a structured sample) using your microscope control software (like "Navigate").
2.  Open **Oblique Plane Microscopy (OPM) Image Processing & Analysis Suite**.
3.  Go to the "Deskew" tab, load the raw skewed TIFF, input microscope parameters, and run deskewing.
![image](https://github.com/user-attachments/assets/0f2d461b-60f8-4fed-bdd0-1b742528f913)

5.  Go to the "PSF Fitting" tab (if using beads) or "Decorrelation" tab.
    *   The deskewed file path might be auto-populated or you may need to select it.
    *   Input relevant pixel sizes.
    *   Run the analysis.
![image](https://github.com/user-attachments/assets/461de6ab-e9ea-4689-b315-8577f65b0428)
![image](https://github.com/user-attachments/assets/fb976809-e40f-4ac3-9bb2-a2ddc5fda9d8)

6.  Review the reported resolution and SNR values in the log or summary plots.
7.  Make physical adjustments to your microscope setup.
8.  Repeat steps 1-5 until optimal performance is achieved.

