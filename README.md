# camera-path

Camera path planning for meteorological 3D visualization in Met3D, focused on large-scale weather phenomena — Tropical Cyclones and Atmospheric Rivers.

## Background

This project is based on an existing neural network pipeline developed as part of academic research. The pipeline downloads live ICON weather forecast data from the German Weather Service (DWD), processes it, and runs it through CGNet — a semantic segmentation network — to detect Tropical Cyclones and Atmospheric Rivers across global weather maps. Results are visualized in Met3D, a scientific tool designed for 3D meteorological data visualization.

The goal of this project is to extend that pipeline with automated camera path planning, so that Met3D can guide users through detected weather phenomena in both 2D and 3D.


## Dependencies

Python 3.10, PyTorch, xarray, numpy, scipy, scikit-image, cartopy, enstools, pygrib, cdo
