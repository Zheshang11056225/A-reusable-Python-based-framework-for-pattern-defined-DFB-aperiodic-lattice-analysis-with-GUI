# A-reusable-Python-based-framework-for-pattern-defined-DFB-aperiodic-lattice-analysis-with-GUI

Introduction
This repository contains the Python code developed for a third-year individual project on the analysis of pattern-defined distributed feedback (DFB) and aperiodic lattice structures.

Context
The project focuses on coded lattice representation, transfer matrix method (TMM) modelling, gain-dependent reflectivity mapping, phase-matched pump-frequency estimation, and a GUI-based framework for comparing uniform, single-defect, and aperiodic gratings.

Requirements

Python 3.x
NumPy
Matplotlib
Tkinter

How to run
Run the main Python file in a Python environment with the required libraries installed. The GUI will open and allow the user to select built-in patterns or import custom TXT patterns.

Main features

built-in pattern selection
custom TXT pattern import
pattern-template generation
grating parameter adjustment
wavelength / normalised-frequency display
figure generation and saving

Known limitations
The nonlinear-efficiency-related quantity used in this implementation is a project-specific proxy rather than a complete first-principles nonlinear observable.

Future improvements
Possible future work includes testing more lattice patterns, improving validation against original published data, and extending the nonlinear modelling.

Performance note

This program is strongly affected by multi-core CPU performance. A CPU equivalent to Intel i7-9700 or AMD Ryzen 7 2700, or stronger, is recommended. The software uses multiprocessing for the numerical simulation, while two logical CPU threads are reserved for GUI and operating-system responsiveness. Therefore, the runtime can vary significantly depending on the available CPU resources.
