# Hist-unmix

Hist-unmix is an open-sourced package, in python language (ipynb-file), to separate susceptibility components of distorted hysteresis curves through a phenomenological model. The Hist-unmix package allows the user to adjust a forward model of up to three ferromagnetic components and a dia/paramagnetic contribution. Optimization of all of the parameters is achieved through least squares fit (Levenberg-Marquardt method) providing an uncertainty for the inverted parameters through a Monte Carlo error propagation. For each ferromagnetic component, it is possible to calculate magnetization saturation (Ms), magnetization saturation of remanence (Mrs) and the mean coercivity (Bc). 

Here is a quick tutorial to install the required libraries to run Hist-unmix.

	1. Firstly, download the zip file containing the Hist-unmix.ipyfile and the Hist-unmix_functions.py file.

	2. Put these files together in a folder of your preference.

	3. Hist-unmix is, essentially, an ipynb-file. So Jupyter Notebook is mandatory! Go ahead to https://www.anaconda.com/ and download and execute
	the ANACONDA graphical installer (depending on your operational system).

	4. Open the Anaconda prompt and type (sequentially, press ENTER):
		pip install numpy matplotlib scipy pandas ipywidgets seaborn plotly

	5. Close the Anaconda prompt, reopen it and type:
		jupyter notebook
	
	6. After launching Jupyter Notebook, search for your folder and double click on the Hist-unmix.ipyfile to open it. 

	7. The environment is set to run your models!

PS: Always make sure to have your csv-data file within the same folder where Hist-unmix is.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bellon-donardelli/Hist-unmix.git/master?urlpath=Code%2FHist-unmix.ipynb)
