# Earth data science labs

These labs are part of the course Earth data science taught at the Institut de physique du globe de Paris. The labs are taught by [Antoine Lucas](http://dralucas.geophysx.org/), [Alexandre Fournier](https://www.ipgp.fr/~fournier/), [Éléonore Stutzmann](https://www.ipgp.fr/~stutz/) and [Léonard Seydoux](https://sites.google.com/view/leonard-seydoux/accueil).

## Organization

We have five slots of 4 hours to teach the labs. The labs are organized as follows:

__0. Self-evaluation__ (~2 hours). This lab will allow you to evaluate your level in Python and scientific computing. It is not graded and you can skip it if you feel confident in your Python skills. Note that it is largely recommended to complete this lab in order to be able to follow the rest of the course. 

__1. River sensor calibration__ (~4 hours). This lab allow to perform a first simple machine learning task: the calibration of a river sensor. The goal is to predict the suspended sediment concentration from the turbidity of the water. 

__2. Earthquake location__ (~4 hours). In this lab, we will use Bayesian inference to locate the earthquake that occurred near the city of Le Teil in November 2019. 

__3. Lidar data classification__ (~8 hours). In this lab, we will classify lidar cloud points into different classes using supervised machine learning tools. 

__4. Deep learning__ (~2 hours). In this lab, we will explore several deep learning architectures to perform a regression task.

Note that the labs will be uploaded progressively during the course in the GitHub repository.

## Solutions

The lab solutions will be uploaded progressively during the course in the GitHub repository. Note that there are many solutions to the labs and that the ones provided here are not necessarily the best ones. Also note that the instructors may propose different solutions during the labs. The main idea of these sessions is for you to be overly curious and to try to find the solutions that best fit your needs, and your understanding of the problem. Some of you may complete the tasks at a faster pace than others, and we encourage you to help your peers during the labs, and also to explore further aspects of the problems that are not covered in the labs.

## Python environment

The easiest way to run most notebooks of this course is to create a new Anaconda environment with the following set of commands. We decided not to go with an environment file to allow for more flexibility in Python versions.

The following lines create a new environment called `earth-data-science` without any package installed. Then, we install the most constrained packages first (namely, `obspy`) which will install the latest compatible version of `python`, `numpy` and `scipy`. Finally, we install the rest of the packages.

```bash
conda create -n earth-data-science
conda activate earth-data-science
conda install -c conda-forge obspy
conda install -c conda-forge numpy scipy matplotlib pandas jupyter scikit-learn cartopy ipywidgets rasterio 
pip install tqdm 
pip install laspy
```

Once this is done, you must select the kernel `earth-data-science` in Jupyter to run the notebooks. Please inform your instructor if you have any problem with this.


## Running the notebooks

The notebooks can be either ran locally or on a remote server. The remote server is available at the following address: https://charline.ipgp.fr. You can log in with your IPGP credentials. Therein, you can apply clone to download the notebooks from this repository (e.g. `git clone https://github.com/leonard-seydoux/earth-data-science.git`). 

