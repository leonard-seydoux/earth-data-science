# Earth data science 

![](./lectures/images/titlepages/oldschool-earth-data-science.png)

## Description

The class Earth data science is a master-level class of the institut de physique du globe de Paris. This course is a legacy of the course of the same name by [Antoine Lucas](http://dralucas.geophysx.org/). The lectures are taught by [Léonard Seydoux](https://sites.google.com/view/leonard-seydoux/accueil) and the practicals by [Antoine Lucas](http://dralucas.geophysx.org/), [Alexandre Fournier](https://www.ipgp.fr/~fournier/), [Éléonore Stutzmann](https://www.ipgp.fr/~stutz/) and [Léonard Seydoux](https://sites.google.com/view/leonard-seydoux/accueil). 

The goal of this course is to introduce students to the basics of scientific computing and to the use of Python for solving geophysical problems. The course mostly consists in practical sessions where students will learn how to use Python to solve problems related to the Earth sciences mith statistical and machine learning methods. The course and notebooks rely on the Python [scikit-learn](https://scikit-learn.org/stable/) library, [pandas](https://pandas.pydata.org/), [pytorch](https://pytorch.org/), and the [deep learning](https://www.deeplearningbook.org/) book by Ian Goodfellow, Yoshua Bengio and Aaron Courville.

## Course content

The course contains a 8-hour lecture followed by 20 hours of practical sessions made with Jupyter notebooks. The lecture notes are available in the `lectures` folder and the practicals in the `labs` folder. You can find an introductory README file in each folder.

## Python environment

The easiest way to run most notebooks of this course is to create a new Anaconda environment with the following set of commands. We decided not to go with an environment file to allow for more flexibility in Python versions.

The following lines create a new environment called `earth-data-science` without any package installed. Then, we install the most constrained packages first (namely, `obspy`) which will install the latest compatible version of `python`, `numpy` and `scipy`. Finally, we install the rest of the packages.

```bash
conda create -n earth-data-science
conda activate earth-data-science
conda install -c conda-forge obspy
conda install -c conda-forge numpy scipy matplotlib pandas jupyter scikit-learn cartopy ipywidgets rasterio
```

Once this is done, you must select the kernel `earth-data-science` in Jupyter to run the notebooks. Please inform your instructor if you have any problem with this.