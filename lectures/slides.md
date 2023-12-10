---
theme: presentation
marp: true
math: katex
---

<!-- _class: titlepage-->

# Earth Data Science

Lecture by Léonard Seydoux the master-level classes of the [institut de physique du globe de Paris](https://www.ipgp.fr) with contents inspired by the [scikit-learn](https://scikit-learn.org/stable/) Python library documentation and the [deep learning](https://www.deeplearningbook.org/) book of Ian Goodfellow.

`Léonard Seydoux` `Antoine Lucas` `́Eléonore Stutzmann` 
`Alexandre Fournier` `David Weissenbach.`

[<img src="images/logo/logo-github.svg" width=25 align="left" style="margin-top:10px; margin-right: -20px;"/> `leonard-seydoux/earth-data-science`](https://github.com/leonard-seydoux/earth-data-science)


---

<!-- paginate: true -->
<!-- _footer: Bergen et al. (2019) -->

## Goal: learn about statistical inference, machine, and deep learning

<div style="flex-basis: 50%;">

1. Identify problems in need for data science
1. Define the problem and analysis workflow
1. Train on real examples
1. Read AI-based papers with a critical eye

</div>
<div style="flex-basis: 40%;">

![drop-shadow width:350px](images/papers/bergen2019machine.png)

</div>

---

<!-- _footer: Mousavi et al. (2022) -->

## Goal: learn about statistical inference, machine, and deep learning

<div style="flex-basis: 50%;">

1. Identify problems in need for data science
1. Define the problem and analysis workflow
1. Train on real examples
1. Read AI-based papers with a critical eye

</div>
<div style="flex-basis: 43%;">

![width:500px](images/papers/mousavi2022papers.jpg)

</div>

---

<!-- _footer:  [www.scikit-learn.org](https://scikit-learn.org/stable/) -->

## Class organization

<div style="flex-basis: 40%;">

__Lectures (8h)__
Motivations, definitions, supervised, unsupervised, and deep learning.

__Notebooks (20h)__
Exercises and examples on various topics of Earth science.

__Final hackathon (4h)__
Solve a problem with machine learning.

</div>
<div style="flex-basis: 57%;">

![](https://scikit-learn.org/stable/_static/ml_map.png)

</div>

---

## Class learning curve

![](images/diagrams/class-learning-curve.png)

---

<!-- _class: titlepage-->

# 1. Introduction

__Machine learning__ for Earth science: why, what, and how? Are any of those methods useful for your research? How to read papers that use machine learning?

[<img src="images/logo/logo-github.svg" width=25 align="left" style="margin-top:10px; margin-right: -20px;"/> `leonard-seydoux/earth-data-science`](https://github.com/leonard-seydoux/earth-data-science)

---

## How much time do you need to describe the following images?

![width:1000px](images/papers/karpathy2015deep-nocap.png)

<!-- _footer: Karpathy & Fei-Fei (2015) -->

---

## How accurate are those descriptions?

![width:1000px](images/papers/karpathy2015deep.png)

<!-- _footer: Karpathy & Fei-Fei (2015) -->

---

## Ingredients for image captioning: hierarchical knowledge

![width:550px](images/deep-learning-book/figure-1-2.png)

<!-- _footer: Goodfellow et al. (2016) -->

---

## Can you spot the seismogram?

![width:500px](images/papers/valentine2012spot.png)

<!-- _footer: Valentine & Trampert (2012).<br>Top to bottom: FTSE; Temperature in Central England; Gaussian noise; Long-period seismogram.-->

---

## Detection and classification of events from seismograms

<div style="flex-basis: 42%;" align=center>

Most humans can pinpoint events. 
<br>

</div>
<div style="flex-basis: 50%;">

![](images/papers/moran2008helens-nolabels.png)

</div>

<!-- _footer: Moran et al. (2008) -->

---

## Detection and classification of events from seismograms

<div style="flex-basis: 42%;" align=center>

Most humans can pinpoint events.
Experts can classify them.

</div>
<div style="flex-basis: 50%;">

![](images/papers/moran2008helens.png)

</div>

<!-- _footer: modified from Moran et al. (2008) -->

---

## Diving into previously unseed data

<div align="center">

Expert-detected marsquake within continuous insight data

![width:1000px](images/papers/clinton2021marsquake.jpg)

</div>

<!-- _footer: Clinton et al. (2021) -->

---

## Target tasks of machine learning

<div style="flex-basis: 40%;">

- Time-consuming tasks
- Unprogrammable tasks
- Hard-to-describe tasks
- Exploration of new data


</div>
<div style="flex-basis: 57%;">

![](https://scikit-learn.org/stable/_static/ml_map.png)

</div>

---

<!-- _class: titlepage-->

# 2. Definitions

__Machine learning__ is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can effectively generalize and thus perform tasks without explicit instructions. 

---

## General definition of machine learning

<div style="flex-basis:45%;"/>

An algorithm learns from <span style="color:var(--color-lava)">experience</span> with respect to a <span style="color:var(--color-terra)">task</span> and <span style="color:var(--color-olivine)">performance</span>, if its <span style="color:var(--color-olivine)">performance</span> at solving the <span style="color:var(--color-terra)">task</span> improves with <span style="color:var(--color-lava)">experience</span>.

__All three elements are required.__

</div>
<div style="flex-basis:30%;">

<svg viewBox="0 0 370 350" font-size="25px" text-anchor="middle" style="padding: 30px">
<circle cx="100" cy="100" r="100" fill=var(--color-magma) opacity="0.1"/>
<circle cx="260" cy="100" r="100" fill=var(--color-terra) opacity="0.1"/>
<circle cx="180" cy="240" r="100" fill=var(--color-olivine) opacity="0.1"/>
<text x="100" y="100" alignment-baseline="middle" fill=var(--color-magma)>Experience</text>
<text x="260" y="100" alignment-baseline="middle" fill=var(--color-terra)>Task</text>
<text x="180" y="240" alignment-baseline="middle" fill=var(--color-olivine)>Performance</text>
</svg>

</div>


---

## The data, the model, and the loss

<div style="flex-basis: 25%;background-color: var(--color-sable); border-radius: 40px; padding: 30px;" align=center>

🙊
__the data__ 

A set of samples and labels to train from 

$$\mathcal{D} = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^N$$

ere $\mathbf{x}_i$ is the input and $\mathbf{y}_i$ is the output. wh

</div>
<div style="flex-basis: 25%;background-color: var(--color-sable); border-radius: 40px; padding: 30px;" align=center>

🙉 
__the model__ 

A function $f_\theta$ that maps data $\mathbf{x}$ to a prediction $\hat{\mathbf{y}}$ 

$$f_\theta : \mathbf{x} \mapsto \hat{\mathbf{y}}$$

where $\theta$ denotes the parameters.

</div>
<div style="flex-basis: 25%;background-color: var(--color-sable); border-radius: 40px; padding: 30px;" align=center>

🙈 
__the loss__

is a functional that measures the error

$$\mathcal{L}(f_\theta(\mathbf{x}), \mathbf{y})$$

it allows to evaluate the model's performance.

</div>

---

## Another more formal definition of learning


<div align="center">

Find the optimal parameters $\theta^*$ that minimizes the loss $\mathcal{L}$

$$\theta^* = \arg\min_\theta \mathcal{L}\Big(f_\theta(\mathbf{x}), \mathbf{y}\Big)$$

</div>

---

## Useful vocabulary and symbols

<div style="flex-basis: 40%;" align="center">

| Symbol | Name |
|:-|:-|
|$\left\{ \mathbf{x}_i \in \mathbb{X} \right\}_{i =  1\ldots N}$| Collection of __data samples__|
|$\left\{ \mathbf{y}_i \in \mathbb{Y} \right\}_{i =  1\ldots N}$| Collection of __labels__|
|$\mathbf{x}=(x_1, \ldots, x_F)$| Set of sample __features__|
|$\mathbf{y}=(y_1, \ldots, y_T)$| Set of label __targets__|
|$N$| Dataset size|
|$F$| Feature space dimensions|
|$T$| Target space dimension|
|$\mathbb{X}$| Data space|
|$\mathbb{Y}$| Label space|

</div>
<div style="flex-basis: 40%;">

__For instance__, an image is a sample $x$ with 

$$x \in \mathbb{X} = \mathbb{R}^{H \times W \times C}$$

where $H$ is the height, $W$ the width, and $C$ the channels. The label of an image can be represented by a category $y$ with

$$y \in \mathbb{Y} = \{0, 1, \ldots, K\}$$

where $K$ is the number of categories.

</div>

---

## Main types of machine learning



<div style="flex-basis: 25%; font-size:smaller;" align=center data-marpit-fragment="0">

![width:265px](images/diagrams/mathworks-supervised.png)

Predict some output $\mathbf{y}$ from input $\mathbf{x}$ (regression, classification).

</div>
<div style="flex-basis: 25%; font-size: smaller" align=center data-marpit-fragment="1">

![width:250px](images/diagrams/mathworks-unsupervised.png)

Learn data distribution $p(\mathbf{x})$ or structure (clustering, reduction).

</div>
<div style="flex-basis: 25%; font-size:smaller; opacity: 0.5" data-marpit-fragment="2">

![width:265px](images/diagrams/mathworks-reinforcement.png)

Learns a policy to maximize the reward (game playing, robotics).

</div>

<!-- _footer: illustration from www.mathworks.com -->

---

<!-- _class: titlepage-->

# 3. Supervised machine learning

How to solve a regression or classification task with machine learning?

[<img src="images/logo/logo-github.svg" width=25 align="left" style="margin-top:10px; margin-right: -20px;"/> `leonard-seydoux/earth-data-science`](https://github.com/leonard-seydoux/earth-data-science)

---

## The two main tasks of supervised machine learning

<div align=center>

__Regression__
$x$ and $y$ are continuous

![width:300px](images/supervised/linear_regression.svg)

</div>
<div align=center>

__Classification__
$x$ is continuous and $y$ is descrete 

![width:300px](images/supervised/linear_classification.svg)

</div>

---

## The two main tasks of supervised machine learning

<div align=center>

__Regression__
$x$ and $y$ are continuous

![width:300px](images/supervised/linear_regression.svg)

</div>
<div align=center style="opacity: 0.3;">

__Classification__
$x$ is continuous and $y$ is descrete 

![width:300px](images/supervised/linear_classification.svg)

</div>

--- 

## The regression task

<div  style="flex-basis: 50%">

Given a dataset 

$$\mathcal{D} = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^N,$$

optimize the parameters $\theta$ of a function $f_\theta$ that best predicts the label $\mathbf{y}$ from the sample $\mathbf{x}$, that is find the optimal parameters $\theta^*$ that minimizes the loss $\mathcal{L}$, such as

$$\theta^* = \arg\min_\theta \mathcal{L}\Big(f_\theta(\mathbf{x}), \mathbf{y}\Big).$$

</div>
<div style="flex-basis: 30%" align=center>

![width:400px](images/supervised/linear_regression_math.svg)

</div>


---

## The linear regression

<div style="flex-basis: 50%">

Find the set of coefficients $\theta = (a, b) \in \mathbb{R}^2$ that best predicts $y$ from $x$ so that

$$f_\theta : x \mapsto ax + b.$$

Here, the best explanation relates to the loss. For instance, the mean squared error:

$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \left( f_\theta(x_i) - y_i \right)^2.$$

__How do we minimize the loss?__

</div>
<div style="flex-basis: 30%" align=center>

![width:400px](images/supervised/linear_regression_math.svg)

</div>

---

## Naive attempt with a grid search

<div style="flex-basis: 50%">

Grid search to find $\theta^*$ among gridded values of $\theta$. 

__Pros__: easy to implement, exhaustive search, uncertainty estimation.

__Cons__: unscalable. If 0.1s / evaluation, then 2 parameters with 100 values each takes 1/2 hour. 
For 5 parameters it takes more than 30 years!

__Any smarter idea?__

</div>
<div style="flex-basis: 30%" align=center>

![width:400px](images/supervised/linear_regression_brute_force.svg)


</div>

---

## Random search

<div style="flex-basis: 50%">

Random search to find $\theta^*$. 

__Pros__: easy to implement, scalable, uncertainty estimation, can include prior knowledge.

__Cons__: not exhaustive, can be slow to converge.

</div>
<div style="flex-basis: 30%" align=center>

![width:400px](images/supervised/linear_regression_random.svg)

</div>

---

## Gradient descent

<div style="flex-basis: 50%">

Estimate the gradient of $\mathcal{L}$ w.r.t. the parameters $\theta$, update the parameters towards gradient descent.

__Pros__: converges faster than random search.

__Cons__: gets stuck in local minima, slow to converge, needs for differentiability.

</div>
<div style="flex-basis: 30%" align=center>

![width:400px](images/supervised/linear_regression_gradient_descent.svg)

</div>

---

## Gradient descent

<div style="flex-basis: 40%">

__Recipe__

1. Define an initial model $\theta = (a_0, b_0)$
1. Compute the gradient $\nabla \mathcal{L}(\theta)$
1. Update the model $\theta \leftarrow \theta - \eta \nabla \mathcal{L}(\theta)$
1. Repeat until convergence

The __learning rate__ $\eta$ controls the update.
A __training epoch__ is the number of iterations.


</div>
<div style="flex-basis: 30%" align=center>

![width:600px](images/supervised/gradient_descent_3d.svg)

</div>

---

## How to deal with learning rate?

<div align=center>

![width:900px](images/supervised/learning_rate.svg)

That's part of the __hyperparameters__ tuning.
More about that in the deep learning lectures.

</div>

---

## The problem of overfitting

<div align=center>

![width:900px](images/supervised/overfitting.svg)

Having a loss close to 0 does not mean that the model __generalizes__ well.

</div>

---

## Key concepts to prevent overfitting: split the dataset

<div align=center>

![width:900px](images/supervised/splitting.svg)

By splitting the dataset into a __training__ and a __testing__ set, we can evaluate the model's performance on unseen data. __You need stationary data.__

</div>

--- 

## Key concepts to prevent overfitting: regularization

<div align=center>

Add a penalty term $\mathcal{R}$ to the loss $\mathcal{L_R} = \mathcal{L} + \lambda \mathcal{R}$, where $\lambda$ is the regularization strength.

![width:900px](images/supervised/regularization.svg)

The regularization penalizes the model's complexity. 

</div>

---

## Why so many regression algorithms?

<div style="flex-basis: 40%">

Because of combination of models, losses, and regularizations.
[Scikit-learn.org](https://scikit-learn.org/stable/) provides a unified interface in a `greybox style`.

__The model selection is made by trial and error, or by experience.__

</div>
<div style="flex-basis: 46%" align=center>

<iframe src="https://scikit-learn.org/stable/supervised_learning.html#supervised-learning" width="550px" height="500px" style="border: none; box-shadow: 0px 0px 20px #ccc; border-radius: 10px; margin-bottom: -20px;"></iframe>

<br>
</div>

---

## Guidelines for exploring relevant models

![](https://scikit-learn.org/stable/_static/ml_map.png)

--- 

<!-- _footer: Jupyter `x` Obsera -->

## Calibrate a turbidity sensor to estimate the suspended load in rivers


![width:550px](images/notebooks/lab_1_sensor_calibration.svg) 

<img src="images/logo/logo-obsera.png" width=110px style="position:absolute; right:70px; bottom:40px;">

![width:550px drop-shadow sepia:0.4](images/notebooks/lab_1_picture_of_river.png)<br>


---

<!-- _footer: Jupyter `x` PhaseNet -->

## Find out _P_ and _S_ waves within continuous seismograms

<div align=center>

![width:600px](images/papers/zhu2018phasenet.png)

How do you addess this regression problem? More after the deep learning lectures.

</div>

---

## The two main tasks of supervised machine learning

<div align=center style="opacity: 0.3;">

__Regression__
$x$ and $y$ are continuous

![width:300px](images/supervised/linear_regression.svg)

</div>
<div align=center>

__Classification__
$x$ is continuous and $y$ is descrete 

![width:300px](images/supervised/linear_classification.svg)

</div>

---

<!-- _footer: scikit-learn.org -->

## The classification task

<div align=center>

![](images/supervised/classification.svg)

Here again, we have many possibilities.

</div>

---

## The classification task

<div style="flex-basis: 45%">

__Experience__: manual labels $\mathbf{y} \in \{0, 1\}$ obtained from various cases, where two features $\mathbf{x} \in \mathbb{R}^2$ are measured.

__Task__: predict the category $\hat{\mathbf{y}}$ of the samples $\mathbf{x}$.

__Performance__: how should we measure the performance of a classifier?

</div>
<div style="flex-basis: 35%" align=center>

![width:600px](images/supervised/svc.svg)

</div>

---

<!-- _footer: Wikipedia on Support Vector Machines -->

## The classification task with support vector machines (SVM)

<div style="flex-basis: 50%">

Support vector machines search the hyperplane of normal vector $\mathbf{w}$ and bias $b$ that split the classes.

> Note: in 2D, a hyperplane is a line.

The support vectors are the samples that are closest to the other class.


</div>
<div style="flex-basis: 35%" align=center>

![width:400px](https://upload.wikimedia.org/wikipedia/commons/7/72/SVM_margin.png)

</div>

---

## The classification task with support vector machines (SVM)

<div style="flex-basis: 45%" >

The decision function $f(\mathbf{x})$ dependson  the sign of the linear combination of the normal vector and the sample:

$$f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$$

The quantity to minimize is the __hinge loss__:

$$\mathcal{L}(\mathbf{w}, b) = \frac{1}{N} \sum_{i=1}^N \max\left(0, 1 - y_i \left(\mathbf{w} \cdot \mathbf{x}_i + b\right)\right)$$

<br>

</div>
<div style="flex-basis: 35%" align=center>

![width:600px](images/supervised/svc.svg)

</div>

---

## The classification task with support vector machines (SVM)

<div style="flex-basis: 45%" >

The decision function $f(\mathbf{x})$ dependson  the sign of the linear combination of the normal vector and the sample:

$$f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$$

The quantity to minimize is the __hinge loss__:

$$\mathcal{L}(\mathbf{w}, b) = \frac{1}{N} \sum_{i=1}^N \max\left(0, 1 - y_i \left(\mathbf{w} \cdot \mathbf{x}_i + b\right)\right)$$

__What about non linear problems?__

</div>
<div style="flex-basis: 35%" align=center>

![width:600px](images/supervised/svc.svg)

</div>

--- 

<!-- _footer: www.medium.com -->

## The kernel trick for non linear classification problems

<div align=center>

The kernel trick allows to map the data to a higher dimensional space made from the input features, e.g., $\phi(\mathbf{x}) = (x_1, x_2, x_1^2 + x_2^2)$, where the problem is linearly separable. 

![width:650px](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*mCwnu5kXot6buL7jeIafqQ.png)

For instance, the Radial Basis Function (RBF) kernel is infinite:

$$K(\mathbf{x}, \mathbf{x}') = \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\sigma^2}\right)$$

</div>

---

## Generalization of the SVM: the support vector classifier (SVC)

<div align=center>

The SVC is a generalization of the SVM that takes into account more than two classes.


![width:1000px](images/supervised/svc_multiclass.svg)

The decition function is linear in the kernel space only. 
We can project it back to the data space to inspect it.

</div>

---

## Various classifcation metrics: the confusion matrix

<div align=center style="flex-basis: 100%;">

![width:800px](images/supervised/svc_multiclass.svg)

</div>
<div align=center style="margin-top:-50px;margin-left:-20px">

![width:810px](images/supervised/svc_confusion.svg)

</div>

---

## Various classifcation metrics: accuracy, precision, recall


![width:900px](https://www.researchgate.net/publication/336402347/figure/fig3/AS:812472659349505@1570719985505/Calculation-of-Precision-Recall-and-Accuracy-in-the-confusion-matrix_W640.jpg)


---

## Decision trees and random forests

---

## Representation matters!

<div align=center>

![width:700px](images/deep-learning-book/figure-1-1.png)

There is no need for a complex model if you have a good representation of the data.

</div>

---

## Learning strategies depeding on the task complexity


<div style="flex-basis:26%;">

We can engineer features from the raw data to improve the model's performance. 

We can also __learn__ the features from the data.

</div>
<div style="flex-basis: 59%;">

![width:720px](images/deep-learning-book/figure-1-5.png)

</div>

---

<!-- _footer: still from Valentine & Trampert (2012) -->

## Why natural data representations are hard to learn from?

<div style="text-align: center">

We can see waveforms $\mathbf{x}\in\mathbb{R}^N$ as points of a $N$-dimensional space 
<img src="images/waveforms/waveform_0.png" width=700/>

Yet, seismic waveform do not occupy this space fully, likely very sparse.

__Lots of dimensions, but few data.__

</div>
<div style="flex-basis: 50%;">

![width:520px](images/papers/valentine2012spot.png)

</div>

---

## Also, images may not be the best representation of data

<div>

Random reshuffling of the pixels of a face. What is the likelihood that the reshuffled image is a face?

<br>

![](images/deep-learning-book/figure-x-x-1.jpg)

</div>
<div>

![](images/deep-learning-book/figure-x-x-2.png)

</div>

<!-- _footer: modified from Goodfellow et al. (2016) -->

---

## Representation matters

<div>

__The exclusive OR problem (XOR)__ is a simple problem not linearly separable, hard to learn using traditional machine learning algorithms. Multi-layer perceptrons can.

</div>

<div style="flex-basis: 40%;">

![](images/supervised/xor.png)

</div>


---

## Supervised learning for sismo-volcanic signal classification

<div>

__Supervised learning__ experiences a set of examples containing features $\mathbf{x}_i \in \mathbb{X}$ associated with labels $\mathbf{y} \in \mathbb{Y}$ to be predicted from the features (here, classification).

<img src="images/examples/malfante_2018.png" width=900/>

</div>

<!-- _footer: from Malfante et al. (2018) -->


---

## Supervised learning for sismo-volcanic signal classification

<div>

In this case, $\mathbf{x}$ lies in $\mathbb{R}^{3 \times N}$, and $\mathbf{y}$ in $[0, \ldots, 5]$. Which __representation__ of $\mathbf{x}$ works best?

<img src="images/examples/malfante_2018.png" width=900/>

</div>

<!-- _footer: from Malfante et al. (2018) -->


---

## Handcrafted features for classical machine learning

<div style="flex-basis: 40%; margin-right: 40px;" align=center>

We need to find relevant descriptors of our data, used as features $\mathbf{x}$.

<img src="images/examples/features_signal.png" width=500/>

</div>
<div style="flex-basis: 50%;">

![](images/examples/features.png)

</div>

<!-- _footer: Jasperson et al. (2022) and Malfante et al. (2018) -->

---

## Performance measure

<div style="text-align: center">

Accuracy of the predictions measures the model's performance (= confusion matrix)

<img src="images/examples/malfante_accuracy.png" width=800/>

<br>

What is the guarantee that the features we choose are the best ones?

</div>

<!-- _footer: Malafante et al. (2018) -->

---

<!-- _class: titlepage-->

# 4. Deep learning

How deep learning works? What is a neural network? How to train it, and what for?

[<img src="images/logo/logo-github.svg" width=25 align="left" style="margin-top:10px; margin-right: -20px;"/> `leonard-seydoux/earth-data-science`](https://github.com/leonard-seydoux/earth-data-science)

---

## A general form of an artificial neuron

<div style="flex-basis: 39%;">

A __neuron__, or unit, takes a set of inputs $\bf x$ and outputs an activation value $h$, as

$$
h = \varphi\left(\sum_{i=0}^{N} w_i x_i + b \right)
$$

with $w_i$ the weights, $b$ the bias, $\varphi$ is the activation function, and $N$ is the number of inputs.

<br>

</div>

<div style="flex-basis: 50%;">

![](images/models/neuron.png)

</div>

---

## A famous neuron: the sigmoid unit

<div style="flex-basis: 39%;">

A __neuron__, or unit, transforms a set of inputs $\bf x$ into an output $h$, as

$$
h = \varphi\left(\sum_{i=0}^{N} w_i x_i + b \right)
$$

with $w_i$ the weights, $b$ the bias, $\varphi$ is the activation function, and $N$ is the number of inputs. Common activation functions include the __sigmoid__ function, defined as

$$
\varphi(z) = \frac{1}{1 + e^{-z}}
$$

</div>
<div style="flex-basis: 40%;">

<img src="images/models/sigmoid.png">

</div>

---

## An even more famous one: the rectified linear unit

<div style="flex-basis: 39%;">

A __neuron__, or unit, transforms a set of inputs $\bf x$ into an output $h$, as

$$
h = \varphi\left(\sum_{i=0}^{N} w_i x_i + b \right)
$$

with $w_i$ the weights, $b$ the bias, $\varphi$ is the activation function, and $N$ is the number of inputs. Common activation functions include the __rectified linear unit__ (ReLU), defined as

$$
\varphi(z) = \max(0, z)
$$

<!-- _footer: ReLU are empirically preferred to sigmoid units for  computational efficiency no saturation when $x$ is large.-->

</div>
<div style="flex-basis: 40%;">

<img src="images/models/relu.png">

</div>

---

## The multilayer perceptron (MLP)

<div style="flex-basis: 39%;">

A __multilayer perceptron__ is a neural network with multiple hidden layers:

$$
\begin{align*}
h_i^{(1)} &= \varphi^{(1)}\left(\sum_j w_{ij}^{(1)}x_j + b_i^{(1)}\right)\\
h_i^{(2)} &= \varphi^{(2)}\left(\sum_j w_{ij}^{(2)}h_j^{(1)} + b_i^{(2)}\right)\\
y_i &= \varphi^{(3)}\left(\sum_j w_{ij}^{(3)}h_j^{(2)} + b_i^{(3)}\right)
\end{align*}
$$

</div>
<div style="flex-basis: 35%;">

<img src="images/models/mlp_annotated.png">

</div>

---

## The multilayer perceptron (MLP)

<div style="flex-basis: 39%;">

A __multilayer perceptron__ is a neural network with multiple hidden layers. Generally speaking (omitting the biases):

$$
y = \varphi^{(\ell)}\left(\mathbf{W}^{(\ell)}\varphi^{(\ell - 1)}\left(\mathbf{W}^{(\ell - 1)} \ldots \varphi^{(1)}\left(\mathbf{W}^{(1)}\mathbf{x}\right) \ldots \right)\right)
$$

</div>
<div style="flex-basis: 35%;">

<img src="images/models/mlp_annotated.png">

</div>

---

## Quick example for solving XOR

<div style="flex-basis: 39%;" align=center>

Multi-layer perceptrons that solves the XOR problem with binary activations:

<img src="images/models/xor.png" width="70%">



</div>

<div style="flex-basis: 40%;">

![](images/datasets/xor.png)

</div>

<!-- _footer: See Section 6.1 of Goodfellow et al. (2016) -->

---

## Gradient descent for neural networks

<div>

We note $f_\theta(x): x \mapsto y$ the model, where $\theta$ are the parameters of the model (including biases and weights).

1. __Learning__ is the process of finding the parameters $\theta^*$ that minimize the loss $\mathcal{L}$.

2. The __backpropagation__ computes the loss function gradient with respect to $\theta$.

3. The __gradient descent__ updates $\theta$ in the direction of the steepest descent.

</div>

<div style="flex-basis: 33%;">

<img src="images/models/gd.gif" width="90%">

</div>

---

## Gradient computation with backpropagation

<div>

1. __Initialization__: the weights are initialized randomly, the biases to zero
2. __Feed forward__: the input is propagated through the network to compute the output
3. __Loss__: the loss is computed between the output and the target
4. __Back propagation__: computation of the gradient from the loss to the input
5. __Gradient descent__: update the parameters in the direction of the steepest descent

</div>

---

## Gradient-based optimization

<div>

Once the gradient is computed, the parameters are updated using the __gradient descent__ algorithm:

$$
\begin{align*}\\
\theta &\leftarrow \theta - \eta \frac{\partial \mathcal L}{\partial \theta}
\end{align*}
$$

where is $\eta$ the __learning rate__ that controls the size of the update.

</div>

<div style="flex-basis: 33%;">

<img src="images/models/gd.gif" width="90%">

</div>

---

## Gradient descent common issues

<div>

- __Local minima__: getting stuck in a local minimum.

- __Sattling points__: behaves as a local minimum but is not.

- __Plateau__: flat loss function, vanishing gradient, slow convergence.

</div>

<div style="flex-basis: 33%;">

<img src="images/models/gd_issues.png">

</div>

---

## Gradient descent common issues with plateau

<img src="images/models/relu.png" height=350 align="left"/>
<img src="images/models/sigmoid.png" height=350 align="left"/>

__Plateau__ are flat regions of the loss function where the gradient is zero. This can happen with activation functions such as the sigmoid function with saturation. It can also happen with the ReLU function for inputs with negative values.

---

## Gradient-descent tricks to avoid issue

- __Learning rate__: seyt up, and maybe adapt.
- __Momentum__: use the gradient of the previous iteration to update the parameters.
- __Normalization__: normalize the inputs of each layer.
- __Stochastic gradient descent__: use a mini-batch of samples to compute the gradient.
- __Dropout__: randomly drop some neurons during training.

---

## Gradient descent and learning rate

<div>

The __learning rate__ is a hyperparameter that controls the size of the update of the parameters:

$$
\theta \leftarrow \theta - \eta \cfrac{\partial \mathcal L}{\partial \theta}
$$

We must look for a learning rate to avoid local minima while still converging fast enough, without diverging.

> We can also __adapt__ the learning rate.

</div>

<div style="flex-basis: 45%;">

<img src="images/models/lr.png" width="100%">

</div>

---

## Gradient descent and momentum

<div>

The __momentum__ is a technique to accelerate the gradient descent by adding a fraction of the gradient of the previous iteration:
$$
\begin{align*}
p &\leftarrow \alpha p - \eta \frac{\partial \mathcal L}{\partial \theta}\\
\theta &\leftarrow \theta + p
\end{align*}
$$
where $\alpha$ is the a damping parameter, and $v_i$ is the __velocity__. Lower values of $\alpha$ give more weight to the current gradient, higher values give more weight to the previous gradients.

<img src="images/models/mom.webp" width=600>

</div>

<!-- _footer: From Zhang et al. (2021) -->

---

## Data normalization

<div

To avoid getting in the saturation of sigmoidal activation functions, it is important to normalize the data. This can be done by __normalizing the input and the features__:

$$
\hat x_i = \frac{x_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}
$$

where $\mu_i$ is the mean of the input, $\sigma_i$ is the standard deviation of the input, and $\epsilon$ is a small constant to avoid division by zero. You can also apply the normalization after the activation function.

</div>

---

## Gradient descent and learning rate

<div>

The __learning rate__ is a hyperparameter that controls the size of the update of the parameters:

$$
\theta \leftarrow \theta - \eta \cfrac{\partial \mathcal L}{\partial \theta}
$$

We must look for a learning rate to avoid local minima while still converging fast enough, without diverging.

> We can also __adapt__ the learning rate.

</div>

<div style="flex-basis: 45%;">

<img src="images/models/lr.png" width="100%">

</div>

---

## Monitor the training curves

<div style="flex-basis: 50%;">

The __training curves__ are a good way to monitor the training of a model.

- Slow: increase the learning rate.
- Growing: decrease the learning rate.
- Cross-validation: within 0.0001 to 0.1

</div>
<div>

<img src="images/models/learning_curve.png" width="80%">

</div>

---

## Stochastic gradient descent

<div>

The gradient of the loss function with respect to the parameters $\theta$ is computed using the __full-batch gradient descent__ equal to:

$$
\frac{\partial \mathcal L}{\partial \theta} = \frac{1}{N} \sum_{i=1}^N \frac{\partial \mathcal L^{(i)}}{\partial \theta}\\
$$

The __stochastic gradient descent__ is a technique to compute the loss gradient from every sample in the dataset at each iteration.

</div>

<div style="flex-basis: 20%;">

<img src="images/models/sgd.png">

</div>

---

## Mini-batch gradient descent

<div>

The __mini-batch gradient descent__ is a technique to compute the gradient of the loss function with respect to a subset of the dataset. It is a compromise between the full-batch gradient descent and the stochastic gradient descent.

<br>

<img src="images/models/mb.png">

</div>


---

## Overfitting and underfitting

<div style="flex-basis: 20%;">

__Overfitting__: too complex model, does not generalize to new data.

__Underfitting__: too simple model, does not capture the data structure.

</div>

<div style="flex-basis: 70%;">

<img src="images/models/fitting_mod.png">

</div>

---

## Splitting the dataset into train and test sets

<div style="flex-basis: 45%;">

The __training set__ is used to train the model. The __test set__ is used to evaluate the model generalization error on unseen data.

> The typical split is 80% for the training set and 20% for the test set.

</div>

<div style="flex-basis: 40%;">

<img src="images/models/train_test_split.png"/>

</div>

---

## Training and test learning curves

<div style="flex-basis: 40%">

We must ensure that both the training and test losses decrease. If the training loss is much lower than the test loss, the model __overfits__ the training set.

<img src="images/models/train_test_split.png" width="80%"/>

</div>

<div>

<img src="images/models/tt_loss.png" width="90%"/>

</div>

---

## Targetting the right model complexity

<div>

The __model complexity__ is roughly the number of parameters of the model. The __model generalization error__ is the error on the test set.

<img src="images/models/complexity.png" width=70%/>

</div>

---

## Regularization

<div>

__Regularization__ is a technique to control overfitting by adding a penalty term $\mathcal{R}$ to the loss function. The __regularization parameter__ $\lambda$ controls the strength of the regularization.

$$
\mathcal{L}_\mathrm{reg} = \mathcal{L} + \lambda \mathcal{R} = \mathcal{L} + \lambda \|\mathbf{\theta}\|^2_2
$$

<img src="images/models/wd.png" width=60%/>

</div>

<!-- _footer: From Goodfellow et al. (2016) -->

---

## A fully connected network for slving the MNIST classification

<!-- _footer: LeCun et _al._ (1998) -->

<div style="flex-basis: 10%; text-align: center;">

Handwritten digits set of grayscale images $x \in \mathbb{R}^{28 \times 28}$ and classes $y \in \{0, \dots, 9\}$.
<img src="images/datasets/mnist.png" style="width: 70%;">
__Goal__: predict the number encoded in the pixels.

</div>

---

## Example in seismology: fully-connected autoencoder

<!-- _footer: from Valentine and Trampert (2012) -->

<div>

Training of a fully-connected autoencoder on real seismic data.

This is an __unsupervised__ learning task: the input and output are the same.

<img src="images/examples/valentine_2.png" width="80%"/>

</div>

<div>

![](images/examples/valentine_ae.png)

</div>

---

## Example in seismology: fully-connected autoencoder

<!-- _footer: from Valentine and Trampert (2012) -->

<div style="flex-basis: 20%;">

We __learned__ a low-dimensional representation for the seismic data.

</div>
<div>

![](images/examples/valentine_1.png)

</div>
<div style="flex-basis: 20%;">

These are the __latent variables__ of the autoencoder.

</div>

---

## Example in seismology: fully-connected autoencoder

<!-- _footer: from Valentine and Trampert (2012) -->

<div style="flex-basis: 20%;">

Example applications:

- quality assessment
- compression

</div>
<div>

![](images/examples/valentine_3.png)

</div>

---

<!-- _class: titlepage-->

<div  style="flex-basis: 20%;">
<img src="https://lottiefolder.com/wp-content/uploads/2022/11/globe-lottie-animation-5-1.gif" style="margin:20px; width: 400px"/>

</div>
<div style="flex-basis: 40%; margin-right: 100px;">

# 5. Convolutional neural networks

What is the problem with fully connected networks? How can we improve them?

<p style="color: white !important">

<img src="images/titlepages/logo-github.svg" width=25 align="left" style="margin-top:5px; margin-right: -10px;"/>

`leonard.seydoux/machine-learning-courses`

</p>

</div>

---

## Limitations of fully connected networks

<div>

Vision is robust to a lot:

- Translation
- Rotation
- Scaling
- Shearing
- Illumination
- Occlusion

We need invariance to these transformations.

</div>

<div>

<img src="images/models/invariance.jpg"/>

</div>

---

## Example: the handwritten digits

<!-- _footer: LeCun et _al._ (1998) -->

<div style="flex-basis: 10%">

Handwritten digits set of grayscale images $x \in \mathbb{R}^{28 \times 28}$ and classes $y \in \{0, \dots, 9\}$.

<!-- ![bg 50%](images/datasets/mnist.png) -->
<img src="images/datasets/mnist.png" style="width: 70%;">

</div>

---

## Limitations of fully connected networks

<div style="flex-basis: 50%">

An image may be of $200 \times 200$ pixels $\times 3$ color channels. With a __fully connected network__ with $1000$ hidden units, we would have $N = 200 \times 200 \times 3 \times 1000 = 120$M parameters.

__This clearly does not scale to large images.__

</div>

<div>

<img src="images/models/densely.png"/>

</div>

---

## Convolutional neural networks

<div>

__Convolutional layers__ are a type of layer that are used in convolutional neural networks. They are composed of a set of learnable filters.

<img src="images/models/convlay.png" width=80%/>

Each hidden unit look a local content from the input image, althought the weights are shared across the entire image.

</div>

---

## Convolutional neural networks

<div style="flex-basis: 50%">

Discrete image convolution:

$$ (A * B)_{ij} = \sum_n \sum_m A_{nm}B_{i-n, j-m} $$

where $A$ is a input image, and $B$ is a convolutional kernel (weights) to learn.

> Convolutional layers extract local features from the input image ≠ fully connected layers that extract global features.

</div>

<div>

<img src="images/models/no_padding_no_strides.gif" height=350/>

</div>

<!-- _footer: From Vincent Dumoulin, Francesco Visin (2016) -->

---

## Convolution operation

<div style="flex-basis: 50%">

$$(A * B)_{ij} = \sum_n \sum_m A_{nm}B_{i-n, j-m}$$

<img src="images/datasets/zebra_filtered.jpeg"/>

</div>

---

## Convolution operation

<div style="flex-basis: 50%">

$$(A * B)_{ij} = \sum_n \sum_m A_{nm}B_{i-n, j-m}$$

<img src="images/datasets/zebra_edges.jpeg"/>

</div>

---

## Convolution unit

![](images/models/zebra_conv.png)

---

## Convolutional neural network: example with VGG16

<div style="text-align: center">
Now, we can understand this winning architecture for image classification.

<br>
<br>
<img src="images/models/vgg16.png" width=80%/>

Note the last three layers are __fully connected__.
When extracting low-dimensional data from images, this is often needed.

</div>

---

## Convolutional neural network: example with VGG16

<!-- _footer: from Zeiler and Fergus (2013) -->

<div>

Here are the __filters from the first layer__ of VGG16 after training on 100k+ images. These filters collect various shapes, scales, colors, etc.

<img src="images/models/vgg16.png" width=80%/>

</div>
<div>

<img src="images/models/vgg_layer_1.png" width=80%/>

</div>

---

<!-- _class: titlepage-->

<div  style="flex-basis: 20%;">
<img src="https://lottiefolder.com/wp-content/uploads/2022/11/globe-lottie-animation-5-1.gif" style="margin:20px; width: 400px"/>

</div>
<div style="flex-basis: 40%; margin-right: 100px;">

# 6. Applications

The illustration of the previous concepts with examples from seismology. And then you will be ready to apply these concepts to your own problems!

<p style="color: white !important">

<img src="images/titlepages/logo-github.svg" width=25 align="left" style="margin-top:5px; margin-right: -10px;"/>

`leonard.seydoux/machine-learning-courses`

</p>

</div>

---

## Deep-learning applications in seismology

<div style="flex-basis: 30%;">

- Signal detection, pattern recognition
- Classification
- Source localization from sparse or evolving datasets
- Denoising and compression

</div>

<div style="flex-basis: 50%;">

<img src="images/references/seismic_signal_class.png">

</div>

---

## Earthquake detection and location with ConvNetQuake

<div style="flex-basis: 40%;">

__Features__: 3-comp. waveform $x \in \mathbb{R}^{N \times 3}$
__Target__: prob. of event in cell $1$ to $6$
__Loss__: cross-entropy with regularization $\mathcal{L} = - \sum_c q_c \log p_c + \lambda \| \mathbf{w}\|^2_2$

<br>

<img src="images/models/perol_2.png" height=300/>

</div>
<div style="flex-basis: 40%;">

<img src="images/models/perol_1.png"/>

</div>

<!-- _footer: From Perol et al. (2016) -->

---

## Seismic phase picking with PhaseNet

<div style="text-align: center">

__Features__: 3-component seismic signal $x \in \mathbb{R}^{3000 \times 3}$
__Targets__: probabilities $p_i(x)$ of $P$, $S$, and $N$oise over time $= y \in \mathbb{R}^{3000 \times 3}$

<img src="images/models/beroza_example.png" height=500px/>

</div>

<!-- _footer: From Zhu et al. (2016) -->

---

## Seismic phase picking with PhaseNet

<div style="text-align: center">

__Features__: 3-component seismic signal $x \in \mathbb{R}^{3000 \times 3}$
__Targets__: probabilities $p_i(x)$ of $P$, $S$, and $N$oise over time $= y \in \mathbb{R}^{3000 \times 3}$
__Loss__: cross-entropy $\mathcal{L} = -\sum_i\sum_x p(x)\log(q(x))$

<img src="images/models/unet_phasnet.jpg" width=80%/>

</div>

<!-- _footer: From Zhu et al. (2016) -->

---

## Seismic phase picking with PhaseNet

<div style="text-align: center">

__Features__: 3-component seismic signal $x \in \mathbb{R}^{3000 \times 3}$
__Predictions__: likelihood $q_i(x)$ of $P$, $S$, and $N$oise over time

<img src="images/models/beroza_2.png" width=80%/>

</div>

<!-- _footer: From Zhu et al. (2016) -->

---

## Seismic phase picking with PhaseNet

<div style="text-align: center">

__Features__: 3-component seismic signal $x \in \mathbb{R}^{3000 \times 3}$
__Predictions__: likelihood $q_i(x)$ of $P$, $S$, and $N$oise over time

<img src="images/models/beroza_3.png" width=80%/>

</div>

<!-- _footer: From Zhu et al. (2016) -->

---

<!-- _class: titlepage-->

<div  style="flex-basis: 20%;">
<img src="https://lottiefolder.com/wp-content/uploads/2022/11/globe-lottie-animation-5-1.gif" style="margin:20px; width: 400px"/>

</div>
<div style="flex-basis: 40%; margin-right: 100px;">

# 7. Other aspects

Fine tuning, data augmentation, and more.

<p style="color: white !important">

<img src="images/titlepages/logo-github.svg" width=25 align="left" style="margin-top:5px; margin-right: -10px;"/>

`leonard.seydoux/machine-learning-courses`

</p>

</div>

---

## Transfer learning and fine-tuning

<div style="flex-basis: 50%;">

__Transfer learning__ is the use of a pre-trained model on a new task as a initial point for training a new model.

__Fine-tuning__ is the partial re-training of a pre-trained model on a new task, while keeping the weights of the pre-trained layers fixed.

</div>
<div style="flex-basis: 40%;">

<img src="images/examples/scedc_mapplot.png" width=60%/>
<img src="images/examples/iquique_mapplot.png" width=80%/>

</div>

---

## Deep-learning libraries

<div style="flex-basis: 30%;">

__Warning__: libraries are constantly evolving, and the documentation is often incomplete.

</div>
<div style="flex-basis: 50%; columns:2;">

<img src="images/examples/logo_sklearn.png" width=200/>
<br>
<img src="images/examples/logo_tf2.png" width=250/>
<br>
<img src="images/examples/logo_keras.png" width=250/>
<br>
<img src="images/examples/logo_pytorch.png" width=250/>
<br>
<img src="images/examples/logo_julia.png" width=200/>
<br>
<img src="images/examples/logo_seisbench.svg" width=400/>

</div>

---

<div>

__Deep Learning__ book by Goodfellow, Bengio, and Courville.

- History of machine learning
- Linear algebra
- Probability
- Optimization
- Deep learning

</div>
<div>

![drop-shadow](images/references/deep_learning_cover.png)

</div>

---

## References

<!-- _footer: Online documentation at [scikit-learn.org](https://scikit-learn.org/stable) -->

<div style="flex-basis: 40%;">

__The Scikit-Learn Python Toolbox documentation.__

- Machine learning in Python
- Online examples
- Explanation of algorithms
- Grey-box models

</div>
<div style="flex-basis: 27%;">

<iframe src="https://scikit-learn.org/stable/" width="400px" height="520px" style="border: none; box-shadow: 0px 0px 20px #ccc; border-radius: 5px;"></iframe>

</div>

---

## Online tools

<div>

<iframe src="https://playground.tensorflow.org/" width="1280px" height="800px" style="zoom: 1; border: none; margin-top: -250px !important;"></iframe>

</div>

<!-- _footer: Available at https://playground.tensorflow.org -->

---

## References

<div>

<iframe src="https://quickdraw.withgoogle.com/data" width="1280px" height="560px" style="zoom: 1; border: none; margin-top: -47px !important;"></iframe>

</div>

<!-- _footer: Available at https://quickdraw.withgoogle.com/data -->

---

<!-- _class: titlepage-->

<div  style="flex-basis: 20%;">
<img src="https://lottiefolder.com/wp-content/uploads/2022/11/globe-lottie-animation-5-1.gif" style="margin:20px; width: 400px"/>

</div>
<div style="flex-basis: 40%; margin-right: 100px;">

# 8. Notebooks

Now we can move to the notebooks!

<p style="color: white !important">

<img src="images/titlepages/logo-github.svg" width=25 align="left" style="margin-top:5px; margin-right: -10px;"/>

`leonard.seydoux/machine-learning-courses`

</p>

</div>

---

### 1. Basics, data inspection, preprocessing, and regression

<div>

#### Problem

River water suspended load prediction from turbidity.

#### Objectives

Machine learning, data inspection, preprocessing, regression.

</div>

![bg right](images/lab_1.png)

---

### 2. Iris classification

<div>

#### Problem

Retrieve flower species from sepal and petal measurements.

#### Objectives

Machine learning, normalization, classification, cross-validation.

</div>

![bg right:60%](https://camo.githubusercontent.com/6d84afeae253d05e4d8fcffc84ddeb47f65acee0e8448f38ac28e1ac82f49e56/68747470733a2f2f6269736877616d69747472612e6769746875622e696f2f696d616765732f696d6c692f697269735f646174617365742e706e67)

---

### 3. Lidar point cloud classification

<div>

#### Problem

Automate the identification of objects in a lidar cloud from labeled subset.

#### Objectives

Supervised learning, classification, non-linear models, multi-scale features.

</div>

![bg right](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTomIqR3vtw9M5K2TyhvAtLKsSb8gGOGmxQcn0lHdoWO4uGjl3MxQ00TJgi2pvO7jRq2rs&usqp=CAU)

---

### 4. MNIST classification

<div>

#### Problem

Handwritten digit classification.

#### Objectives

Supervised learning, classification, multi-layer perceptron, convolutional neural network.

![bg right 85%](https://www.researchgate.net/publication/306056875/figure/fig1/AS:393921575309346@1470929630835/Example-images-from-the-MNIST-dataset.png)

</div>

---

### 5. Earthquake detection

<div>

#### Problem

Picking of seismic waves from continuous recordings, retrain a pre-trained model to adapt to new data.

#### Objectives

Transfer learning, fine-tuning, convolutional neural network.

![bg right 90%](https://d3i71xaburhd42.cloudfront.net/5ae0f6a3b5fc882ce0b05ff1e8f333caf2e0549e/6-Figure4-1.png)

</div>
