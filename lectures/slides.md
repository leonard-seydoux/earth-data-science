---
theme: presentation
marp: true
math: katex
---

<!-- _class: titlepage-->

# Earth Data Science

Lecture by Léonard Seydoux the master-level classes of the [institut de physique du globe de Paris](https://www.ipgp.fr) with contents inspired by the [scikit-learn](https://scikit-learn.org/stable/) Python library documentation and the [deep learning](https://www.deeplearningbook.org/) book of Ian Goodfellow.

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
|$\mathbf{x}=\{x_1, \ldots, x_F\}$| Set of sample __features__|
|$\mathbf{y}=\{y_1, \ldots, y_T\}$| Set of label __targets__|
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

<div style="flex-basis: 25%; font-size:smaller;" align=center>

![width:265px](images/diagrams/mathworks-supervised.png)

Predict some output $\mathbf{y}$ from input $\mathbf{x}$ (regression, classification).

</div>
<div style="flex-basis: 25%; font-size: smaller" align=center>

![width:250px](images/diagrams/mathworks-unsupervised.png)

Learn data distribution $p(\mathbf{x})$ or structure (clustering, reduction).

</div>
<div style="flex-basis: 25%; font-size:smaller;">

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

By splitting the dataset into a __training__ and a __testing__ set, we can evaluate the model's performance on unseen data. __You need enough data!__

</div>

--- 

## Key concepts to prevent overfitting: regularization

<div align=center>

![width:900px](images/supervised/regularization.svg)

The loss can incorporate a regularization term to penalize the model's complexity.
By adding a regularization term to the loss, we can penalize the model's complexity.

</div>

---

## Why so many regression algorithms?

<div style="flex-basis: 50%">

Every algorithm has its own assumptions and hyperparameters. The [scikit-learn](https://scikit-learn.org/stable/) library provides a unified interface to use them.

- Various models
- Various loss functions
- Various regularizations
- Various optimization algorithms

</div>
<div style="flex-basis: 30%" align=center>

<iframe src="https://scikit-learn.org/stable/supervised_learning.html#supervised-learning" width="400px" height="500px" style="border: none; box-shadow: 0px 0px 20px #ccc; border-radius: 10px; margin-bottom: -20px;"></iframe>

<br>
</div>

---

## Supervised machine learning: the classification task

<div align=center style="max-width: 780px;">

Classification of sismo-volcanic events: dataset of waveforms $\mathbf{x}$ associated with categorical labels $\mathbf{y}$.

<img src="images/examples/malfante_2018.png" width=900/>

</div>

<!-- _footer: from Malfante et al. (2018) -->

---

## Supervised machine learning: the classification task

<div align=center style="max-width: 780px;">

In other terms, $\mathbf{x}$ lies in $\mathbb{R}^{3 \times N}$, and $\mathbf{y}$ in $[0, \ldots, 5]$. 
Which representation of $\mathbf{x}$ works best?

<img src="images/examples/malfante_2018.png" width=900/>

</div>

<!-- _footer: from Malfante et al. (2018) -->

