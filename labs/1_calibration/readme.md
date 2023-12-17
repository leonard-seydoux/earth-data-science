# River load sensor calibration with machine learning

<img src="images/logo-obsera.png" style="margin:20px; height:80px;"/> <img src="images/logo-ipgp-upc.png" height=100 style="margin:20px; height:80px;"/>

![](./images/station-hydrologique-digue.jpeg)
_Station hydrologique du site de la Digue, sur la rivière de Capesterre. © ObsEra_

Welcome to the first lab session of the Earth Data Science course. In this lab, we will focus on understanding and preparing our data before diving into machine learning algorithms. Our hands-on exercise will involve calibrating a turbidity probe to best predict the suspended load in rivers.

The original version of this notebook was made by Antoine Lucas on top of the study made by Amande Roque-Bernard with the help of Gregory Sainton. In order to know more about the scientific context, please refer to Roque-Bernard et al. ([2023](https://doi.org/10.5194/esurf-11-363-2023)) "Phenomenological model of suspended sediment transport in a small catchment", Earth Surf. Dynam., 11, 363–381. The dataset comes from the _OBServatoire de l’Eau et de l’éRosion aux Antilles_ ([ObsEra](https://www.ozcar-ri.org/fr/observatoire-obsera/)), an observatory located in Guadeloupe that gives us information on the erosion of this volcanic island in a few watersheds. With this notebook, you will learn how to prepare and clean a dataset.

The current notebook was edited in 2023 by Leonard Seydoux (seydoux@ipgp.fr) to be used in the course "Earth Data Science" at the [Institut de Physique du Globe de Paris](https://www.ipgp.fr/fr) (IPGP). If you have found a bug or have a suggestion, please feel free to contact me.
