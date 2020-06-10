<img src="https://cesarsotovalero.github.io/img/logos/TS-Classification_logo.svg" height="100px"  alt="TS-Classification"/>

[Weka](https://www.cs.waikato.ac.nz/ml/weka) is a powerful machine learning framework. However, it lacks of tools to handle time series data analysis. **TS-Classification** is a package for facilitating time series classification tasks in Weka. 

This package implements the following functionalities: 

- **DTWDistance**: a distance function based on the dynamic time warping dissimilarity measure, [DTW](https://en.wikipedia.org/wiki/Dynamic_time_warping) 
- **DTWSearch**: a nearest neighbors algorithm for the classification of time series, which takes advantage of the [Keogh’s lower bound technique](https://www.cs.ucr.edu/~eamonn/LB_Keogh.htm) in order to reduce the computational cost of the classification with DTW
- **NumerosityReduction**: a filter for numerosity reduction of time series, which is an implementation of the "[Fast time series classification using numerosity reduction](https://dl.acm.org/doi/10.1145/1143844.1143974)" algorithm for Weka.


Visit the wiki for details about the [Installation](https://github.com/cesarsotovalero/timeSeriesClassification/wiki/Installation) and [Usage](https://github.com/cesarsotovalero/timeSeriesClassification/wiki/Usage) examples.

# Citation

If you use this tool, please cite the following [research paper](https://www.researchgate.net/publication/290379731_Paquete_para_la_clasificacion_de_series_temporales_en_Weka):

```
@inproceedings{SotoValero2016,
    author = {C\'esar Soto-Valero, Mabel Gonz\'alez Castellanos},
    title = {Paquete para la clasificación de series temporales en Weka},
    year = {2016},
    publisher = {Ediciones Futuro},
    address = {Cuba},
    booktitle = {III Conferencia Internacional en Ciencias Computacionales e Informáticas},
    pages = {1–13},
    numpages = {13},
    location = {La Havana, CU},
    series = {CICCI' 2016}
}
```
