# Time series classification in Weka

[Weka](https://www.cs.waikato.ac.nz/ml/weka) is a powerful machine learning framework. However, it lacks of tools to handle time series data analysis. **timeSeriesClassification** is a package for facilitating time series classification tasks in Weka. 

This package implements the following functionalities: 

- **DTWDistance**: a distance function based on the dynamic time warping dissimilarity measure, [DTW](https://en.wikipedia.org/wiki/Dynamic_time_warping) 
- **DTWSearch**: a nearest neighbors algorithm for the classification of time series, which takes advantage of the [Keogh’s lower bound technique](https://www.cs.ucr.edu/~eamonn/LB_Keogh.htm) in order to reduce the computational cost of the classification with DTW
- **NumerosityReduction**: a filter for numerosity reduction of time series, which is an implementation of the "[Fast time series classification using numerosity reduction](https://dl.acm.org/doi/10.1145/1143844.1143974)" algorithm for Weka.


Visit the wiki for details about the [Installation](https://github.com/cesarsotovalero/timeSeriesClassification/wiki/Installation) and [Usage](https://github.com/cesarsotovalero/timeSeriesClassification/wiki/Usage) examples.

## Citation

If you use this tool, please cite the following paper:


**César Soto Valero, Mabel González Castellanos**. Paquete para la clasificación de series temporales en Weka. In III Conferencia Internacional en Ciencias Computacionales e Informáticas (CICCI' 2016), La Havana, Cuba. [PDF](https://www.researchgate.net/publication/290379731_Paquete_para_la_clasificacion_de_series_temporales_en_Weka)
