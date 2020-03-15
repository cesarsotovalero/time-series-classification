---
layout: default
---

<img src="https://cesarsotovalero.github.io/img/logos/TS-Classification_logo.svg" height="100px"  alt="TS-Classification"/>

[Weka](https://www.cs.waikato.ac.nz/ml/weka) is a powerful machine learning framework. However, it lacks of tools to handle time series data analysis. **TS-Classification** is a package for facilitating time series classification tasks in Weka. 

This package implements the following functionalities: 

- **DTWDistance**: a distance function based on the dynamic time warping dissimilarity measure, [DTW](https://en.wikipedia.org/wiki/Dynamic_time_warping) 
- **DTWSearch**: a nearest neighbors algorithm for the classification of time series, which takes advantage of the [Keogh’s lower bound technique](https://www.cs.ucr.edu/~eamonn/LB_Keogh.htm) in order to reduce the computational cost of the classification with DTW
- **NumerosityReduction**: a filter for numerosity reduction of time series, which is an implementation of the "[Fast time series classification using numerosity reduction](https://dl.acm.org/doi/10.1145/1143844.1143974)" algorithm for Weka.


# Citation

If you use this tool, please cite the following paper:


**César Soto Valero, Mabel González Castellanos**. Paquete para la clasificación de series temporales en Weka. In III Conferencia Internacional en Ciencias Computacionales e Informáticas (CICCI' 2016), La Havana, Cuba. [PDF](https://www.researchgate.net/publication/290379731_Paquete_para_la_clasificacion_de_series_temporales_en_Weka)


# Installation 

To use the **timeseriesClassification** package, make sure you have installed Weka > 3.7.

Go to `~/wekafiles/packages` and decompress the `timeSeriesClassification.rar` file there. 

In Linux, you can do this by executing the following commands:

```
git clone https://github.com/cesarsotovalero/timeSeriesClassification.git
cd timeSeriesClassification
tar timeSeriesClassification -C ~/wekafiles/packages  
```

After this, open Weka normally. The **timeseriesClassification** package will be automatically loaded and its features should be available through the GUI and CLI user interfaces provided by Weka.

# Usage

Here, we will rely on an example to illustrate how to use the **timeSeriesClassification** package to classify time series data. 

## Classification example

Download the [ECG5000](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000) dataset.

Load the train dataset `ECG5000_TRAIN.arff` in the Weka explorer.

![Weka Explorer](https://github.com/cesarsotovalero/cesarsotovalero.github.io/blob/master/img/posts/time_series_classification/weka_explorer.png)

In the explorer, go to Classify and add `ECG5000_TEST.arff` file as the test set.

![Add Test Set](https://github.com/cesarsotovalero/cesarsotovalero.github.io/blob/master/img/posts/time_series_classification/weka_test.png)

Then, configure the classifier by selecting Lazy > Ibk > Choose > **DTWSearch**

![DTWSearch](https://github.com/cesarsotovalero/cesarsotovalero.github.io/blob/master/img/posts/time_series_classification/weka_dtw.png)

Now you can run the classifier with the **DTWDistance** function, you should obtain the following result:

![DTWSearch Results](https://github.com/cesarsotovalero/cesarsotovalero.github.io/blob/master/img/posts/time_series_classification/weka_dtw_result.png)

BTW, the accuracy using 1NN with the Euclidean Distance instead of DTW for this dataset is 92.2444%.

## Preprocess example

For using the **NumerosityReduction** filter. In the Weka explorer go to Choose > weka > filters > supervised >
instance > **NumerosityReduction** and select the percentage of instances to be removed

![Weka Preprocess](https://github.com/cesarsotovalero/cesarsotovalero.github.io/blob/master/img/posts/time_series_classification/weka_preprocess.png)

After applying the filter with **percentageToRemove** = 50, the dataset will contain half of the original instances, while preserving the representativeness of each one of the classes for classification

![Weka Preprocess](https://github.com/cesarsotovalero/cesarsotovalero.github.io/blob/master/img/posts/time_series_classification/weka_reduced.png)
