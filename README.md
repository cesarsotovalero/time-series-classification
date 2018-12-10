# timeSeriesClassification
A package for time series classification in Weka

For more information, take a look at the paper: 
**César Soto Valero, Mabel González Castellanos**: _A package for time series classification in Weka_ (CICCI' 2016) [![PDF](http://wwwimages.adobe.com/content/dam/acom/en/legal/images/badges/Adobe_PDF_file_icon_32x32.png)](https://www.researchgate.net/publication/290379731_Paquete_para_la_clasificacion_de_series_temporales_en_Weka)

Weka is a powerful machine learning framework. However, it lacks of tools to handle time series data analysis. 

**timeSeriesClassification** is a package for Weka that implements extensions specially developed for performing the classification of time series data. Those tools are the following: 
- **DTWDistance**: a distance function based on the dynamic time warping dissimilarity measure, knows as DTW 
- **DTWSearch**: a nearest neighbors algorithm for the classification of time series, which takes advantage of the Keogh’s lower bound technique in order to reduce the computational cost of the classification
- **NumerosityReduction**: a filter for numerosity reduction of time series datasets

The experimental results obtained using these extensions on several traditional time series datasets show the validity and utility of this package.
