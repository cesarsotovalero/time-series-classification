# timeSeriesClassification
A package for time series classification in Weka

Weka is a powerful machine learning framework. However, it lacks of tools to handle time series data analysis. 

**timeSeriesClassification** is a package for Weka that implements extensions specially developed for performing the classification of time series data. Those tools are the following: 
- **DTWDistance**: a distance function based on the dynamic time warping dissimilarity measure, knows as DTW 
- **DTWSearch**: an nearest neighbors algoritm for classification of time series, which takes advantage of the Keogh’s lower bound technique in order to reduce the computational cost of the classification
- **NumerosityReduction**: a filter for numerosity reduction of time series datasets

The experimental results obtained using these extensions on several traditional time series datasets show the validity and utility of the developed package.
