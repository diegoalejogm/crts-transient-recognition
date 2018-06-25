# Transient Event Recognition

This project contains the implementation used for Astronomical Transient Event Recognition using Machine Learning. It is a research project at [Universidad de los Andes](https://uniandes.edu.co) developed by me, and having as supervisors: [Marcela Hernández](https://scholar.google.com.co/citations?user=9nnSYmMAAAAJ&hl=es&oi=ao), [Pablo Arbeláez](https://scholar.google.com.co/citations?user=k0nZO90AAAAJ&hl=es) and [Jaime Forero](https://scholar.google.com.co/citations?user=TLTK6WgAAAAJ&hl=es). This project is also my undergraduate thesis.

State of the art results were obtained by applying the proposed methodology. Random forests were the best performing models, obtaining the following f1 scores:

- Binary Classification: 87.27%
- Six-Transient Classification: 77.54%
- Seven-Transient Classification: 66.39%
- Six-Transient + Non-Transient Classification: 75.05%
- Seven-Transient + Non-Transient Classification: 66.05%

<!--For a brief explanation of the methodology, results and conclusions reached in this project, I invite you to check out this [brief presentation](https://goo.gl/kHqQem).-->  

<!--For the full thesis document, check out this [file](https://github.com/diegoalejogm/crts-transient-recognition/blob/master/Transient%20Recognition%20Thesis.pdf).-->

## Data Used

The input data used in this project can be found in the folder [data](https://github.com/diegoalejogm/crts-transient-recognition/tree/master/data). It was obtained from the [Catalina Real Time Transient Survey](http://crts.caltech.edu). Raw transient dataset consists of a light curve pandas dataframe and a transient catalogue. On the other hand, non-transient raw data is composed by a light curve dataframe only.

## Methodology

The methodology proposed in this project can be found in the [notebooks](https://github.com/diegoalejogm/crts-transient-recognition/tree/master/notebooks) directory. The approach proposed in this research is briefly summarized next. It is recommended to read the full thesis document in the link above:

- **Filtering:** Light curves were filtered in order to have subsets with enough observations. Two subsets of light curves were obtained, by filtering by those having at least 5 and 10 observations minimum, respectively.
- **Oversampling:** For each one of the filtered datasets, a new dataset containing balanced amount of light curves for each transient class was created by executing an oversampling a process. This process consited in using a Gaussian probability distribution for each observation of every light curve in the dataset. Such distribution had the magnitude as its mean and the error as its variance (sigma).
- **Feature extraction:** For each of the 4 datasets generated previously, 31 different measurements were extracted from each light curve.
- **Feature Scaling:** Two different feature scaling methods were implemented so that machine learning methods would take into account each feature with the same relevance.
- **Classification:** Three different machine learning algorithms were trained, using grid search for hyper-parameter tuning, and 2-fold cross validation. The models tested were: Support Vector Machines, Random Forests, Neural Networks.
