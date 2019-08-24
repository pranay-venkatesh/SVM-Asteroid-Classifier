# SVM-Asteroid-Classifier

Using SVMs to classify asteroids as hazardous or not. Data from Kaggle : https://www.kaggle.com/shrutimehta/nasa-asteroids-classification

Before running the code ensure you've installed the following packages: pandas, numpy, matplotlib, seaborn, datetime, sklearn, random

Which can be done conveniently by using:

```
pip3 install insertpackagenamehere

```
in cmd / terminal.


* Cleaning the data

So, the way I've started the project, is by removing redundant data. For example, the name of a given asteroid is not going to be useful in classifying whether it will be hazardous or not. There are other parameters that come into play when deciding that.

Similarly, a large chunk of data is pretty much repeated. For example, Relative Velocity is given in different units. However, it's pretty much the same data being presented to us. So, I've extracted the same data with only one unit.

There's also a lot of data with minimal amount of correlation to the actual problem. Hence we remove data with low correlation. Seaborn allows us to create a nice heatmap with correlation data, so it lets us identify how many columns are actually useful.

Correlation is a good indicator of values to keep and values to exclude

![heatmap](https://raw.githubusercontent.com/pranay-venkatesh/SVM-Asteroid-Classifier/master/screenshots/Heatmap.png)


After a few tests, it became clear to me that the optimum range for exclusion of values lay somewhere between 0.2 and 0.3
(meaning that if the value of correlation was less than 0.3, it would be excluded, because it becomes evident that the parameter has nothing to do with how hazardous an asteroid is)

0.05 corresponded to roughly 85% accuracy
0.1 corresponded to roughly 92% accuracy
0.2 corresponded to roughly 98% accuracy

And so on.

* Dates

A simple way to deal with the date data given in the dataset is to use datetime package. I've converted all the dates to Epoch Date Format. Where we basically count the number of seconds elapsed since 1st January 1970.

![date information](https://raw.githubusercontent.com/pranay-venkatesh/SVM-Asteroid-Classifier/master/screenshots/Date%20in%20seconds.png)

* Splitting the data into training and testing data

My biggest learning impact from this project was that sklearn allows you to split a given dataset into training and testing data.

```
features_matrix, test_matrix, labels, test_labels = train_test_split(df, df['Hazardous'], test_size=0.2, random_state=0)
```
![splitting data](https://raw.githubusercontent.com/pranay-venkatesh/SVM-Asteroid-Classifier/master/screenshots/Splitting%20data.png)

After the data has been put in place, we just have to fit the model with the appropriate parameters and test the data with our testing matrix.

We then compare our testing matrix's labels with the actual labels from the given data and we see how accurate our model has been.

![accuracy](https://raw.githubusercontent.com/pranay-venkatesh/SVM-Asteroid-Classifier/master/screenshots/Accuracy.png)
