# Baltimore City Crime Prediction #

**The objective of this analysis is to develop a model to predict crime type given certain attributes about the crime (location, time, weapon, etc.) using the Baltimore crime dataset with over 350K instances of crime.**

Understanding the actual distribution of crimes across a city is vital to addressing the problem from a community perspective. Such a model would aid this understanding by detecting mislabeling and missing labels for crime type and adjust the data to better reflect the reality of the community.

Specifically the crime type for the purpose of this prediction task is defined as the first value of the crime code (which is a digit followed by a number of characters e.g. 3C, 6D, 8AV, etc.). This digit defines 9 unique crime types: **(HOMICIDE, RAPE, ROBBERY, ASSUALT, BURGLARY, LARCENY, AUTO THEFT, ARSON, SHOOTING)**

This is done as creating a class for each unique crime code would not be feasible. There are 86 unique crime code values with 31 values having less than 100 instances. This would not be enough data to both train an evaluate on. Furthermore, the initial focus should be to create a model which can predict crime type to simplify the training task for the model. More data for each unique crime code could then allow for building a specialized crime subtype model on top of the original model. The crime description model has 14 unique values (5 being subsets of the original 9). Again, the crime type is used instead of description for the same reason of simplifying the task for the model and making it easier to create specialized models to predict the subtype.

## Data Enrichment ##
<hr/>

To build on this dataset, additional data from baltimorecity.gov was joined with the crime dataset to add information about each neighborhood. Features like population, population density, housing, average household size may be valuable indicators of certain characteristics of neighborhoods useful for prediction. These charactersitcs can reasonably be expected to vary over the years so these values are a potentially useful estimeate, but going forward, year specific data would be better if available.

The next section outlines how this data was joined with the original crime dataset.

## Data Preparation ##
<hr/>

During the data preparation phase, the primary goal is to divide the dataset into a training, validation, and test set.

Importantly, this needs to be done before any analysis of features as well as cleaning so as not to contaminate the validation and test set by biasing the model or feature selection by having seen some of the validation and test distributions. The records where the label is missing should however be dropped, but no instances of this were found.

The only features of the data that are examined are the crime code and descriptions to select the appropraite classes for the prediction task. An additional crime type column is created from the crime code column including just the first digit of the code. Other than that, the dataset is left in its raw form.

Finally, the crime data is joined with the neighborhood data to include neighborhood features including housing, population, population density, population change, average household size, occupied housing, vacant housing. Performing this join removed instances where the neighborhood names did not match. This was only the case for about 1K / 350K records.

**Dataset Split:** The full dataset is split into a training (64%), validation (16%) and test (20%). Using stratified sampling so as to ensure the proper crime type distribution is maintained across each dataset.

**Note on Bias:** Additional demographic information on age and population race was included. The decision was made to ignore these values as they may raise a data bias problem. Certain confounding variables may act through information like race to give some predictive strength. The problem may come in if such a model is applied to a different city or the underlying distribution changes and the model holds on to the demographic information in cases where the confounding variable does not have an effect and an invalid prediction is made on the basis of the demographics.

**Note on Chronological Ordering:** The crime dataset contains records going back to the 1970s, but is skewed towards recent years. It is reasonable to expect the crime type data distribution to change over time. However, the test set is constructed by shuffling the entire dataset instead of using the most recent to simulate this data drift. This is because the purpose of the evaluation is to evaluate the model fit on the known distribution. Without shuffling, this evaluation would be obscured. In the model deployment section below, methods of addressing this issue are outlined.

**Note on Cross Validation:** Cross validation may be a valuable technique as it allows for training and evaluation on the full training set. However, it is not used here as doing so would force the evaluation measure to only reflect model type and hyper parameter performance and not the decisions around feature selection. This is because the decsisions around feature selection would then be using information from the validation set and reducing its value as an evaluation. Feature selection is an important part to subject to evaluation and so a seperate validation set is created and held away from the analysis and feature selection process just as the test set.

## Initial Feature Selection ##
<hr/>

At this stage, the initial features of interest are selected. This is done before the data cleaning so that more rows are not removed than necessary.

The date and time, weapon, inside/outside, district, neighborhood demographics, and premise are selected and experimented with.

Location, post, and geo location were not selected here as this level of granularity may not be necessary given neighborhood and make the model susceptable to memorizing training examples and cause overfitting. In a further analysis, they can be further explored, but for now, are left out.

VRI name is left out as the more complete district column can be used for the same purpose as a lower granularity location feature.

Total incidents is always 1 so it is not useful either.

Neighborhood demographic information including housing, population, population density, and population change is standardized to simplify training for the model. Houshold size and housing per population were already on a similiar scale to all other features so were left in the original form.

## Exploratory Data Analyisis ##
<hr/>

At this point, with the remaining columns, the focus shifts to exploring the distributions of the classes and selected features, dealing with missing values, and feature transformations. This is done only on the training set, again, so as to keep the information from evaluation datasets hidden and better simulate the evaluation on future unseen data.

**Class Imbalance:** It is observed here that the output classes are very imbalanced with the most frequent (larceny) having 78K instances and least frequent (arson) having fewer than 1K. Often when the output classes are heavily imbalanced, a technique to balance the dataset is needed. Either a new sampling process or class balancing in the model loss to give more weight to lower instance classes. However, such techniques would not be appropriate here because such balancing also removes information about the prior distribution. While this may good if there is reason to believe the data collection process was biased, in this case, information about what types of crimes are more prevelant than others is real and valuable. For this reason, it is observed that artifical balanceing here reduces the model prediction performace.

**Missing Values:** In this analysis, I decided not to remove any null or missing features because the instance of a missing value may very well give an indication of the circumstances of the incident. Instead a dummy value is created for instances of missing values to add in this information.

**Date Time Transformation:** From the data time column, the most meaningful information is the time of year and time of day so month and hour data is extracted from the datetime object. Instead of scaling the month and day to a 0-1 scale or one hot encoding it, a more appropriate format that preserves the cyclic nature of the feature so it does not have to be learned is the sin-cos transformation. Month and hour are then represented as...

**Month** = sin(2π (month / 12)), cos(2π (month / 12))

**Hour** = sin(2π (hour / 24)), cos(2π (hour / 24))

**One Hot Encoding:** For the rest of the selected features (weapon, inside/outside, district, neighborhood, and premise), dummy variable were created for each unique value. This however results in over 410 columns primarily due to the neighborhood (279) and premise (129) columns. In the next step, the relevant dummy variable are selected from these features to reduce the number of features being used. It is important to keep this number at a minimum to prevent overfitting.

## Feature Selection ##
<hr/>

In the case of the weapon, date, and inside/outside features, all values are kept including null for the reason described above.

Because the one hot encoded neighborhood and premise values resulted in 279 and 129 features respectively, it is important to select only the most predictive features. This is done using the the multi class F test to determine which values for both neighborhood and premise are most valuable in predicting the crime type given a parameter for the best k values to return. The k value is experimented with during the model development process. Given these top k features, a correlation matrix was used to get some sense of which selected premises are correlated with each other making them redundant. Each selected premise pair had a low correlation so all were kept.

To select the neighborhood demographic information, a correlation matrix was plotted to see which features are strongly correlated and thus mostly redundant. Housing per population and vacant per population were strongly correlated and housing and population were strongly correlated so housing per population and population were selected among each pair respectively and the others removed.

**Note on Concept Drift:** As the nature of crime in the city can reasonably expected to change over the years, so must the model. Importantly, this feature selection process is as much a part of the entire modelling process as the ML model parameters. This is because it involves deciding which information to pay attention to and which to ignore given the current crime distribution. As new data comes in, this feature selection process should be updated along with the model.

## Model Experimentaion ##
<hr/>

**Evaluation Metric:** To effectively meausure the performance of a model on this task given that the classes are so imbalanced, accuracy would not be appropriate. Instead F1 score is used since it takes into account the false positive and false negative rate. For multi class prediction, an f1 for each class is observed and combined using a weighted average based on the occurance of each class.

**Models:** During the experimentation phase, 7 different types of classifier models were evaluated: K-Nearest Neighbor (KNN), Logistic Regression, Gaussian Naive Bayes, Decision Tree, Multi Layered Perceptron, Random Forest, and Gradient Boost. KNN is a very simple model often used for baselines and so it was here. It achieved a weighted f1 score of 0.54 on the validation set.

Next, Naive Bayes and Logistic Regression achieved a weighted f1 score of 0.35 and 0.62 on validation respectively.

Beyond theses relatively simple models, the performace appeared to cap at about 0.6-0.63 on the validation set.

Model hyperparameters such as tree depth, layer dimensions, or classifiers count in the case of ensemble models was experimented with to minimize the overfitting effect while given the model enough degrees of freedom to better learn the task.

**Features:**

In addition to hyper parameter tuning, the input feature selections were also experimented with. Mostly yielding similiar results. The random forest classifier appeared to most consistently get the highest score at around 0.62.

Variation in the number of premises selected had little effect as long as at least the best 10 were used. the inside/outside, neighborhood, month, and hour features had some predictive power but relatively less compared to the weapon and premise data as was expected.

**Ensembling:**

An additional ensemble model was created to combine the the models which performed well on different categories. This may hve been compensated with higher flase positives and false negatives on other classes, but an ensemble was created to see this. Despite this, the validation performace again reached a maximum of 0.63 indicating the compensation in other class performances might be the case.

## Results ##
<hr/>

To test the final performance of each model, the 4 models which performed the best on the validation set were trained on the combined train and validation set and evaluated on the test set. Below is a classification report for each on the final test.


**LOGISTIC REGRESSION**

              precision    recall  f1-score   support

           1       0.04      0.64      0.08        28
           2       0.03      0.34      0.06        50
           3       0.46      0.67      0.55      5093
           4       0.44      0.63      0.52     14254
           5       0.42      0.50      0.45      8096
           6       0.90      0.54      0.68     40494
           7       0.01      0.32      0.03       285
           8       1.00      1.00      1.00       289
           9       0.99      0.70      0.82      1317

    weighted avg   0.72      0.57      0.61     69906

**DECISION TREE**

              precision    recall  f1-score   support

           1       0.25      0.41      0.31       273
           2       0.08      0.25      0.12       159
           3       0.44      0.67      0.53      4881
           4       0.49      0.62      0.54     15875
           5       0.49      0.52      0.50      8975
           6       0.86      0.56      0.68     37725
           7       0.04      0.39      0.07       649
           8       0.97      1.00      0.98       280
           9       0.82      0.71      0.76      1089

    weighted avg   0.69      0.58      0.61     69906

**MLP**

              precision    recall  f1-score   support

           1       0.12      0.62      0.21        91
           2       0.02      0.44      0.03        18
           3       0.49      0.67      0.56      5389
           4       0.49      0.63      0.55     15551
           5       0.49      0.53      0.51      8947
           6       0.87      0.56      0.68     38224
           7       0.01      0.41      0.01       100
           8       1.00      1.00      1.00       289
           9       0.98      0.71      0.82      1297

    weighted avg   0.71      0.58      0.62     69906

**RANDOM FOREST**

              precision    recall  f1-score   support

           1       0.18      0.45      0.26       181
           2       0.07      0.41      0.11        80
           3       0.45      0.72      0.55      4649
           4       0.52      0.64      0.57     16309
           5       0.51      0.55      0.53      8896
           6       0.87      0.56      0.68     37670
           7       0.04      0.42      0.08       655
           8       1.00      1.00      1.00       288
           9       0.90      0.71      0.79      1178

    weighted avg   0.70      0.59      0.62     69906

The models achieved a very similiar performance indicating minimal overfitting. Classes 1, 2, and 7 (HOMICIDE, RAPE, AUTO THEFT) were the most difficult to predict likely due to not having unique indicators in terms of weapon and premise compared to the other classes. In the case of class 7 (AUTO THEST), other class subtypes like LARCENY FROM AUTO and ROBBERY - CARJACKING may have had similiar features. Class 8 (ARSON) performed the best as weapon = "fire" was a perfect predictor. Same in the case of class 9 (SHOOTING) as firearm was a strong predictor.

## Next Steps ##
<hr/>

I believe the best next step to improving performance with more time is to experiment with different feature combinations,feature transformations, and looking for potential other datasets to include. The models appeared to reach the highest performance they could with the selected features.

One more sophisticated approach to feature selection would be to use a Variational Autoencoder (VAE) the encode all features of the data into a lower dimensional latent vector. A VAE is actually a form on nonlinear principal component analysis (PCA) given that it is trained to reconstruct on the input from the input. It is an unsupervised approach and thus can make use of unlabeled data. Regular PCA was not used as the MLP model should implicitly do this. With more time, a VAE feature encoder may be a valuable method to try.

## Model Deployment ##
<hr/>

Full training

How to deploy

Measure concpt drift...