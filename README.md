# **ML Assignment 2**

### **Problem Statement: Impact of Social Media on Student Well-being**

The objective of this project is to investigate the influence of social media usage on students' daily lives and overall well-being. Utilizing survey data collected from 4,500 students across various educational levels, the study aims to examine the relationships between digital engagement patterns, sleep habits, and stress levels.

Various Machine Learning classification algorithms will be implemented to predict students' well-being and stress categories based on their social media usage and lifestyle-related factors. Furthermore, the performance of the different classification models will be evaluated and compared using appropriate evaluation metrics to identify the most effective model in terms of predictive accuracy and reliability.

The study seeks to derive meaningful insights into the potential effects of digital engagement on student well-being while demonstrating the applicability of Machine Learning techniques in identifying behavioral patterns and predicting well-being outcomes.

**Dataset Description:**

* Source: Kaggle \- Impact of Social Media on Life (harishyadav0506), survey data on screen time, sleep, stress and GPA.  
* Size: 4,500 rows x 16 columns \-\> 14 predictive features after removing \`Student\_ID\` (identifier) and the target \`Overall\_Impact\`.  
* Feature types: 9 numeric (age, daily/weekend usage hours, sleep hours, sleep quality, late-night flag, stress score, mental-health index, GPA), 4 nominal (gender, academic level, primary platform, device type), 1 ordinal (social-comparison frequency: Never \< Rarely \< Sometimes \< Frequently \< Always).  
* Target Column: \`Overall\_Impact\` \- Beneficial 3,681 (81.8%), Neutral 654 (14.5%), Negative 165 (3.7%).  
* Missing values: 131 cells (46 in \`Perceived\_Stress\_Score\`, 85 in \`Academic\_Performance\_GPA\`), imputed with the median inside the pipeline.  
* Split: one stratified 80/20 split (seed 42): 3,600 fit rows, 900 hold-out rows (736 Beneficial / 131 Neutral / 33 Negative). The hold-out rows are \`test\_data.csv\`.

**GitHub Repository:** [https\://github.com/PrachiiShah/Socialmedia-lifeimpact-classifier](https://github.com/PrachiiShah/Socialmedia-lifeimpact-classifier)  

**Models Used:**

| ML Model Name  | Accuracy  | AUC  | Precision  | Recall  | F1  | MCC  |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Logistic Regression**  | .9844 | .9982 | .9556 | .9543 | .9548 | .9500 |
| **Decision Tree**  | .9722 | .9561 | .9134 | .9518 | .9311 | .9112 |
| **kNN**  | .9278 | .9692 | .8663 | .8097 | .8359 | .7563 |
| **Naive Bayes**  | .9300 | .9870 | .8311 | .9187 | .8697 | .7983 |
| **Support Vector Machine** | .9878 | .9985 | .9708 | .9674 | .9690 | .9606 |

| ML Model Name  | Observation about model performance  |
| :---- | :---- |
| **Logistic Regression**  |  |
| **Decision Tree**  | The best settings had no depth limit and one sample per leaf, so it scores 1.000 on the training rows but only 0.933 macro-F1 on the test rows. That points to over-fitting.  |
| **kNN**  | It also scores 1.000 on the training rows, and it has the weakest test recall (0.8275) of all five models. Fourteen features, several of them one-hot columns, dilute the distance measure.  |
| **Naive Bayes**  | It has high AUC (0.9906) but the lowest precision (0.8605), and its best setting was at the extreme end of the smoothing range. Independence assumptions are poor for correlated features like stress, mental health and GPA.  |
| **Support Vector Machine**  | **1\. It generalises best.** In the over-fitting check, SVM has the highest cross-validated macro-F1 on the training data (0.9526) and the highest test macro-F1 (0.9787). Its train score (0.9865) sits close to the others, so there is no sign of memorising the training rows. **2\. It handles the rare class well.** Recall is the metric where SVM leads most clearly (0.9880 vs. 0.9758 for logistic regression). Because the Negative class is only 3.7% of the data, macro-recall, F1 and MCC matter more than accuracy here, and SVM is strongest on them. |
| **Overall, Winner for your dataset?**  | SVM Model with best scores in all 6 metrics. This model generalises the best scores and handles the rare classes well. However, this model ties with the Logistic Regression model, where accuracy and AUC scores differ by only 0.001 and F1 by .006.  |

**Streamlit App:**

