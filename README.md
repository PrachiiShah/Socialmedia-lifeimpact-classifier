# **ML Assignment 2**

### **Problem Statement: Impact of Social Media on Student Well-being**

The objective of this project is to investigate the influence of social media usage on students' daily lives and overall well-being. Utilizing survey data collected from 4,500 students across various educational levels, the study aims to examine the relationships between digital engagement patterns, sleep habits, and stress levels.

Various Machine Learning classification algorithms will be implemented to predict students' well-being and stress categories based on their social media usage and lifestyle-related factors. Furthermore, the performance of the different classification models will be evaluated and compared using appropriate evaluation metrics to identify the most effective model in terms of predictive accuracy and reliability.

The study seeks to derive meaningful insights into the potential effects of digital engagement on student well-being while demonstrating the applicability of Machine Learning techniques in identifying behavioral patterns and predicting well-being outcomes.

**Dataset Description:**

* Source: Kaggle - Impact of Social Media on Life (harishyadav0506), survey data on screen time, sleep, stress and GPA.
* Size: 4,500 rows x 16 columns -> 14 predictive features after removing `Student_ID` (identifier) and the target `Overall_Impact`.
* Feature types: 9 numeric (age, daily/weekend usage hours, sleep hours, sleep quality, late-night flag, stress score, mental-health index, GPA), 4 nominal (gender, academic level, primary platform, device type), 1 ordinal (social-comparison frequency: Never < Rarely < Sometimes < Frequently < Always).
* Target Column Classification: `Overall_Impact`, Distribution - Beneficial 3,681 (81.8%), Neutral 654 (14.5%), Negative 165 (3.7%). Multi-class classification
* Missing values: 131 cells (46 in `Perceived_Stress_Score`, 85 in `Academic_Performance_GPA`), imputed with the median inside the pipeline.
* Split: one stratified 80/20 split (seed 42): 3,600 fit rows, 900 hold-out rows (736 Beneficial / 131 Neutral / 33 Negative). The hold-out rows are stored in `test_data.csv`.
* Executed using - Python 3.12 and scikit-learn 1.6.1.

**GitHub Repository:** [https\://github.com/PrachiiShah/Socialmedia-lifeimpact-classifier](https://github.com/PrachiiShah/Socialmedia-lifeimpact-classifier)  

**Repository Layout:**
app.py						    Streamlit front-end dashboard & reporting app
data/						    Contains Social_media_impact_on_life.csv full data file
figures/						Stores the graphs and visuals as figures
model/eda.py 					Data cleaning and EDA analysis on base dataset
model/train.py					Runs 5 Classifier models on training set and builds         pipelines
model/evaluate.py				Runs the evaluation score metric for each model 
model/plotting.py				Helper class for figure storage
model/scoring.py				Creates the scoring metrics function
model/meta_data.py				Contains variables and re-use functions
model/*.joblib, run_meta.json and other files
Requirements.txt, README.md, run_app.bat & test_data.csv Application support and test document

**Streamlit App:** https://socialmedia-lifeimpact-classifier.streamlit.app/ 

**Models Used:**

| ML Model Name  | Accuracy  | AUC  | Precision  | Recall  | F1  | MCC  |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Logistic Regression**  | .9844 | .9982 | .9556 | .9543 | .9548 | .9500 |
| **Decision Tree**  | .9722 | .9561 | .9134 | .9518 | .9311 | .9112 |
| **kNN**  | .9278 | .9692 | .8663 | .8097 | .8359 | .7563 |
| **Naive Bayes**  | .9300 | .9870 | .8311 | .9187 | .8697 | .7983 |
| **Support Vector Machine** | .9878 | .9985 | .9708 | .9674 | .9690 | .9606 |

**Hyperparameters chosen:**
* Logistic Regression: {'C': 10}
* Decision Tree: {'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 1}
* kNN: {'n_neighbors': 5, 'p': 2, 'weights': 'distance'}
* Naive Bayes: {'var_smoothing': 0.1}
* Support Vector Machine: {'C': 10, 'kernel': 'linear'}

**| ML Model Name  | Observation about model performance  |**
| :---- | :---- |
| **Logistic Regression**  | Achieves excellent performance with 98.44% accuracy and 99.82% AUC. Its precision, recall and F1-score are consistently high, indicating reliable classification across the three categories. It is a strong and balanced model, performing very close to SVM.  |
| **Decision Tree**  | Achieves 97.22% accuracy with good recall (95.18%), demonstrating its ability to identify different student well-being categories. However, its lower AUC (96.51%) and MCC (91.12%) indicate comparatively weaker overall predictive performance. Its unrestricted depth and minimum leaf size of one also make overfitting worth checking.   |
| **kNN**  | Achieves 92.78% accuracy but has comparatively lower recall (80.97%) and F1-score (83.59%). Its performance suggests difficulty consistently identifying all three well-being categories. Distance-based classification may be affected by the mixed feature types and high-dimensional encoded features.  |
| **Naive Bayes**  | Achieves a relatively high AUC of 98.70%, indicating good class-separation capability. However, its lowest precision (83.11%) and moderate F1-score (86.97%) suggest comparatively weaker prediction reliability. Its independence assumption may limit performance when lifestyle and well-being features are correlated.  |
| **Support Vector Machine**  | Delivers the best overall performance, achieving 98.78% accuracy and 99.85% AUC. It also records the highest precision (97.08%), recall (96.74%), F1-score (96.90%) and MCC (96.06%). Its consistently strong results make it the most reliable overall classifier among the five evaluated models. |
| **Overall, Winner for your dataset?**  | Support Vector Machine (SVM) is the best-performing model, securing the highest scores across all six evaluation metrics. It demonstrates excellent classification accuracy, balanced predictive performance and strong agreement between actual and predicted classes. Therefore, SVM is selected as the final model for predicting the impact of social media on student well-being.  |