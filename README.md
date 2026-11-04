# Socialmedia-lifeimpact-classifier
## Problem Statement: Impact of Social Media on Student Well-being
The objective of this project is to investigate the influence of social media usage on students' daily lives and overall well-being. Utilizing survey data collected from 4,500 students across various educational levels, the study aims to examine the relationships between digital engagement patterns, sleep habits, and stress levels.

Various Machine Learning classification algorithms will be implemented to predict students' well-being and stress categories based on their social media usage and lifestyle-related factors. Furthermore, the performance of the different classification models will be evaluated and compared using appropriate evaluation metrics to identify the most effective model in terms of predictive accuracy and reliability.

The study seeks to derive meaningful insights into the potential effects of digital engagement on student well-being while demonstrating the applicability of Machine Learning techniques in identifying behavioral patterns and predicting well-being outcomes.

## Dataset Description:
* Source: Kaggle - Impact of Social Media on Life (harishyadav0506), survey data on screen time, sleep, stress and GPA.
* Size: 4,500 rows x 16 columns -> 14 predictive features after removing `Student_ID` (identifier) and the target `Overall_Impact`.
* Feature types: 9 numeric (age, daily/weekend usage hours, sleep hours, sleep quality, late-night flag, stress score, mental-health index, GPA), 4 nominal (gender, academic level, primary platform, device type), 1 ordinal (social-comparison frequency: Never < Rarely < Sometimes < Frequently < Always).
* Target Column: `Overall_Impact` - Beneficial 3,681 (81.8%), Neutral 654 (14.5%), Negative 165 (3.7%).
* Missing values: 131 cells (46 in `Perceived_Stress_Score`, 85 in `Academic_Performance_GPA`), imputed with the median inside the pipeline.
* Split: one stratified 80/20 split (seed 42): 3,600 fit rows, 900 hold-out rows (736 Beneficial / 131 Neutral / 33 Negative). The hold-out rows are `test_data.csv`.
  
## GitHub Repository: https://github.com/PrachiiShah/Socialmedia-lifeimpact-classifier  

## Models Used:


ML Model Name 
	Accuracy 
	AUC 
	Precision 
	Recall 
	F1 
	MCC 
	Logistic Regression 
	

	

	

	

	

	

	Decision Tree 
	

	

	

	

	

	

	kNN 
	

	

	

	

	

	

	Naive Bayes 
	

	

	

	

	

	

	Support Vector Machine
	

	

	

	

	

	

	

## Streamlit App:
