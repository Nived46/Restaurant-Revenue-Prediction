# Restaurant-Revenue-Prediction
This project predicts restaurant revenue using machine learning, based on a Kaggle competition. The Python notebook covers a full data science workflow: feature engineering, hyperparameter tuning (RandomizedSearchCV), and training multiple models (Random Forest, SVR, MLP). A 2-level stacked ensemble model to improve predictive accuracy.
The project follows a structured machine learning workflow:
Data Preprocessing: Loading the train.csv and test.csv datasets.
Exploratory Data Analysis (EDA): Checking for null values and understanding data distributions.
Feature Engineering:
Extracting year and month from the Open Date column.
Dropping duplicate features using feature_engine.selection.DropDuplicateFeatures.
Dropping highly correlated features (Pearson threshold > 0.8) using feature_engine.selection.DropCorrelatedFeatures.
Applying One-Hot Encoding to categorical variables: City, City Group, and Type.
Scaling all numerical features using StandardScaler.
Model Training (Base Models): Developing and training several individual regression models.
Hyperparameter Tuning: Using RandomizedSearchCV and GridSearchCV  to find the optimal parameters for each model.
Ensemble Modeling (Stacking):  Combining the base models into a stacked ensemble to improve predictive performance.

💾 Dataset
The dataset provides obfuscated data related to restaurant locations. Key fields include:
Id: A unique identifier for each restaurant.
Open Date: The date the restaurant opened.
City: The city where the restaurant is located.
City Group: A label for the city type (e.g., "Big Cities" or "Other").
Type: The type of restaurant (e.g., "FC" for Food Court, "IL" for Inline, "DT" for Drive Thru).
P1 - P37: Obfuscated data points representing demographic, real estate, and commercial information.
Revenue: The target variable, representing the (transformed) annual revenue.

🤖 Models Implemented
Both individual models and a stacked ensemble were built and evaluated.
1. Base Models
A variety of regression algorithms were tuned and trained as baseline predictors:

Support Vector Regressor (LinearSVR) 
Decision Tree Regressor 
Random Forest Regressor 
MLP Regressor (Multi-Layer Perceptron)
SGD Regressor (Stochastic Gradient Descent)

2. Stacking Ensemble Model
To improve performance, a 2-level stacking model was implemented using the vecstack library.
Level 0 (Base Models): The predictions from the following models were used as features for the next level:
SGDRegressor
RandomForestRegressor 
DecisionTreeRegressor
MLPRegressor
LinearSVR 
Level 1 (Meta-Model): Several models were tested as the final meta-regressor to combine the base model predictions:
LinearSVR 
DecisionTreeRegressor 
RandomForestRegressor 

💡 Key Insights
Based on the model development and submissions:
Ensemble Power: The stacked ensemble models consistently outperformed individual base models, demonstrating the value of combining diverse predictors.

Tuning Matters: Hyperparameter optimization using RandomizedSearchCV and GridSearchCV was crucial for maximizing the performance of every model.
Feature Engineering is Key: Preprocessing steps like one-hot encoding, scaling, and removing correlated features were essential for building an effective model.

🚀 Future Improvements
As outlined in the project summary, further enhancements could include:

External Data Integration: Incorporating external datasets like local economic indicators or more granular demographic data.

Advanced Ensembling: Exploring more advanced ensembling techniques, such as deep learning-based stacking methods.

Temporal Modeling: Implementing time series analysis to capture any seasonal trends or revenue changes over time.
