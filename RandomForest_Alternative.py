from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

# Update in the preprocessor:
relevant_feat_pass = relevant_Feat.copy()
relevant_feat_pass.remove('State_Risk_Rating')
relevant_feat_pass.remove('Population_Density')

preprocessor = ColumnTransformer([
    ('state_risk', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ['State_Risk_Rating']),
    ('population', OrdinalEncoder(categories=[['Village', 'Small_Town', 'Town', 'City', 'Metropolitan_Area']], 
                                   handle_unknown='use_encoded_value', unknown_value=-1), ['Population_Density']),
    ('pass', 'passthrough', relevant_feat_pass),
])

# Random Forest Pipeline
pipeline_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ))
])

# Hyperparameter distribution for Random Forest
hiperparam_distributions_rf = {
    'classifier__n_estimators': [50, 100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30, 50],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2', None],
    'classifier__class_weight': [None, 'balanced']
}

# Randomized Search for Random Forest
randomized_search_rf = RandomizedSearchCV(
    pipeline_rf,
    hiperparam_distributions_rf,
    n_iter=20,  # More iterations for Random Forest
    cv=5,
    scoring='roc_auc',  # Or use dict for multiple metrics
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# Fit Random Forest
randomized_search_rf.fit(X_train, y_train)

# Best Random Forest model
best_rf = randomized_search_rf.best_estimator_

# Print results
print("Random Forest Best params:", randomized_search_rf.best_params_)
print("Random Forest Best score:", randomized_search_rf.best_score_)