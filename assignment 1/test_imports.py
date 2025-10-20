# Test script to verify all imports work
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import ttest_ind, chi2_contingency, spearmanr, mannwhitneyu
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve, 
                           precision_recall_curve, average_precision_score, f1_score, 
                           precision_score, recall_score, accuracy_score)
from sklearn.feature_selection import (mutual_info_classif, SelectKBest, f_classif, 
                                     RFE, SelectFromModel, chi2)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                            ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (train_test_split, cross_val_score, GridSearchCV, 
                                   StratifiedKFold, learning_curve, validation_curve)
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import shap
import lime
import lime.lime_tabular
import optuna
from optuna.integration import LightGBMPruningCallback
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import joblib
import pickle
from datetime import datetime
import json

# Suppress warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 140)
pd.set_option("display.max_rows", 100)

# Set plotting parameters
plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["axes.grid"] = True
# Use default matplotlib style instead of seaborn to avoid compatibility issues
plt.style.use('default')
sns.set_palette("husl")

# Global constants
TARGET = "Churn"
RANDOM_STATE = 42
CV_FOLDS = 5

print("Advanced libraries imported successfully")
print("Display options configured")
print("Target variable:", TARGET)
print("Cross-validation folds:", CV_FOLDS)
print("Analysis started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
