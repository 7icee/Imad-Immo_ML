# Immo_ML

## Overview
Immo_ML is a machine learning project aimed at predicting real estate prices based on data scraped from Immoweb/ The project involves data preprocessing, feature selection, and model evaluation to provide insights into the housing market.

## Addition of Additional Information through Statbel.com

- **Connection Creation:** 
  - Established a connection between the INS code (fiscal code) and postal code enabling the integration of additional data from "Statbel.be".
  
- **Data Integration:** 
  - Enriched the dataset by incorporating additional features based on the connection with "Statbel.be".

## Cleaning of the Dataset

- **Irrelevant Columns Removal:** 
  - Dropped columns that were not pertinent to the analysis.

- **Handling Missing Values:** 
  - Filled missing values for `terraceSurface` with 0 to indicate the absence of a terrace, while missing values in `facade` were filled with the mode to maintain the most common value in the dataset.

- **Row Deletion for Excessive Missing Data:** 
  - Removed rows that contained too many missing or critical values to enhance the quality of the dataset.

- **Transformation of Qualitative Columns:** 
  - Converted qualitative features, such as `buildingState`, into quantitative values.

## Final Dataset
- **Size:** 14,655 rows and 13 columns
- **Features to Predict Price:**
  - **Primary Features:**
    - Bedrooms
    - Property Type
    - Facades
    - Terrace
    - Terrace Surface
    - Building State
    - Garden
    - Garden Surface
    - Pool
    - Living Area
    - Surface of the Plot

  These features were collected from the Immoweb website scraping. The values have been cleaned, standardized, and processed for machine learning usage.

  - **Additional Features:**
    - Wealth Index
    - Density

  These additional features were sourced from Statbel (https://statbel.fgov.be/fr) and represent the economic situation and the density of municipalities, providing more insights for the analysis.

## Model Type
- **Model Used:** Linear Regression
- **Model Parameters:** ("copy_X": True, "fit_intercept": True, "n_jobs": None, "positive": False)
## Steps Taken to Improve Model Results

1. **Outlier Removal:** 
   - Outliers were removed using the Interquartile Range (IQR) method on the "price" and "livingArea" columns.

2. **Feature Standardization:** 
   - Standardization was applied to ensure that each feature contributes equally to the model's performance.

3. **Clustering:** 
   - KMeans clustering was used to segment the dataset into two groups, which helps enhance model performance.

## Results Metrics

- **Mean Absolute Error (MAE):**
  - **Training:** 67,081.71
  - **Test:** 64,599.81

- **Root Mean Squared Error (RMSE):**
  - **Training:** 86,578.45
  - **Test:** 82,388.62

- **R² Score:**
  - **Training:** 0.75
  - **Test:** 0.76

- **Mean Absolute Percentage Error (MAPE):**
  - **Training:** 21.84%
  - **Test:** 21.45%

- **Symmetric Mean Absolute Percentage Error (sMAPE):**
  - **Training:** 19.66%
  - **Test:** 19.47%

- **Cross-Validation Scores:** 
  - **Scores:** [0.76398412, 0.7226379, 0.71377983, 0.74481101, 0.739905]
  - **Mean CV Score:** 0.7370
