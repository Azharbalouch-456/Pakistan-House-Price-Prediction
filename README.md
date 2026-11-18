# 🏠 Pakistan House Price Prediction (Machine Learning & Web App)

An end-to-end Machine Learning regression project and interactive web application that predicts residential property prices across major cities in Pakistan using real estate listing data.

## 📌 Project Overview
Real estate pricing in Pakistan depends heavily on location, property size, and structure. In this project, I cleaned and analyzed a dataset of **168,000+ property listings**, trained and compared regression models achieving **91.8% accuracy**, and deployed an interactive **Gradio Web App** for real-time price estimation.

## ⚙️ Step-by-Step Workflow
1. **Data Cleaning & Outlier Removal:**
   - Dropped columns with excessive missing values (`agency`, `agent`) and non-predictive metadata (`property_id`, `page_url`, `date_added`).
   - Filtered the dataset to include only `For Sale` residential listings.
   - Removed unrealistic outliers in `price`, `Total_Area`, and `baths` using domain filtering and quantile bounds (2%–98%).
2. **Feature Engineering & Encoding:**
   - Extracted the **Top 30 prime locations** and grouped the remaining areas into `Others` to prevent high dimensionality.
   - Applied **One-Hot Encoding** (`pd.get_dummies`) to convert categorical features (`property_type`, `city`, `province_name`, `location`) into numerical format.
   - Applied **Log Transformation** (`np.log1p`) to `price` and `Total_Area` to normalize heavily skewed distributions.
3. **Model Training & Evaluation:**
   - Split the cleaned dataset into **80% training** and **20% testing** sets.
   - Trained a baseline **Linear Regression** model and upgraded to a **Random Forest Regressor** for higher accuracy.
4. **Interactive Web App (Gradio) & Serialization:**
   - Saved the trained Random Forest model using `joblib`.
   - Built a **Gradio Web Interface** displaying the model's **91.8% R² accuracy** and allowing users to select City, Location, Property Type, Bedrooms, Bathrooms, and Area (in Marla) to get instant price predictions in PKR.

## 📊 Model Performance Progression

| Stage / Model | R² Score (Accuracy) | MSE (Log Scale) |
| :--- | :---: | :---: |
| Baseline Linear Regression (Raw) | `0.178` (17.8%) | High |
| Linear Regression + Log Transform | `0.401` (40.1%) | `0.427` |
| Linear Regression + Top 30 Locations | `0.557` (55.7%) | `0.316` |
| **Random Forest Regressor (Final)** | **`0.918` (91.8%)** | **`0.058`** |

## 🛠️ Technologies & Libraries Used
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (`LinearRegression`, `RandomForestRegressor`, `train_test_split`, `r2_score`, `mean_squared_error`)
- **Web Interface & Deployment:** Gradio, Joblib
- **Environment:** Google Colab

## 🚀 How to Run
1. Click **Open in Colab** at the top of the notebook file (`.ipynb`) or open it in Jupyter Notebook.
2. Download the **Pakistan House Price Prediction** dataset (`Entities.csv`) from Kaggle and update the file path in the data-loading cell.
3. Run all cells sequentially to train the models and launch the **Gradio Live Web App** interface.
