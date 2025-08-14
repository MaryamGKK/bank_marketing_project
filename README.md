# Bank Marketing Campaign Planner

### Project Description
This project focuses on planning a direct marketing campaign. The primary goal is to optimize a call-based campaign to maximize the number of clients who subscribe to a term deposit. Given limited call center resources, the core task is to develop a predictive model that ranks potential customers by their likelihood of conversion, thereby enabling a data-driven call prioritization strategy.

### Objectives
The case study is divided into two main parts:

1.  **Predictive Modelling:** Build a machine learning model to predict the outcome of a marketing call—whether a client will agree to open a deposit account (`y`). The model is a LightGBM classifier, optimized using automated hyperparameter tuning.

2.  **Supporting Marketing Campaign Planning:**
    * **Call Prioritization:** Suggest a rule for calling potential customers to maximize the efficiency of the call center.
    * **Simulation & Visualization:** Use the trained model to simulate and visualize the growth of new contracts as the number of calls increases.
    * **Target Estimation:** Estimate the number of calls required to secure a specific target number of new contracts (e.g., 300).

### Dataset
The project utilizes the **UCI Bank Marketing Dataset**, which contains information about a previous direct marketing campaign. The dataset includes customer demographics, details of the last contact, and social/economic indicators. A key feature, `duration`, is dropped from the model to prevent data leakage, as it's only known after a call is completed.

### Project Structure
The project is organized into a series of Jupyter notebooks and a Streamlit application, detailing the end-to-end machine learning lifecycle.

* `1_data_understanding_eda.ipynb`: Contains the initial data exploration, including descriptive statistics, value counts, and visualizations to build an understanding of the dataset.
* `2_data_preprocessing.ipynb`: Focuses on cleaning the raw data, handling missing values, performing feature engineering, and fitting a `ColumnTransformer` pipeline. This notebook saves both the cleaned data and the fitted preprocessor for consistency.
* `3_model_training_evaluation.ipynb`: Loads the preprocessed data and the pipeline to train a robust LightGBM model. It includes automated hyperparameter tuning with Optuna and a detailed evaluation of the model's performance, including feature importance.
* `4_campaign_simulation.ipynb`: Applies the final, trained model to the full dataset to simulate the call campaign, generating visualizations and key performance metrics to guide decision-making.
* `app.py`: A user-friendly Streamlit web application that allows you to upload a new customer list, get predictions, and run campaign simulations interactively.
* `requirements.txt`: Lists all the necessary Python dependencies to run the project.

### How to Run the Project
To run this project on your local machine, follow these steps:

1.  **Clone the repository** (if applicable) or download all the project files.
2.  **Install dependencies** by running the following command in your terminal:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Execute the Jupyter notebooks** (`1_data_understanding_eda.ipynb` through `4_campaign_simulation.ipynb`) in sequential order to generate the required model and preprocessor files (`preprocessor.pkl` and `best_model.pkl`).
4.  **Run the Streamlit application** from your terminal:
    ```bash
    python3 -m  streamlit run app_streamlit.py
    ```
    This will launch the web application in your browser, where you can upload a new CSV file and use the predictive model for campaign planning.