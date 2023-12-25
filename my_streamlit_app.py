
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

# Load data
nba_data_all = pd.read_csv("nba_2022-23_all_stats_with_salary3.csv")

# Select relevant columns
nba_data7 = nba_data_all[["Salary", "PTS", "AST", "TRB", "Age"]]

# Split data
seed = 123
train_data7, test_data7 = train_test_split(nba_data7, test_size=0.2, random_state=seed)

# Preprocess
preProcValues7 = StandardScaler().fit(train_data7[["PTS", "AST", "TRB", "Age"]])
train_data_scaled7 = preProcValues7.transform(train_data7[["PTS", "AST", "TRB", "Age"]])
test_data_scaled7 = preProcValues7.transform(test_data7[["PTS", "AST", "TRB", "Age"]])

# Build Random Forest model
rf_model7 = RandomForestRegressor()
rf_model7.fit(train_data_scaled7, train_data7["Salary"])

# Streamlit app
st.title("NBA Player Salary Prediction (2022-23)")

# User input
st.header("User Input Features")
points = st.number_input("Points:", value=20)
assists = st.number_input("Assists:", value=5)
rebounds = st.number_input("Rebounds:", value=5)
age = st.number_input("Age:", value=30)

# Create a DataFrame with named features for the user input
user_input_df = pd.DataFrame({
    'PTS': [points],
    'AST': [assists],
    'TRB': [rebounds],
    'Age': [age]
})

# Preprocess user input using the same scaler
new_data_scaled = preProcValues7.transform(user_input_df)

# Predict salary using the Random Forest model
predicted_salary = rf_model7.predict(new_data_scaled)


# Display predicted salary with commas
formatted_salary = f"${predicted_salary[0]:,.2f}"
st.subheader("Predicted Salary")
st.write(f"The predicted NBA salary is: {formatted_salary}")

