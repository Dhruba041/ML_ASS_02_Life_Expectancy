import streamlit as st
import joblib
import numpy as np
import pickle
import pandas as pd

st.markdown(
    """
    <style>
    .stApp {
        background-color: #90EE90;
        color: brown;   /* default text color */
    }

    h1 {
        color: #1f77b4;  /* blue title */
    }

    label {
        color: #1f77b4 !important;
        font-weight: 600;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label span p {
        color: brown !important;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

oe = pickle.load(open("oe_life_expectancy.pkl", "rb"))          # OrdinalEncoder
pt = pickle.load(open("powertransformer_life_expectancy.pkl", "rb"))  # PowerTransformer
scaler = pickle.load(open("scaler_life_expectancy.pkl", "rb"))  # StandardScaler
model = pickle.load(open("best_model_xgb_life_expectancy.pkl", "rb")) # XGBoost model


st.title("Life Expectancy Prediction App")
col1, col2 = st.columns([2, 1])  # left bigger, right smaller

with col1:
    st.write("Enter the details below to predict life expectancy:")
    country = st.selectbox("Country", oe.categories_[0])
    year = st.slider(
    "Year",
    min_value=2000,
    max_value=2100,
    value=2026,
    step=1
    )
    #status = st.selectbox("Status", oe.categories_[1])  
    status = st.radio(
    "Status",
    oe.categories_[1]
    )
    #adult_mortality = st.number_input("Adult Mortality", min_value=0, value=100)
    adult_mortality = st.slider(
    "Adult Mortality",
    min_value=0,
    max_value=1000,   # set a reasonable upper limit
    value=100,
    step=1
    )
    infant_mortality = st.slider(
    "Infant Mortality",
    min_value=0,
    max_value=100,
    value=10,
    step=1
    )
    alcohol = st.number_input("Alcohol Consumption", min_value=0.0, value=5.0)
    #percentage_expenditure = st.number_input("Percentage Expenditure", min_value=0.0, value=5.0)
    percentage_expenditure = st.slider(
    "Percentage Expenditure",
    min_value=0.0,
    max_value=2000.0,
    value=50.0,
    step=1.0
    )
    hiv_aids = st.number_input("HIV/AIDS", min_value=0.0, value=0.1)
    #measles = st.number_input("Measles", min_value=0, value=100)
    measles = st.slider(
    "Measles",
    min_value=0,
    max_value=5000,  # adjust if needed based on your dataset
    value=100,
    step=1
)
    #bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    bmi = st.slider(
    "BMI",
    min_value=0.0,
    max_value=150.0,
    value=25.0,
    step=.5
    )
    #under_five_deaths = st.number_input("Under Five Deaths", min_value=0, value=10)
    under_five_deaths = st.slider(
    "Under Five Deaths",
    min_value=0,
    max_value=1000,
    value=10,
    step=1
    )
    #polio = st.number_input("Polio", min_value=0.0, value=100.0)
    polio = st.slider(
    "Polio",
    min_value=0,
    max_value=500,
    value=100,
    step=1
    )
    total_expenditure = st.number_input("Total Expenditure", min_value=0.0, value=5.0)
    #diphtheria = st.number_input("Diphtheria", min_value=0.0, value=100.0)
    diphtheria = st.slider(
    "Diphtheria",
    min_value=0,
    max_value=500,
    value=100,
    step=1
    )
    gdp = st.number_input("GDP", min_value=0.0, value=10000.0)
    population = st.number_input("Population", min_value=0.0, value=1000000.0)
    thinness_1_19_years = st.number_input("Thinness 1-19 Years", min_value=0.0, value=5.0)
    thinness_5_9_years = st.number_input("Thinness 5-9 Years", min_value=0.0, value=5.0)
    income_composition_of_resources = st.number_input("Income Composition of Resources", min_value=0.0, value=0.5)
    schooling = st.number_input("Schooling", min_value=0.0, value=10.0) 
    hepatitis_b = st.number_input("Hepatitis B", min_value=0.0, value=100.0)  # Added Hepatitis B input

    if st.button("Predict Life Expectancy"):

        input_data = pd.DataFrame({
            'country': [country],
            'year': [year],
            'status': [status],
            'adult_mortality': [adult_mortality],
            'infant_deaths': [infant_mortality],
            'alcohol': [alcohol],
            'percentage_expenditure': [percentage_expenditure],
            'hiv_aids': [hiv_aids],
            'measles': [measles],
            'bmi': [bmi],
            'under_five_deaths': [under_five_deaths],
            'polio': [polio],
            'total_expenditure': [total_expenditure],
            'diphtheria': [diphtheria],
            'gdp': [gdp],
            'population': [population],
            'thinness_1_19yrs': [thinness_1_19_years],
            'thinness_5_9yrs': [thinness_5_9_years],
            'income_composition': [income_composition_of_resources],
            'schooling': [schooling],
            'hepatitis_b': [hepatitis_b]
        })

        categorical_cols = ['country', 'status']

        input_data[categorical_cols] = oe.transform(
            input_data[categorical_cols]
        )

        skewed_cols = [
        'status',
        'adult_mortality',
        'infant_deaths',
        'alcohol',
        'percentage_expenditure',
        'hepatitis_b',
        'measles',
        'under_five_deaths',
        'polio',
        'diphtheria',
        'hiv_aids',
        'gdp',
        'population',
        'thinness_1_19yrs',
        'thinness_5_9yrs',
        'income_composition'
        ]

        input_data[skewed_cols] = pt.transform(
            input_data[skewed_cols]
        )

        feature_order = [
        'country',
        'year',
        'status',
        'adult_mortality',
        'infant_deaths',
        'alcohol',
        'percentage_expenditure',
        'hepatitis_b',
        'measles',
        'bmi',
        'under_five_deaths',
        'polio',
        'total_expenditure',
        'diphtheria',
        'hiv_aids',
        'gdp',
        'population',
        'thinness_1_19yrs',
        'thinness_5_9yrs',
        'income_composition',
        'schooling'
        ]

        input_data = input_data[feature_order]

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)


        #st.success(
            #f"Predicted Life Expectancy: {prediction[0]:.2f} years"
        #)
        st.markdown(
                f"""
                <div style="
                    background-color:#e8f0ff;
                    padding:15px;
                    border-radius:10px;
                    border:2px solid #0B3D91;
                    color:#0B3D91;
                    font-size:18px;
                    font-weight:bold;
                    text-align:center;
                ">
                    Predicted Life Expectancy: {prediction[0]:.2f} years
                </div>
                """,
                unsafe_allow_html=True
            )


with col2:
    st.image("life_exp_img.jpg", caption="Life Expectancy", use_container_width=True)