import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. LOAD THE TRAINED MODEL ---
try:
    with open('best_model.pk1', 'rb') as f:
        best_model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'best_model.pk1' is in the same directory.")
    st.stop()

# --- 2. MANUALLY DEFINE MAPPINGS AND QUESTIONS ---
MANUAL_MAPPINGS = {
    'gender': {'f': 0, 'm': 1},
    'jaundice': {'no': 0, 'yes': 1},
    'austim': {'no': 0, 'yes': 1},
    'used_app_before': {'no': 0, 'yes': 1},
    'relation': {'Others': 0, 'Self': 1},
    'ethnicity': {
        'Asian': 0, 'Black': 1, 'Hispanic': 2, 'Latino': 3,
        'Middle Eastern ': 4, 'Others': 5, 'Pasifika': 6,
        'South Asian': 7, 'Turkish': 8, 'White-European': 9
    },
    'contry_of_res': {
        'Afghanistan': 0, 'Angola': 1, 'Argentina': 2, 'Armenia': 3, 'Aruba': 4,
        'Australia': 5, 'Austria': 6, 'Azerbaijan': 7, 'Bahamas': 8, 'Bangladesh': 9,
        'Belgium': 10, 'Bolivia': 11, 'Brazil': 12, 'Canada': 13, 'China': 14,
        'Costa Rica': 15, 'Cyprus': 16, 'Czech Republic': 17, 'Ecuador': 18,
        'Egypt': 19, 'Ethiopia': 20, 'Finland': 21, 'France': 22, 'Germany': 23,
        'Iceland': 24, 'India': 25, 'Indonesia': 26, 'Iran': 27, 'Iraq': 28,
        'Ireland': 29, 'Italy': 30, 'Japan': 31, 'Jordan': 32, 'Kazakhstan': 33,
        'Malaysia': 34, 'Mexico': 35, 'Netherlands': 36, 'New Zealand': 37,
        'Nicaragua': 38, 'Niger': 39, 'Oman': 40, 'Pakistan': 41, 'Philippines': 42,
        'Romania': 43, 'Russia': 44, 'Saudi Arabia': 45, 'Serbia': 46,
        'Sierra Leone': 47, 'South Africa': 48, 'Spain': 49, 'Sri Lanka': 50,
        'Sweden': 51, 'Tonga': 52, 'Ukraine': 53, 'United Arab Emirates': 54,
        'United Kingdom': 55, 'United States': 56, 'Uruguay': 57, 'Vietnam': 58
    }
}

FEATURE_ORDER = [
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
    'age', 'gender', 'ethnicity', 'jaundice', 'austim',
    'contry_of_res', 'used_app_before', 'result', 'relation'
]

# The actual AQ-10 questions for a better user experience
AQ_QUESTIONS = {
    'A1_Score': "1. I often notice small sounds when others do not.",
    'A2_Score': "2. I usually concentrate more on the whole picture, rather than the small details.",
    'A3_Score': "3. I find it easy to do more than one thing at once.",
    'A4_Score': "4. If there is an interruption, I can switch back to what I was doing very quickly.",
    'A5_Score': "5. I find it easy to ‘read between the lines’ when someone is talking to me.",
    'A6_Score': "6. I know how to tell if someone listening to me is getting bored.",
    'A7_Score': "7. When I’m reading a story, I find it difficult to work out the characters’ intentions.",
    'A8_Score': "8. I like to collect information about categories of things (e.g., types of cars, birds, trains).",
    'A9_Score': "9. I find it easy to work out what someone is thinking or feeling just by looking at their face.",
    'A10_Score': "10. I find it difficult to work out people’s intentions."
}

# User-friendly options for the UI dropdowns
RELATION_OPTIONS = ['Self', 'Parent', 'Relative', 'Health care professional', 'Others']
ETHNICITY_OPTIONS = list(MANUAL_MAPPINGS['ethnicity'].keys())
COUNTRY_OPTIONS = list(MANUAL_MAPPINGS['contry_of_res'].keys())

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="ASD Prediction", layout="wide")
st.title("🧩 Autism Spectrum Disorder (ASD) Prediction")
st.write(
    "This app predicts the likelihood of having ASD based on the AQ-10 screening test and demographic data. Please fill in the information below.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Screening Questions (AQ-10)")

    # FIX: Add an expander to explain the reverse-scoring to the user
    with st.expander("ℹ️ How is the score calculated?"):
        st.markdown("""
                This test uses **reverse scoring** for some questions.
                - A score of **1** is given for an **Agree** response to questions **1, 7, 8, and 10**.
                - A score of **1** is given for a **Disagree** response to questions **2, 3, 4, 5, 6, and 9**.
                This is why the total score might decrease when you select "Agree" for certain questions.
            """)
    a_scores_responses = {}
    options = ['Disagree', 'Agree']
    # Define which questions are reverse-scored (1 point for 'Disagree')
    REVERSE_SCORED_QUESTIONS = [2, 3, 4, 5, 6, 9]

    for i in range(1, 11):
        feature_name = f'A{i}_Score'
        # FIX: Display the full question text as the label for the radio button
        response = st.radio(AQ_QUESTIONS[feature_name], options, key=feature_name, horizontal=True)

        # FIX: Implement the correct AQ-10 reverse scoring logic
        if i in REVERSE_SCORED_QUESTIONS:
            a_scores_responses[feature_name] = 1 if response == 'Disagree' else 0
        else:
            a_scores_responses[feature_name] = 1 if response == 'Agree' else 0

    # FIX: Automatically calculate the result score
    calculated_result = float(sum(a_scores_responses.values()))
    result = st.number_input(
        'Screening Result Score (Auto-calculated)',
        value=calculated_result,
        disabled=True,
        help="This score is the sum of the answers above, applying clinical reverse-scoring rules."
    )

with col2:
    st.subheader("Demographic & Background Information")
    age = st.number_input('Age', min_value=4, max_value=100, value=25)
    gender = st.selectbox('Gender', ['f', 'm'])
    ethnicity = st.selectbox('Ethnicity', ETHNICITY_OPTIONS)
    jaundice = st.selectbox('Born with Jaundice?', ['no', 'yes'])
    autism_in_family = st.selectbox('Family member with Autism?', ['no', 'yes'])
    contry_of_res = st.selectbox('Country of Residence', COUNTRY_OPTIONS)
    used_app_before = st.selectbox('Used a screening app before?', ['no', 'yes'])
    relation = st.selectbox('Relation to the person being screened', RELATION_OPTIONS)

# --- 4. PREDICTION LOGIC ---
if st.button('Predict Likelihood of ASD', type="primary"):

    # Preprocessing of input data
    age_processed = 25 if age > 54 else age
    relation_map = {'Parent': 'Others', 'Relative': 'Others', 'Health care professional': 'Others'}
    relation_processed = relation_map.get(relation, relation)

    input_data = {
        **a_scores_responses,
        'age': age_processed, 'gender': gender, 'ethnicity': ethnicity,
        'jaundice': jaundice, 'austim': autism_in_family, 'contry_of_res': contry_of_res,
        'used_app_before': used_app_before, 'result': result, 'relation': relation_processed
    }

    final_data = {}
    for feature, value in input_data.items():
        if feature in MANUAL_MAPPINGS:
            final_data[feature] = MANUAL_MAPPINGS[feature][value]
        else:
            final_data[feature] = value

    input_list = [final_data[feature] for feature in FEATURE_ORDER]
    input_array = np.array(input_list).reshape(1, -1)

    prediction = best_model.predict(input_array)
    prediction_proba = best_model.predict_proba(input_array)

    # --- 5. DISPLAY RESULTS ---
    st.markdown("---")
    st.subheader("Prediction Result")
    asd_probability = prediction_proba[0][1]

    if prediction[0] == 1:
        st.error(f"Prediction: **High Likelihood of ASD** (Probability: {asd_probability * 100:.2f}%)")
        st.warning(
            "Disclaimer: This is a prediction based on a machine learning model and is not a clinical diagnosis.")
    else:
        st.success(
            f"Prediction: **Low Likelihood of ASD** (The model estimates a {asd_probability * 100:.2f}% probability of ASD)")
        st.info("Disclaimer: This result is not a substitute for a professional medical evaluation.")