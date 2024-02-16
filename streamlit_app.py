import streamlit as st
import pandas as pd
import joblib
import sklearn
import category_encoders
import plotly.express as px

Model = joblib.load("Model.pkl")
Inputs = joblib.load("Inputs.pkl")
df = joblib.load("data.pkl")

def Make_Prediction(Gender, Married, Dependents, Education, Self_Employed,LoanAmount, Loan_Amount_Term, Credit_History, Property_Area):
    Pr_df = pd.DataFrame(columns=Inputs)
    Pr_df.at[0,"Gender"] = Gender
    Pr_df.at[0,"Married"] = Married
    Pr_df.at[0,"Dependents"] = Dependents
    Pr_df.at[0,"Education"] = Education
    Pr_df.at[0,"Self_Employed"] = Self_Employed
    Pr_df.at[0,"LoanAmount"] = LoanAmount
    Pr_df.at[0,"Loan_Amount_Term"] = Loan_Amount_Term
    Pr_df.at[0,"Credit_History"] = Credit_History
    Pr_df.at[0,"Property_Area"] = Property_Area
    
    result_proba = Model.predict_proba(Pr_df)[0][1] * 100
    return result_proba
    
def main():
    st.title("Zomato Restaurants")
    with st.sidebar:
        Gender = st.selectbox("Gender", ['Male', 'Female']) 
        Married = st.selectbox("Married", ['Yes', 'No'])
        Dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])
        Education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
        Self_Employed = st.selectbox("Self_Employed", ['No', 'Yes'])
        Property_Area = st.selectbox("Property_Area", ['Urban', 'Rural', 'Semiurban'])
        Loan_Amount_Term = st.slider("Loan_Amount_Term", min_value=1, max_value=1000, value=1, step=1)
        Credit_History = st.slider("Credit_History", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        LoanAmount = st.slider("LoanAmount", min_value=1, max_value=1000, value=1, step=1)

    if st.button("Predict"):
        Results = Make_Prediction(Gender, Married, Dependents, Education, Self_Employed, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area)
        list_success = ["Your Restaurant May Fail", "Your Restaurant will succeed"]

        if 25 <= Results <= 50:
            st.text(f"Your loan may have challenges with a percentage: {int(Results)}% of success. You may reconsider changing some features.")
        elif 50 < Results <= 85:
            st.text(f"Your loan may succeed with a percentage: {int(Results)}% of success.")
        elif Results > 85:
            st.text(f"Your loan has great potential to succeed with a percentage of: {int(Results)}% of success.")
        else:
            st.text(f"Your loan has a less opportunity to succeed with a percentage: {int(Results)}%. You may reconsider changing some features.")

        fig = px.histogram(df, y="LoanAmount", x='Property_Area', color='Dependents')
        st.plotly_chart(fig)

main()

