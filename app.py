import streamlit as st 
import joblib
import pandas as pd  

model = joblib.load("house_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")

st.sidebar.title("🏠 House Price Predictor")

st.sidebar.info(
    """
    This app predicts house prices using a Machine Learning model.
    Enter the house details and click **Predict House Price**.
    """
)

st.title("🏠 AI House Price Prediction")

st.markdown("### Predict house prices using Machine Learning")

st.divider()
st.markdown("## 🏡 Property Details")


overall_qual = st.slider("Overall Quality", 1, 10, 5)
grlivarea = st.number_input("Living Area (sq ft)", value=1500)
garage_cars = st.number_input("Garage Cars", value=2)
year_built = st.number_input("Year Built", value=2000)

new_house = pd.DataFrame(0, index=[0], columns=model_columns)

new_house["OverallQual"] = overall_qual
new_house["GrLivArea"] = grlivarea
new_house["GarageCars"] = garage_cars
new_house["YearBuilt"] = year_built

if st.button("Predict House Price"):
    st.markdown("## 📊 House Summary")
    st.markdown(f"⭐ **Overall Quality:** {overall_qual}/10")
    st.markdown(f"🏠 **Living Area:** {grlivarea:,} sq ft")
    st.markdown(f"🚗 **Garage Cars:** {garage_cars}")
    st.markdown(f"📅 **Year Built:** { year_built}")
    st.divider()
    predicted_price = model.predict(new_house)
    st.markdown("## 💰 Estimated House Price")
    st.success(f"${predicted_price[0]:,.2f}")
    st.balloons()

st.divider()
st.caption("Made with ❤️ by Kaustabhjyoti Baishya")