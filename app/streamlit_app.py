import streamlit as st
import random
import matplotlib.pyplot as plt

#  Dummy prediction function 
def dummy_predict(text):
    # Simulate model logic: random result
    is_viral = random.choice([True, False])
    confidence = round(random.uniform(0.5, 1.0), 2)
    emotion = random.choice(["Positive", "Negative", "Neutral"])
    return is_viral, confidence, emotion

# Streamlit interface 
st.title(" VK Post Virality Predictor")
st.write("Enter the text of a VK post. Since the model is still in development, a demo placeholder is used.")

user_input = st.text_area("Post text:", height=200)

if st.button("Predict"):
    if not user_input.strip():
        st.error(" Please enter text for analysis!")
    else:
        st.info(" Predicting (demo placeholder)...")
        is_viral, confidence, emotion = dummy_predict(user_input)

        if is_viral:
            st.success(f" This post is **LIKELY VIRAL** (confidence: {confidence * 100:.1f}%)")
        else:
            st.warning(f" This post is **LIKELY NOT VIRAL** (confidence: {confidence * 100:.1f}%)")

        st.write(f" Emotional tone: **{emotion}**")


    #  Display confidence as a pie chart 
    fig, ax = plt.subplots()
    ax.pie([confidence, 1 - confidence],
        labels=[f"Confidence {confidence*100:.1f}%", "Remaining"],
        startangle=90,
        autopct='%.1f%%',
        colors=['#4CAF50', '#E0E0E0'])
    ax.axis('equal')
    st.pyplot(fig)

    # Highlight emotional tone st
    if emotion == "Positive":
        st.markdown(f"<div style='padding:10px;background-color:#C8E6C9;border-radius:10px;text-align:center;font-size:18px;'>Emotional tone: <b>{emotion}</b></div>", unsafe_allow_html=True)
    elif emotion == "Negative":
        st.markdown(f"<div style='padding:10px;background-color:#FFCDD2;border-radius:10px;text-align:center;font-size:18px;'>Emotional tone: <b>{emotion}</b></div>", unsafe_allow_html=True)
    else:  # Neutral
        st.markdown(f"<div style='padding:10px;background-color:#F0F4C3;border-radius:10px;text-align:center;font-size:18px;'>Emotional tone: <b>{emotion}</b></div>", unsafe_allow_html=True)
