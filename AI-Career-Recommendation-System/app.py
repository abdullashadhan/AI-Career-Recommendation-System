import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Career Recommendation System",
    page_icon="",
    layout="centered"
)

st.markdown("<h1 style='text-align: center;'>AI Career Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Find the best career based on your skills and interests</p>", unsafe_allow_html=True)
st.divider()

data = pd.read_csv("career_data.csv")
data["profile"] = data["skills"] + " " + data["interests"]
data.dropna(inplace=True)

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(data["profile"])
y = data["career"]

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)
model.fit(X, y)

st.subheader(" Enter Your Skills & Interests")
user_input = st.text_area(
    "",
    placeholder="Example: python machine learning data analysis",
    height=120
)
recommended_career = None
if st.button(" Recommend Career"):
    if user_input.strip() == "":
        st.warning(" Please enter your skills and interests")
    else:
        user_vector = vectorizer.transform([user_input])
        prediction = model.predict(user_vector)[0]
        recommended_career = prediction

        st.success(f" Recommended Career: **{prediction}**")

st.divider()

fresh_projects = [
    {
        "title": "Smart Study Planner",
        "description": "Build a planner that adapts schedules based on goals and focus time.",
        "tags": ["Python", "Streamlit", "Productivity"]
    },
    {
        "title": "AI Resume Feedback Tool",
        "description": "Analyze resumes and suggest improvements using NLP.",
        "tags": ["NLP", "Data", "Career"]
    },
    {
        "title": "Personal Finance Dashboard",
        "description": "Track expenses and visualize spending trends with insights.",
        "tags": ["Data Viz", "Analytics", "SQL"]
    },
    {
        "title": "IoT Home Energy Monitor",
        "description": "Collect sensor data and display live energy usage.",
        "tags": ["IoT", "Sensors", "Realtime"]
    },
    {
        "title": "Portfolio Project Showcase",
        "description": "Create a responsive site to highlight your best work.",
        "tags": ["Frontend", "UI", "Web"]
    }
]

st.subheader(" New & Fresh Projects")
st.caption("Kickstart your portfolio with these ideas.")
for project in fresh_projects:
    tag_list = ", ".join(project["tags"])
    st.markdown(f"**{project['title']}** — {project['description']}")
    st.markdown(
        f"<span style='font-size:12px; color:gray;'>Tags: {tag_list}</span>",
        unsafe_allow_html=True
    )

st.divider()

st.markdown(
    """
    <div style="text-align:center; font-size:14px; color:gray;">
        Developed by <b>Abdulla Shadhan S</b> | AI-Powered Career Recommendation System
    </div>
    """,
    unsafe_allow_html=True
)
