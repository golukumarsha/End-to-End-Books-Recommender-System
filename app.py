import os
import sys
import pickle
import streamlit as st
import numpy as np

from books_recommender.config.configuration import AppConfiguration
from books_recommender.pipeline.training_pipeline import TrainingPipeline
from books_recommender.exception.exception_handler import AppException


# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="📚 Books Recommender",
    page_icon="📖",
    layout="wide"
)

# -------------------------------
# Custom Styling
# -------------------------------
st.markdown("""
<style>
.big-title {
    font-size:40px !important;
    font-weight:700;
    color:#4CAF50;
}
.subtitle {
    font-size:18px;
    color:gray;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# Recommendation Class
# -------------------------------
class Recommendation:
    def __init__(self, app_config=AppConfiguration()):
        try:
            self.recommendation_config = app_config.get_recommendation_config()
        except Exception as e:
            raise AppException(e, sys)

    def load_pickle(self, path):
        if not os.path.exists(path):
            st.error(f"File not found: {path}")
            st.stop()
        return pickle.load(open(path, "rb"))

    def fetch_poster(self, suggestion):
        book_pivot = self.load_pickle(
            self.recommendation_config.book_pivot_serialized_objects
        )
        final_rating = self.load_pickle(
            self.recommendation_config.final_rating_serialized_objects
        )

        book_names = [book_pivot.index[i] for i in suggestion[0]]

        poster_urls = []
        for name in book_names:
            idx = np.where(final_rating["title"] == name)[0][0]
            poster_urls.append(final_rating.iloc[idx]["image_url"])

        return book_names, poster_urls

    def recommend_book(self, book_name):
        model = self.load_pickle(
            self.recommendation_config.trained_model_path
        )
        book_pivot = self.load_pickle(
            self.recommendation_config.book_pivot_serialized_objects
        )

        book_id = np.where(book_pivot.index == book_name)[0][0]
        distance, suggestion = model.kneighbors(
            book_pivot.iloc[book_id, :].values.reshape(1, -1),
            n_neighbors=6
        )

        return self.fetch_poster(suggestion)

    def train_engine(self):
        obj = TrainingPipeline()
        obj.start_training_pipeline()
        st.success("✅ Training Completed Successfully!")


# -------------------------------
# UI Layout
# -------------------------------

st.markdown('<p class="big-title">📚 Books Recommendation System</p>',
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">Collaborative Filtering Based ML Project</p>',
            unsafe_allow_html=True)

st.divider()

obj = Recommendation()

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🚀 Train Model"):
        obj.train_engine()

# Load Book Names
BASE_DIR = os.getcwd()
BOOK_PATH = os.path.join(BASE_DIR, "templates", "book_names.pkl")

if not os.path.exists(BOOK_PATH):
    st.warning("⚠️ Train the model first to generate book_names.pkl")
    st.stop()

book_names = pickle.load(open(BOOK_PATH, "rb"))

selected_book = st.selectbox(
    "🔎 Search or Select a Book",
    book_names
)

if st.button("📖 Show Recommendations"):
    with st.spinner("Finding similar books..."):
        recommended_books, poster_urls = obj.recommend_book(selected_book)

    st.subheader("✨ Recommended Books")

    cols = st.columns(5)
    for i in range(1, 6):
        with cols[i - 1]:
            st.image(poster_urls[i], use_container_width=True)
            st.caption(recommended_books[i])
