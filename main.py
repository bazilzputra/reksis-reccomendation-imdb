import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Upload Dataset
uploaded_file = st.file_uploader("Upload your IMDb dataset", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    # Clean missing values
    df = df.fillna('')

    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Title'])

    # Compute Cosine Similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Recommendation Function
    def get_recommendations(title, cosine_sim=cosine_sim):
        indices = pd.Series(df.index, index=df['Title']).drop_duplicates()
        idx = indices[title]

        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]

        movie_indices = [i[0] for i in sim_scores]
        return df['Title'].iloc[movie_indices]

    # Input dan Output
    movie = st.text_input("Enter a movie title")
    if movie and movie in df['Title'].values:
        st.write("Recommended Movies:")
        st.write(get_recommendations(movie))
    elif movie:
        st.write("Movie not found!")
