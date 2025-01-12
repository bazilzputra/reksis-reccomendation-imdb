import streamlit as st
import pandas as pd
from surprise import Dataset, Reader
from surprise import KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load dataset
uploaded_file = st.file_uploader("Upload IMDb Dataset", type=["csv"])

if uploaded_file:
    # Read dataset
    df = pd.read_csv(uploaded_file)

    # Preprocessing
    df['IMDb Rating'] = pd.to_numeric(df['IMDb Rating'], errors='coerce').fillna(df['IMDb Rating'].mean())

    # Rename columns for compatibility with Surprise library
    df_surprise = df.rename(columns={
        "Age Rating": "user_id",
        "Title": "item_id",
        "IMDb Rating": "rating"
    })

    # Ensure required columns exist
    required_columns = ["user_id", "item_id", "rating"]
    if not all(col in df_surprise.columns for col in required_columns):
        st.error("Dataset must contain 'Age Rating', 'Title', and 'IMDb Rating' columns.")
    else:
        # Prepare data for Surprise
        reader = Reader(rating_scale=(df_surprise["rating"].min(), df_surprise["rating"].max()))
        data = Dataset.load_from_df(df_surprise[required_columns], reader)

        # Train-test split
        trainset, testset = train_test_split(data, test_size=0.25)

        # Configure collaborative filtering model
        sim_options = {
            "name": "cosine",
            "user_based": True,
        }
        algo = KNNBasic(sim_options=sim_options)

        # Train the model
        algo.fit(trainset)

        # Evaluate the model
        predictions = algo.test(testset)
        rmse = accuracy.rmse(predictions)
        st.success(f"Model trained with RMSE: {rmse:.4f}")

        # Recommendation section
        st.title("IMDb Recommendation System")

        age_rating = st.text_input("Enter Age Rating (e.g., PG-13):")
        movie_title = st.text_input("Enter Movie Title (e.g., The Dark Knight):")
        recommend_button = st.button("Get Recommendation")

        if recommend_button:
            if not age_rating or not movie_title:
                st.error("Please provide both Age Rating and Movie Title.")
            else:
                # Predict rating
                prediction = algo.predict(uid=age_rating, iid=movie_title)
                st.write(f"Predicted IMDb Rating for '{movie_title}' (Age Rating: {age_rating}): {prediction.est:.2f}")

                # Display top recommendations
                st.write("Top Recommendations:")
                user_ratings = algo.get_neighbors(algo.trainset.to_inner_uid(age_rating), k=10)
                recommended_titles = [algo.trainset.to_raw_iid(inner_id) for inner_id in user_ratings]
                st.write(recommended_titles)

else:
    st.info("Please upload a valid IMDb dataset.")
