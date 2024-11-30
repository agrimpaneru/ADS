import streamlit as st
from bertopic import BERTopic
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the trained BERTopic model
@st.cache_resource
def load_model():
    return BERTopic.load("Trained_models/bertopic_news_model")

# Load the dataset
@st.cache_resource
def load_dataset():
    import json
    with open("samplesentenceList.json", "r") as file:
        return json.load(file)

# Load model and dataset
topic_model = load_model()
docs = load_dataset()

# Visualizations
def visualize_barchart():
    st.subheader("Barchart of Top Words per Topic")
    num_topics = st.slider(
        "Select the number of topics to visualize",
        min_value=1,
        max_value=len(topic_model.get_topics()),
        value=5,
        step=1,
    )
    fig = topic_model.visualize_barchart(top_n_topics=num_topics)
    st.plotly_chart(fig, use_container_width=True)

def visualize_heatmap():
    st.subheader("Topic-Topic Similarity Heatmap")
    fig = topic_model.visualize_heatmap()
    st.plotly_chart(fig)

def visualize_hierarchy():
    st.subheader("Topic Hierarchy (Dendrogram)")
    fig = topic_model.visualize_hierarchy()
    st.plotly_chart(fig)

def visualize_topics():
    st.subheader("Topic Overview")
    fig = topic_model.visualize_topics()
    st.plotly_chart(fig)

def visualize_wordcloud():
    st.subheader("WordCloud of Top Words per Topic")
    
    # Select a specific topic number to visualize
    topic_number = st.selectbox(
        "Select the Topic Number",
        options=[i for i in range(len(topic_model.get_topics()))],
        index=0,
    )
    
    # Get the top words for the selected topic
    words = topic_model.get_topic(topic_number)
    all_words = " ".join([word for word, _ in words])
    
    # Generate WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_words)
    
    # Display WordCloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

def analyze_article():
    st.subheader("Analyze a News Article")
    
    # Input for a new article
    article = st.text_area("Paste or write a news article below:", height=150)
    
    if st.button("Analyze Article"):
        if not article.strip():
            st.warning("Please provide an article for analysis.")
            return
        
        # Get topic distribution for the article from the news dataset model
        topics, probs = topic_model.transform([article])

        if probs is None or len(probs[0]) == 0:
            st.warning("No topics were assigned to this article.")
            return

        # Prepare data for visualization (Top 10 topics)
        topic_probs = sorted(
            [(topic, prob) for topic, prob in enumerate(probs[0])],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Get top words for each topic
        topic_names = [", ".join([word for word, _ in topic_model.get_topic(topic)]) for topic, _ in topic_probs]
        
        # Create a DataFrame for table
        df = pd.DataFrame(topic_probs, columns=["Topic", "Probability"])
        df["Topic"] = df["Topic"].apply(lambda x: f"Topic {x}")
        df["Top Words"] = topic_names
        
        # Display table
        st.write("### Topic Distribution Table")
        st.dataframe(df, use_container_width=True)
        
        # Plot Pie chart
        fig_pie = px.pie(
            df,
            values="Probability",
            names="Topic",
            title="Topic Distribution (Pie Chart)",
            color="Topic",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie)

        # Plot Bar chart
        fig_bar = px.bar(
            df,
            x="Topic",
            y="Probability",
            title="Topic Distribution (Bar Chart)",
            color="Topic",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_bar)

# Main app
def main():
    st.title("BERTopic Visualization Dashboard")

    st.sidebar.title("Navigation")
    pages = [
        "Home - Visualizations",
        "Analyze News Article",
    ]
    selected_page = st.sidebar.selectbox("Select Page", pages)

    if selected_page == "Home - Visualizations":
        st.header("Visualizations")

        # Move checkboxes to the sidebar
        barchart = st.sidebar.checkbox("Show Barchart of Top Words per Topic")
        heatmap = st.sidebar.checkbox("Show Topic-Topic Similarity Heatmap")
        hierarchy = st.sidebar.checkbox("Show Topic Hierarchy (Dendrogram)")
        topics = st.sidebar.checkbox("Show Topic Overview")
        wordcloud = st.sidebar.checkbox("Show WordCloud of Top Words per Topic")

        # Display selected visualizations
        if barchart:
            visualize_barchart()
        if heatmap:
            visualize_heatmap()
        if hierarchy:
            visualize_hierarchy()
        if topics:
            visualize_topics()
        if wordcloud:
            visualize_wordcloud()

    elif selected_page == "Analyze News Article":
        analyze_article()

# Run the app
if __name__ == "__main__":
    main()
