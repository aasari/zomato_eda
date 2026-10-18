# Zomato Bangalore – EDA & LSTM Sentiment Model

An end-to-end data science project that explores Zomato's Bangalore
restaurant data and builds a deep learning model to classify restaurant
reviews as **positive** or **negative**.

The project starts with detailed exploratory data analysis (EDA) and
visualizations, moves into natural language processing on reviews, and
finishes with an **LSTM** sentiment classifier built in Keras.

## What's inside

**1. Exploratory Data Analysis**
- Top restaurant chains in Bangalore
- Online ordering vs. table booking availability
- Rating and cost distributions
- Cost vs. rating relationship
- Most common restaurant types, cuisines, and "foodie" areas
- A simple filter to find the best budget restaurants in any location

**2. Geospatial analysis (maps)**
- Extracts latitude/longitude for each location using **Geopy**
- Heatmaps of restaurant density across the city (**Folium**)
- Location maps of popular casual dining, quick bites, and cafe chains

**3. Text analysis (NLP)**
- Word clouds of liked dishes and restaurant reviews
- Topic modelling of positive vs. negative reviews (**LDA / Gensim**)
- Word2Vec + t-SNE visualizations of review words and adjectives

**4. Sentiment model**
- Prepares reviews and labels them positive/negative from ratings
- Tokenizes and pads text sequences
- Trains an **LSTM** neural network (Keras) to classify sentiment
- Validates the model and reports accuracy

## Tech stack

- **Python**
- **pandas / numpy** – data handling
- **matplotlib / seaborn / plotly** – visualization
- **geopy / folium** – maps and geocoding
- **wordcloud** – word clouds
- **nltk / gensim** – text processing and topic modelling
- **scikit-learn** – TF-IDF, t-SNE, train/test split
- **keras (TensorFlow)** – LSTM sentiment model

## Dataset

This project uses the **Zomato Bangalore Restaurants** dataset (`zomato.csv`),
commonly available on Kaggle. The script expects it at `../input/zomato.csv`
(the default Kaggle/Colab path), so update that path if you run it locally.

Key columns include: restaurant name, address, online order / table booking
flags, rating, votes, location, restaurant type, dishes liked, cuisines,
approximate cost for two, and a list of reviews.

## How to run

This was originally a Google Colab / Kaggle notebook, so the easiest way to
run it is in a notebook environment.

1. Download the Zomato dataset and place `zomato.csv` where the script
   expects it (or edit the path in the code).

2. Install the dependencies:
```bash
   pip install numpy pandas seaborn matplotlib plotly wordcloud geopy folium tqdm nltk gensim scikit-learn keras tensorflow
```

3. Download the required NLTK data:
```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
```

4. Run the script (or paste the cells into a notebook):
```bash
   python zomato_complete_eda_and_lstm_model.py
```

## Note

The interactive Plotly and Folium maps render best inside a Jupyter/Colab
notebook rather than a plain script. The map-chain sections use a Mapbox
access token, so you may need to supply your own for those plots.

## Author

**Aasari Prashanth** – Data Scientist
