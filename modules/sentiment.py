# Fungsi untuk analisis sentimen dengan VADER
def sentiment_analysis_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_dict = analyzer.polarity_scores(text)
    return sentiment_dict['compound']

# Fungsi untuk analisis sentimen dengan TextBlob
def sentiment_analysis_textblob(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Fungsi untuk mengubah nilai sentimen menjadi skala Likert
def sentiment_to_likert(score, scale=5):
    if scale == 5:
        if score >= 0.5:
            return 5  # Sangat Puas Sekali
        elif score >= 0.2:
            return 4  # Sangat Puas
        elif score >= -0.2:
            return 3  # Puas
        elif score >= -0.5:
            return 2  # Cukup Puas
        else:
            return 1  # Kurang Puas