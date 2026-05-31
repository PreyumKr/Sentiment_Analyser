# Sentiment Analysis App

A web-based sentiment analysis application using **VADER** and **RoBERTa** models built with Streamlit.

## Things with Sentiment Analysis different from other NLP tasks:
- **Subjectivity**: Sentiment is subjective and can vary based on context, culture, and individual interpretation.
- **Context Dependency**: The sentiment of a word or phrase can change based on the surrounding context (e.g., "not good" vs. "good").
- **Sarcasm and Irony**: These can be particularly challenging for sentiment analysis models to detect accurately. Models like RoBERTa, which are context-aware, can perform better in these cases compared to rule-based models like VADER.
- Removal of **stop words** can sometimes lead to loss of sentiment information, as certain stop words (e.g., "not") can significantly alter the sentiment of a sentence.
- Removal of **punctuation** can also affect sentiment analysis, as punctuation can convey emotions (e.g., "!" can indicate excitement or emphasis).
- **Other tasks** where we remove stop words and punctuation (e.g., **topic modeling**, **text classification**) may not be as sensitive to these elements, whereas sentiment analysis can be heavily influenced by them.

## Approaches

1) **VADER (Valence Aware Dictionary and sEntiment Reasoner)** : 
    * It breaks the text into words and looks up each word in its lexicon to determine its sentiment score.
    * It also considers the context of words (e.g., negations, intensifiers) and punctuation to adjust the sentiment scores accordingly.
    * It provides a compound score that summarizes the overall sentiment of the text, as well as individual scores for positive, negative, and neutral sentiments.
    * Unknown words are ignored, and the model relies on its predefined lexicon to analyze sentiment. This can lead to inaccuracies if the text contains slang, misspellings, or domain-specific language that is not included in the lexicon.
2) **RoBERTa (Robustly Optimized BERT Pretraining Approach)** :
    * It is a transformer-based model that has been fine-tuned for sentiment analysis tasks, including Twitter sentiment analysis.
    * It uses attention mechanisms to understand the context of words in a sentence, allowing it to capture nuances and subtleties in language that may affect sentiment.
    * It can handle shorthands, slang, and misspellings better than VADER due to its training on large datasets that include such variations.
    * The text needs to be tokenized and converted into a format suitable for the model, which can be more computationally intensive than VADER's rule-based approach.
    * With GPU acceleration, RoBERTa can provide faster inference times, but it can still run on CPU with reasonable performance for single inputs, though it may be slower than VADER.

## Features

- Input text area for user to enter review or comment text
- Option to select which models to use (VADER, RoBERTa, or both)

## Installation

1. **Create virtual environment** (if not already done):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements_app.txt
   ```

   Or if you already have the packages installed, just add Streamlit:
   ```bash
   pip install streamlit==1.41.1
   ```

## Running the App

```bash
streamlit run app.py
```

This will:
- Open a browser window at `http://localhost:8501`
- Display the sentiment analysis interface
- Models will be downloaded on first run (may take a few minutes)

## Usage

1. **Enter Text**: Type or paste text in the text area
2. **Click Analyze**: Press the "Analyze Sentiment" button
3. **View Results**: 
   - See VADER scores and sentiment
   - See RoBERTa prediction with confidence
   - Compare both models side-by-side

## Models Used

### VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Type**: Lexicon-based
- **Speed**: Very fast
- **Resource**: CPU only
- **Score Range**: -1 to 1
- **Output**: Positive, Negative, Neutral, Compound scores

### RoBERTa (Robustly Optimized BERT Pretraining Approach)
- **Type**: Fine-tuned transformer model
- **Speed**: Slower than VADER
- **Resource**: CPU-compatible
- **Score Range**: 0 to 1 (per label)
- **Model**: `cardiffnlp/twitter-roberta-base-sentiment`
- **Output**: Negative, Neutral, Positive probabilities

## Interpreting Results

### VADER Compound Score
- **≥ 0.05**: Positive sentiment
- **≤ -0.05**: Negative sentiment
- **Between -0.05 and 0.05**: Neutral sentiment

### RoBERTa Confidence
- Shows probability for each sentiment label
- Higher score = more confident prediction
- Best for context-aware analysis

## Example Outputs

**Positive Input**: "This product is amazing!"
- VADER: Compound ≈ 0.76 (Positive)
- RoBERTa: Positive ≈ 0.98

**Negative Input**: "Terrible experience, very disappointed"
- VADER: Compound ≈ -0.81 (Negative)
- RoBERTa: Negative ≈ 0.99

## Requirements

- Python 3.8+
- 2GB+ RAM
- GPU optional (app runs on CPU)
- Internet connection (for downloading models on first run)

## File Structure

```
Semantic_Analyser/
├── app.py                 # Streamlit app
├── requirements_app.txt   # Python dependencies
├── Sentiment_analysis.ipynb # Original notebook
├── data/
│   └── Reviews.csv       # Sample data
└── README.md             # This file
```

## Troubleshooting

**Models take too long to load?**
- First run downloads models from HuggingFace (~1GB total)
- Subsequent runs use cached models

**Port 8501 already in use?**
```bash
streamlit run app.py --server.port 8502
```

**Out of memory?**
- VADER uses minimal memory (< 50MB)
- RoBERTa uses ~500MB
- Should work on most systems

## Performance

- **Single paragraph (< 512 tokens)**: ~1-3 seconds
- **VADER alone**: < 0.1 seconds
- **RoBERTa alone**: ~1-2 seconds
- **Both models**: ~2-3 seconds

## Future Enhancements

- [ ] Batch processing for multiple texts
- [ ] Aspect-based sentiment analysis
- [ ] Emotion detection
- [ ] Language support beyond English
- [ ] Save results to CSV
- [ ] Model fine-tuning interface

## License

MIT License - Feel free to use and modify

## References

- VADER: https://github.com/cjhutto/vaderSentiment
- RoBERTa: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
- Streamlit: https://streamlit.io/
