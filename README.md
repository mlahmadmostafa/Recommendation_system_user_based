# Recommendation_system_user_based



This project involves building a deep learning model to predict the relevance of a search term to a product. It preprocesses and tokenizes textual data, uses Word2Vec embeddings, and leverages an LSTM-based architecture to extract meaningful patterns from product titles, descriptions, and search terms. 

![Screenshot 2024-11-21 201533](https://github.com/user-attachments/assets/3f42c149-6a31-4b1b-855a-087cb0431c72)

### Project Highlights

1. **Data Preprocessing:**
   - Removes special characters and normalizes text (lowercasing and symbol removal).
   - Extracts keywords from product descriptions using the RAKE (Rapid Automatic Keyword Extraction) algorithm.
   - Normalizes relevance scores for model training.

2. **Word Embeddings:**
   - Utilizes pre-trained Word2Vec embeddings for text representation.
   - Creates embedding matrices for product titles, descriptions, and search terms.

3. **Model Architecture:**
   - **Sub-models** for each input (title, description, and search term) using an Embedding layer followed by an LSTM.
   - Combines outputs of sub-models via a concatenation layer.
   - Outputs a relevance score using a Dense layer with a sigmoid activation.

4. **Callbacks and Training:**
   - Uses ModelCheckpoint to save the best-performing model.
   - Implements early stopping to prevent overfitting.
   - Trains with a huber loss function to balance between mean squared error and mean absolute error.

5. **Performance:**
   - Achieved a mean absolute error (MAE) of 0.18 on training and 0.21 on validation datasets.

### Key Components

#### Preprocessing and Feature Engineering
- Data cleaning to remove noise from textual inputs.
- Keyword extraction for better feature representation.
- Sequence padding and tokenization of the input text.

#### Model Structure
- Title Input ➡️ Embedding ➡️ LSTM ➡️ Dense
- Description Input ➡️ Embedding ➡️ LSTM ➡️ Dense
- Search Term Input ➡️ Embedding ➡️ LSTM ➡️ Dense
- Combined Outputs ➡️ Concatenate ➡️ Dense ➡️ Relevance Score

#### Training
- Input: Tokenized sequences for product titles, descriptions, and search terms.
- Output: Normalized relevance scores.
- Optimizer: Adam.
- Loss Function: Huber.

### Sample Result
A sample prediction from the trained model:  
```python
model.predict([X_test_title, X_test_desc, X_test_term])[0]
# Example output: [0.87] (predicted relevance score)
