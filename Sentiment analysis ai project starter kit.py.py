# ============================================================================
# TENSORFLOW/KERAS TEXT CLASSIFICATION TEMPLATE (Base 44 Architecture)
# Binary Sentiment Analysis - Clean & Structured Implementation
# ============================================================================

# [1] IMPORTS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# [2] CONFIGURATION SECTION
class Config:
    """Configuration parameters for the text classification model"""
    
    # Tokenization parameters
    MAX_WORDS = 10000
    MAX_LEN = 100
    
    # Model architecture
    EMBEDDING_DIM = 50
    HIDDEN_UNITS = 32
    
    # Training parameters
    EPOCHS = 5
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    
    # Data split
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Model compilation
    OPTIMIZER = 'adam'
    LOSS = 'binary_crossentropy'
    METRICS = ['accuracy']

# [3] DATA LOADING & PREPROCESSING
def load_and_preprocess_data(file_path, text_column='text', label_column='label'):
    """
    Load and preprocess text data from CSV file
    
    Args:
        file_path (str): Path to CSV file
        text_column (str): Name of text column
        label_column (str): Name of label column
        
    Returns:
        tuple: (texts, labels) preprocessed data
    """
    try:
        # Load data
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        # Validate columns
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in data")
        if label_column not in df.columns:
            raise ValueError(f"Column '{label_column}' not found in data")
        
        # Extract texts and labels
        texts = df[text_column].astype(str).tolist()
        labels = df[label_column].tolist()
        
        # Data validation
        print(f"Total samples: {len(texts)}")
        print(f"Positive samples: {sum(labels)}")
        print(f"Negative samples: {len(labels) - sum(labels)}")
        
        # Remove empty texts
        valid_indices = [i for i, text in enumerate(texts) if text.strip()]
        texts = [texts[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        
        print(f"Valid samples after cleaning: {len(texts)}")
        
        return texts, labels
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

# [4] TOKENIZATION & SEQUENCING
def create_tokenizer_and_sequences(texts, labels):
    """
    Create tokenizer and convert texts to sequences
    
    Args:
        texts (list): List of text strings
        labels (list): List of binary labels
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, tokenizer)
    """
    try:
        # Split data
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts, labels, 
            test_size=Config.TEST_SIZE, 
            random_state=Config.RANDOM_STATE,
            stratify=labels
        )
        
        print(f"Training samples: {len(X_train_text)}")
        print(f"Test samples: {len(X_test_text)}")
        
        # Create tokenizer
        tokenizer = Tokenizer(
            num_words=Config.MAX_WORDS,
            oov_token="<OOV data-filename='pages/TemplateGenerator' data-linenumber='151' data-visual-selector-id='pages/TemplateGenerator151'>",
            lower=True,
            split=' '
        )
        
        # Fit tokenizer on training data only
        tokenizer.fit_on_texts(X_train_text)
        
        # Convert texts to sequences
        X_train_seq = tokenizer.texts_to_sequences(X_train_text)
        X_test_seq = tokenizer.texts_to_sequences(X_test_text)
        
        # Pad sequences
        X_train = pad_sequences(X_train_seq, maxlen=Config.MAX_LEN, padding='post', truncating='post')
        X_test = pad_sequences(X_test_seq, maxlen=Config.MAX_LEN, padding='post', truncating='post')
        
        # Convert labels to numpy arrays
        y_train = np.array(y_train, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32)
        
        print(f"Training sequences shape: {X_train.shape}")
        print(f"Test sequences shape: {X_test.shape}")
        print(f"Vocabulary size: {len(tokenizer.word_index)}")
        
        return X_train, X_test, y_train, y_test, tokenizer
        
    except Exception as e:
        print(f"Error in tokenization: {str(e)}")
        raise

# [5] MODEL BUILDING
def build_model():
    """
    Build the Sequential model with Base 44 architecture
    
    Returns:
        Sequential: Compiled Keras model
    """
    try:
        model = Sequential([
            # Embedding layer
            Embedding(
                input_dim=Config.MAX_WORDS,
                output_dim=Config.EMBEDDING_DIM,
                input_length=Config.MAX_LEN,
                name='embedding_layer'
            ),
            
            # Global Average Pooling
            GlobalAveragePooling1D(name='global_avg_pooling'),
            
            # Dense hidden layer with ReLU
            Dense(
                Config.HIDDEN_UNITS,
                activation='relu',
                name='hidden_layer'
            ),
            
            # Output layer with sigmoid for binary classification
            Dense(1, activation='sigmoid', name='output_layer')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(),
            loss=Config.LOSS,
            metrics=Config.METRICS
        )
        
        print("Model built successfully!")
        model.summary()
        
        return model
        
    except Exception as e:
        print(f"Error building model: {str(e)}")
        raise

# [6] TRAINING
def train_model(model, X_train, y_train):
    """
    Train the model
    
    Args:
        model: Keras model
        X_train: Training sequences
        y_train: Training labels
        
    Returns:
        History: Training history
    """
    try:
        print("Starting model training...")
        
        history = model.fit(
            X_train, y_train,
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            validation_split=Config.VALIDATION_SPLIT,
            verbose=1
        )
        
        print("Training completed!")
        return history
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

# [7] EVALUATION
def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test data
    
    Args:
        model: Trained Keras model
        X_test: Test sequences
        y_test: Test labels
        
    Returns:
        tuple: (test_loss, test_accuracy)
    """
    try:
        print("Evaluating model on test data...")
        
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return test_loss, test_accuracy
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

# [8] PREDICTION FUNCTION
def predict_sentiment(model, tokenizer, texts):
    """
    Predict sentiment for new texts with confidence scores
    
    Args:
        model: Trained Keras model
        tokenizer: Fitted tokenizer
        texts (list or str): Text(s) to predict
        
    Returns:
        list: List of dictionaries with predictions and confidence scores
    """
    try:
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        # Handle empty input
        if not texts:
            return []
        
        # Convert texts to sequences with OOV handling
        sequences = tokenizer.texts_to_sequences(texts)
        
        # Check for all-OOV texts
        for i, seq in enumerate(sequences):
            if not seq or all(token == 1 for token in seq):  # 1 is typically OOV token index
                print(f"Warning: Text {i+1} contains only out-of-vocabulary words")
        
        # Pad sequences
        padded_sequences = pad_sequences(sequences, maxlen=Config.MAX_LEN, padding='post', truncating='post')
        
        # Make predictions
        predictions = model.predict(padded_sequences, verbose=0)
        
        # Format results
        results = []
        for i, (text, prob) in enumerate(zip(texts, predictions.flatten())):
            confidence = float(prob) if prob > 0.5 else float(1 - prob)
            sentiment = "positive" if prob > 0.5 else "negative"
            
            results.append({
                'text': text,
                'sentiment': sentiment,
                'probability': float(prob),
                'confidence': confidence,
                'prediction_score': f"{confidence:.3f}"
            })
        
        return results
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return []

# [9] EXAMPLE USAGE
def main():
    """
    Main function demonstrating the complete workflow
    """
    try:
        # Example with dummy data (replace with your actual data loading)
        print("="*60)
        print("TENSORFLOW/KERAS TEXT CLASSIFICATION EXAMPLE")
        print("="*60)
        
        # For demonstration - create sample data
        # Replace this section with: texts, labels = load_and_preprocess_data('your_data.csv')
        sample_texts = [
            "This movie is absolutely fantastic and amazing!",
            "I love this product, it works perfectly",
            "Great service, highly recommended",
            "This is terrible and awful",
            "I hate this, worst experience ever",
            "Poor quality, very disappointed",
            "Average movie, nothing special",
            "It's okay, could be better"
        ]
        sample_labels = [1, 1, 1, 0, 0, 0, 0, 0]  # 1=positive, 0=negative
        
        print("Using sample data for demonstration...")
        texts, labels = sample_texts, sample_labels
        
        # Create tokenizer and sequences
        X_train, X_test, y_train, y_test, tokenizer = create_tokenizer_and_sequences(texts, labels)
        
        # Build model
        model = build_model()
        
        # Train model
        history = train_model(model, X_train, y_train)
        
        # Evaluate model
        test_loss, test_accuracy = evaluate_model(model, X_test, y_test)
        
        # Test predictions with 4 examples
        print("\n" + "="*40)
        print("TESTING PREDICTIONS")
        print("="*40)
        
        test_texts = [
            "This is absolutely wonderful and amazing!",  # Clear positive
            "This is horrible and disgusting",           # Clear negative  
            "It's okay, nothing special really",         # Neutral/mixed
            "xyz qwerty zxcvbn asdfgh"                   # All-OOV text
        ]
        
        results = predict_sentiment(model, tokenizer, test_texts)
        
        for i, result in enumerate(results, 1):
            print(f"\nTest {i}:")
            print(f"Text: {result['text']}")
            print(f"Prediction: {result['sentiment']} ({result['prediction_score']} confidence)")
            print(f"Raw probability: {result['probability']:.4f}")
        
        print("\n" + "="*40)
        print("TEMPLATE EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*40)
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
