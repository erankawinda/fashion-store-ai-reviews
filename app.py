from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import re
from difflib import SequenceMatcher
import nltk
from nltk.stem import PorterStemmer
from collections import defaultdict

# Download required NLTK data (only needs to run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(__name__)

# Initialize globals
model = None
vectorizer = None
items_df = None
stemmer = PorterStemmer()

# Category emoji mapping
CATEGORY_EMOJIS = {
    'dresses': '👗',
    'tops': '👚',
    'shirts': '👔',
    'pants': '👖',
    'jeans': '👖',
    'skirts': '👗',
    'shorts': '🩳',
    'sweaters': '🧥',
    'jackets': '🧥',
    'coats': '🧥',
    'scarves': '🧣',
    'accessories': '👜',
    'shoes': '👠',
    'bags': '👜',
    'jewelry': '💍',
    'default': '👕'
}


def get_category_emoji(category):
    """Get emoji for clothing category"""
    if not category:
        return CATEGORY_EMOJIS['default']

    category_lower = category.lower()
    for key, emoji in CATEGORY_EMOJIS.items():
        if key in category_lower:
            return emoji
    return CATEGORY_EMOJIS['default']


def load_model_and_data():
    """Load ML model, vectorizer, and dataset"""
    global model, vectorizer, items_df
    try:
        # Load pre-trained model
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
            print("✓ Model loaded successfully")

        # Load vectorizer
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
            print("✓ Vectorizer loaded successfully")

    except Exception as e:
        print(f"❌ ERROR loading model/vectorizer: {str(e)}")
        raise

    # Load dataset
    items_df = pd.read_csv('assignment3_II.csv')
    items_df['Clothing ID'] = items_df['Clothing ID'].astype(int)

    # Create search index for better performance
    create_search_index()

    print(f"✓ Loaded {len(items_df)} items from dataset")


# Search index for improved performance
search_index = defaultdict(set)
stem_mapping = {}


def create_search_index():
    """Create inverted index for efficient searching based on categories only"""
    global search_index, stem_mapping

    search_index = defaultdict(set)
    stem_mapping = {}

    for idx, row in items_df.iterrows():
        # Index ONLY category names (Class Name)
        if pd.notna(row['Class Name']):
            category_words = row['Class Name'].lower().split()
            for word in category_words:
                stem = stemmer.stem(word)
                search_index[stem].add(row['Clothing ID'])
                stem_mapping[stem] = word


def smart_search(query):
    """
    Enhanced search algorithm using stemming and fuzzy matching
    ONLY searches in category names (Class Name)
    Handles plural forms and similar words automatically
    """
    if not query:
        return items_df

    query_lower = query.lower().strip()
    query_words = query_lower.split()

    # Get stems for query words
    query_stems = [stemmer.stem(word) for word in query_words]

    # Find matching items using inverted index
    matching_ids = set()

    # Exact stem matches (handles plurals automatically)
    for stem in query_stems:
        if stem in search_index:
            matching_ids.update(search_index[stem])

    # Fuzzy matching for close matches
    if not matching_ids:
        for stem in query_stems:
            for indexed_stem in search_index:
                similarity = SequenceMatcher(None, stem, indexed_stem).ratio()
                if similarity > 0.8:  # 80% similarity threshold
                    matching_ids.update(search_index[indexed_stem])

    # If still no matches, try substring matching in categories only
    if not matching_ids:
        for idx, row in items_df.iterrows():
            if pd.notna(row['Class Name']) and query_lower in row['Class Name'].lower():
                matching_ids.add(row['Clothing ID'])

    # Filter dataframe by matching IDs
    if matching_ids:
        result_df = items_df[items_df['Clothing ID'].isin(matching_ids)].copy()

        # Calculate relevance scores based on category match only
        result_df['relevance_score'] = result_df.apply(
            lambda row: calculate_category_relevance_score(row, query_stems), axis=1
        )

        # Sort by relevance score, then by rating
        result_df = result_df.sort_values(['relevance_score', 'Rating'], ascending=[False, False])
        result_df = result_df.drop('relevance_score', axis=1)

        return result_df

    return pd.DataFrame()  # Empty dataframe if no matches


def calculate_category_relevance_score(row, query_stems):
    """Calculate relevance score based on category match only"""
    score = 0

    # Category match scoring
    if pd.notna(row['Class Name']):
        category_stems = [stemmer.stem(word) for word in row['Class Name'].lower().split()]

        # Exact stem match
        for stem in query_stems:
            if stem in category_stems:
                score += 10

        # Partial match bonus
        category_lower = row['Class Name'].lower()
        for query_word in query_stems:
            if stem_mapping.get(query_word, query_word) in category_lower:
                score += 5

    # Add small rating bonus for tie-breaking
    if pd.notna(row['Rating']):
        score += float(row['Rating']) * 0.1

    return score


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/api/items')
def get_items():
    """Get items with category-based smart search"""
    query = request.args.get('search', '').strip()

    if query:
        # Use smart search algorithm (category-based only)
        df = smart_search(query)
        print(f"Category search '{query}': Found {len(df)} matching items")
    else:
        # Return all items if no search query
        df = items_df.copy()

    # Get unique items with aggregated ratings
    unique_items = []
    seen_items = set()

    for _, row in df.iterrows():
        item_id = row['Clothing ID']
        if item_id not in seen_items:
            seen_items.add(item_id)
            # Get all reviews for this item to calculate average rating
            item_reviews = items_df[items_df['Clothing ID'] == item_id]
            avg_rating = item_reviews['Rating'].mean() if len(item_reviews) > 0 else 0

            category = row.get('Class Name', 'Unknown Category')
            unique_items.append({
                'id': int(item_id),
                'title': row.get('Clothes Title', 'Unknown Title'),
                'category': category,
                'category_emoji': get_category_emoji(category),
                'description': row.get('Clothes Description', 'No description available'),
                'rating': round(avg_rating, 1),
                'review_count': len(item_reviews)
            })

    # Limit results
    unique_items = unique_items[:50]

    # If searching, get matched categories for display
    matched_categories = set()
    if query and unique_items:
        for item in unique_items:
            matched_categories.add(item['category'])

    return jsonify({
        'count': len(unique_items),
        'search_query': query,
        'matched_categories': list(matched_categories) if query else [],
        'items': unique_items
    })


@app.route('/api/item/<int:item_id>')
def get_item(item_id):
    """Get detailed information for a specific item"""
    # Get all reviews for this item
    item_reviews = items_df[items_df['Clothing ID'] == item_id]
    if item_reviews.empty:
        return jsonify({'error': 'Item not found'}), 404

    # Sort by timestamp if available (newest first)
    if 'Timestamp' in item_reviews.columns:
        item_reviews = item_reviews.sort_values('Timestamp', ascending=False)

    # Get item details from first row
    item_info = item_reviews.iloc[0]
    category = item_info.get('Class Name', 'Unknown Category')

    # Separate reviews by recommendation
    recommended_reviews = []
    not_recommended_reviews = []

    for _, review in item_reviews.iterrows():
        review_data = {
            'title': review.get('Title', ''),
            'text': review.get('Review Text', ''),
            'rating': int(float(review.get('Rating', 0))),
            'recommended': int(review.get('Recommended IND', 0))
        }

        if review_data['recommended'] == 1:
            recommended_reviews.append(review_data)
        else:
            not_recommended_reviews.append(review_data)

    # Limit the number of reviews shown (show latest 5 of each type)
    total_recommended = len(recommended_reviews)
    total_not_recommended = len(not_recommended_reviews)
    recommended_reviews = recommended_reviews[:5]
    not_recommended_reviews = not_recommended_reviews[:5]

    # Calculate statistics
    avg_rating = item_reviews['Rating'].mean() if 'Rating' in item_reviews.columns else 0

    return jsonify({
        'id': item_id,
        'title': item_info.get('Clothes Title', 'Unknown Title'),
        'category': category,
        'category_emoji': get_category_emoji(category),
        'description': item_info.get('Clothes Description', 'No description available'),
        'rating': round(avg_rating, 1),
        'total_reviews': len(item_reviews),
        'recommended_count': total_recommended,
        'not_recommended_count': total_not_recommended,
        'recommended_reviews': recommended_reviews,
        'not_recommended_reviews': not_recommended_reviews,
        'has_more_recommended': total_recommended > 5,
        'has_more_not_recommended': total_not_recommended > 5
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict recommendation using both title and description"""
    try:
        data = request.get_json() or {}
        title = data.get('review_title', '').strip()
        text = data.get('review_text', '').strip()

        if not text:
            return jsonify({'error': 'Review text is required'}), 400

        # Combine title and text for better prediction
        combined_text = f"{title} {text}".strip()

        print(f"\n=== AI PREDICTION ===")
        print(f"Title: '{title}'")
        print(f"Text: '{text}'")
        print(f"Combined: '{combined_text}'")

        # Transform text based on vectorizer configuration
        if hasattr(vectorizer, 'token_pattern'):
            token_pattern = vectorizer.token_pattern

            # Check if comma-separated format is expected
            if '[^,]+' in str(token_pattern):
                # Convert to comma-separated tokens
                tokens = combined_text.lower().split()
                tokens = [t for t in tokens if len(t) >= 2]
                processed_text = ','.join(tokens)
                print(f"Using comma-separated format: '{processed_text}'")
                X = vectorizer.transform([processed_text])
            else:
                # Use standard format
                X = vectorizer.transform([combined_text])
        else:
            # Default to standard format
            X = vectorizer.transform([combined_text])

        print(f"Feature vector: shape={X.shape}, non-zero features={X.nnz}")

        # Make prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        # Get probability for positive class
        prob_recommended = float(probabilities[1])
        confidence = prob_recommended if prediction == 1 else (1 - prob_recommended)

        print(f"Prediction: {prediction}, Confidence: {confidence:.2f}")

        return jsonify({
            'recommendation': int(prediction),
            'probability': prob_recommended,
            'recommended_text': 'Recommended' if prediction == 1 else 'Not Recommended',
            'confidence': f"{confidence * 100:.1f}%"
        })

    except Exception as e:
        print(f"ERROR in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Prediction failed. Please check server logs.'}), 500


@app.route('/api/reviews', methods=['POST'])
def add_review():
    """Add a new review to the dataset"""
    global items_df

    data = request.get_json() or {}

    # Validate required fields
    item_id = data.get('item_id')
    review_title = data.get('title', '').strip()
    review_text = data.get('description', '').strip()
    rating = data.get('rating')
    recommendation = data.get('recommendation')

    if not all([item_id, review_text, rating is not None, recommendation is not None]):
        return jsonify({'error': 'Missing required fields'}), 400

    # Get item information
    item_rows = items_df[items_df['Clothing ID'] == int(item_id)]
    if item_rows.empty:
        return jsonify({'error': 'Item not found'}), 404

    item_info = item_rows.iloc[0]

    # Create new review with unique ID and timestamp for sorting
    import time
    new_review_id = int(time.time() * 1000)  # Use timestamp as unique ID
    new_review = {
        'Clothing ID': int(item_id),
        'Class Name': item_info['Class Name'],
        'Title': review_title,
        'Review Text': review_text,
        'Rating': int(rating),
        'Recommended IND': int(recommendation),
        'Clothes Title': item_info['Clothes Title'],
        'Clothes Description': item_info['Clothes Description'],
        'Review ID': new_review_id,
        'Timestamp': time.time()  # Add timestamp for sorting
    }

    # Add to dataframe at the beginning (most recent)
    new_df = pd.DataFrame([new_review])
    items_df = pd.concat([new_df, items_df], ignore_index=True)

    # Update search index with new review
    create_search_index()

    return jsonify({
        'success': True,
        'message': 'Review added successfully',
        'review_id': new_review_id,
        'item_id': item_id,
        'review_url': f'/api/review/{new_review_id}'
    })


@app.route('/api/review/<int:review_id>')
def get_review(review_id):
    """Get specific review by ID"""
    if 'Review ID' not in items_df.columns:
        return jsonify({'error': 'Review system not initialized'}), 404

    review_rows = items_df[items_df['Review ID'] == review_id]
    if review_rows.empty:
        return jsonify({'error': 'Review not found'}), 404

    review = review_rows.iloc[0]
    category = review['Class Name']

    return jsonify({
        'review_id': review_id,
        'item_id': int(review['Clothing ID']),
        'item_title': review['Clothes Title'],
        'category': category,
        'category_emoji': get_category_emoji(category),
        'review_title': review.get('Title', ''),
        'review_text': review.get('Review Text', ''),
        'rating': int(review.get('Rating', 0)),
        'recommended': int(review.get('Recommended IND', 0)),
        'recommended_text': 'Recommended' if review.get('Recommended IND', 0) == 1 else 'Not Recommended'
    })


if __name__ == '__main__':
    print("🚀 Starting Fashion Store Application...")
    load_model_and_data()
    app.run(debug=True, port=5000)