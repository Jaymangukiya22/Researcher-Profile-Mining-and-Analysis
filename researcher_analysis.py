import pandas as pd
from scholarly import scholarly
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time
import os
import random
import re
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def get_output_filename():
    """Generate a unique output filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f'researcher_analysis_{timestamp}.xlsx'

def extract_user_id(url):
    """Extract user ID from Google Scholar profile URL."""
    match = re.search(r'user=([^&]+)', url)
    return match.group(1) if match else None

def get_researcher_publications(profile_url):
    """Fetch publications for a researcher from Google Scholar using their profile URL."""
    try:
        # Add random delay between requests
        time.sleep(random.uniform(2, 5))
        
        # Extract user ID from URL
        user_id = extract_user_id(profile_url)
        if not user_id:
            print(f"Could not extract user ID from URL: {profile_url}")
            return [], None
        
        # Get author profile using user ID
        author = scholarly.search_author_id(user_id)
        if not author:
            print(f"Could not find author with ID: {user_id}")
            return [], None
        
        # Get detailed author info
        author = scholarly.fill(author)
        
        # Get publications
        publications = []
        for pub in author['publications'][:20]:  # Get top 20 publications
            try:
                pub = scholarly.fill(pub)
                publications.append({
                    'title': pub['bib'].get('title', ''),
                    'abstract': pub['bib'].get('abstract', ''),
                    'year': pub['bib'].get('pub_year', '')
                })
                time.sleep(random.uniform(1, 3))  # Random delay between publications
            except Exception as e:
                print(f"Error fetching publication: {str(e)}")
                continue
        
        return publications, author['name']
    except Exception as e:
        print(f"Error fetching data from URL {profile_url}: {str(e)}")
        return [], None

def analyze_research_themes(abstracts):
    """Analyze research themes using TF-IDF and return formatted themes."""
    if not abstracts:
        return "No data available"
        
    # Combine all abstracts
    combined_text = ' '.join([abs for abs in abstracts if abs])
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=15, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([combined_text])
    
    # Get top keywords
    feature_names = vectorizer.get_feature_names_out()
    top_keywords = [feature_names[i] for i in tfidf_matrix.toarray()[0].argsort()[::-1][:15]]
    
    # Group related terms
    themes = []
    current_theme = []
    
    for keyword in top_keywords:
        if len(current_theme) < 3:  # Group 3 related terms
            current_theme.append(keyword)
        else:
            themes.append(', '.join(current_theme))
            current_theme = [keyword]
    
    if current_theme:
        themes.append(', '.join(current_theme))
    
    return '; '.join(themes)

def get_model_path():
    """
    Get the path to the local model or download it if not present
    """
    # Define the local model path
    model_path = os.path.join(os.path.expanduser("~"), ".cache", "torch", "sentence_transformers", "all-MiniLM-L6-v2")
    
    if not os.path.exists(model_path):
        print("Model not found locally. Please download it manually:")
        print("1. Visit: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2")
        print("2. Download the model files")
        print("3. Place them in:", model_path)
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    return model_path

def calculate_diversity_score(abstracts):
    """
    Calculate diversity score using sentence transformers with a smaller model
    """
    try:
        # Use a smaller model
        model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cpu')
        
        # Get embeddings for all abstracts
        embeddings = model.encode(abstracts)
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(similarity)
        
        # Calculate average similarity
        avg_similarity = np.mean(similarities) if similarities else 0
        
        # Determine diversity level
        if avg_similarity > 0.8:
            diversity = "Low Diversity"
        elif avg_similarity > 0.5:
            diversity = "Medium Diversity"
        else:
            diversity = "High Diversity"
        
        return avg_similarity, diversity
    except Exception as e:
        print(f"Error in calculate_diversity_score: {e}")
        return 0, 0

def generate_wordcloud(text, researcher_name):
    """Generate and save word cloud."""
    if not text.strip():
        return
        
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Research Themes - {researcher_name}')
    
    # Create wordclouds directory if it doesn't exist
    os.makedirs('wordclouds', exist_ok=True)
    
    # Save wordcloud with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'wordclouds/wordcloud_{researcher_name.replace(" ", "_")}_{timestamp}.png'
    plt.savefig(filename)
    plt.close()

def save_to_excel(publications_df, themes_df, diversity_df, output_file):
    """Save DataFrames to Excel file."""
    try:
        with pd.ExcelWriter(output_file) as writer:
            publications_df.to_excel(writer, sheet_name='Publications', index=False)
            themes_df.to_excel(writer, sheet_name='Research Themes', index=False)
            diversity_df.to_excel(writer, sheet_name='Diversity Analysis', index=False)
        print(f"\nAnalysis complete! Results saved to: {output_file}")
        return True
    except PermissionError:
        print(f"\nError: Could not save to {output_file}. Please make sure the file is not open in another program.")
        return False

def main():
    # List of Google Scholar profile URLs
    profile_urls = [
        'https://scholar.google.com/citations?user=2dyg3WgAAAAJ&hl=en',
        'https://scholar.google.com/citations?user=oZORQtwAAAAJ&hl=en',
        'https://scholar.google.com/citations?user=OF3eat8AAAAJ&hl=en',
        'https://scholar.google.com/citations?user=ERp8VTsAAAAJ&hl=en',
        'https://scholar.google.com/citations?user=Zk-f3UwAAAAJ&hl=zh-CN',
        'https://scholar.google.com/citations?user=Vhq5-s0AAAAJ&hl=en',
        'https://scholar.google.co.uk/citations?user=429MAoUAAAAJ&hl=en',
        'https://scholar.google.com/citations?user=vnVm31kAAAAJ&hl=en',
        'https://scholar.google.com/citations?user=8Xyqf_0AAAAJ&hl=en',
        'https://scholar.google.com/citations?user=xkH30GgAAAAJ&hl=en',
        'https://scholar.google.com/citations?user=d1YlKkAAAAAJ&hl=zh-CN',
        'https://scholar.google.co.in/citations?user=VR-M01wAAAAJ&hl=en',
        'https://scholar.google.com/citations?user=DFkVNJwAAAAJ&hl=en',
        'https://scholar.google.com/citations?user=KynAS2gAAAAJ&hl=en',
        'https://scholar.google.com/citations?user=lj388xkAAAAJ&hl=en',
        'https://scholar.google.co.kr/citations?user=UkdHqoYAAAAJ&hl=en',
        'https://scholar.google.com/citations?user=xuHi5v0AAAAJ&hl=en',
        'https://scholar.google.com/citations?user=iUSjQ5QAAAAJ&hl=en',
        'https://scholar.google.com/citations?user=wyUQBOIAAAAJ&hl=en'
    ]
    
    # Initialize DataFrames for different outputs
    publications_df = pd.DataFrame(columns=['S.No', 'Researcher Name', 'Title of the Paper', 'Abstract'])
    themes_df = pd.DataFrame(columns=['Researcher', 'Top Research Themes'])
    diversity_df = pd.DataFrame(columns=['Researcher', 'Average Similarity', 'Diversity Score'])
    
    # Process each researcher
    for profile_url in profile_urls:
        print(f"Processing profile: {profile_url}")
        
        # Get publications
        publications, researcher_name = get_researcher_publications(profile_url)
        
        if not publications or not researcher_name:
            print(f"No publications found for profile: {profile_url}")
            continue
        
        print(f"Found {len(publications)} publications for {researcher_name}")
        
        # Add publications to DataFrame
        for idx, pub in enumerate(publications, 1):
            publications_df = pd.concat([publications_df, pd.DataFrame({
                'S.No': [idx],
                'Researcher Name': [researcher_name],
                'Title of the Paper': [pub['title']],
                'Abstract': [pub['abstract']]
            })], ignore_index=True)
        
        # Analyze research themes
        abstracts = [pub['abstract'] for pub in publications if pub['abstract']]
        if abstracts:
            # Generate word cloud
            combined_text = ' '.join(abstracts)
            generate_wordcloud(combined_text, researcher_name)
            
            # Get research themes
            themes = analyze_research_themes(abstracts)
            themes_df = pd.concat([themes_df, pd.DataFrame({
                'Researcher': [researcher_name],
                'Top Research Themes': [themes]
            })], ignore_index=True)
            
            # Calculate diversity score
            avg_similarity, diversity = calculate_diversity_score(abstracts)
            diversity_df = pd.concat([diversity_df, pd.DataFrame({
                'Researcher': [researcher_name],
                'Average Similarity': [avg_similarity],
                'Diversity Score': [diversity]
            })], ignore_index=True)
        
        print(f"Completed processing {researcher_name}")
    
    # Try to save with timestamped filename
    output_file = get_output_filename()
    if not save_to_excel(publications_df, themes_df, diversity_df, output_file):
        # If failed, try with backup filename
        backup_file = f'researcher_analysis_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        if not save_to_excel(publications_df, themes_df, diversity_df, backup_file):
            print("\nError: Could not save results. Please close any open Excel files and try again.")

if __name__ == "__main__":
    main() 