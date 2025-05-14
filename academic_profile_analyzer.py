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

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class AcademicProfileAnalyzer:
    def __init__(self):
        self.publications_data = pd.DataFrame(columns=['ID', 'Scholar Name', 'Paper Title', 'Abstract'])
        self.research_themes_data = pd.DataFrame(columns=['Scholar', 'Key Research Areas'])
        self.diversity_metrics = pd.DataFrame(columns=['Scholar', 'Similarity Index', 'Diversity Level'])
        
    def generate_timestamped_filename(self):
        """Create a unique filename with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f'academic_analysis_{timestamp}.xlsx'
    
    def extract_scholar_id(self, profile_url):
        """Extract scholar ID from Google Scholar URL."""
        pattern = re.search(r'user=([^&]+)', profile_url)
        return pattern.group(1) if pattern else None
    
    def fetch_scholar_publications(self, profile_url):
        """Retrieve publications for a scholar from Google Scholar."""
        try:
            time.sleep(random.uniform(2, 5))
            
            scholar_id = self.extract_scholar_id(profile_url)
            if not scholar_id:
                print(f"Invalid scholar ID in URL: {profile_url}")
                return [], None
            
            scholar_profile = scholarly.search_author_id(scholar_id)
            if not scholar_profile:
                print(f"Scholar not found: {scholar_id}")
                return [], None
            
            scholar_profile = scholarly.fill(scholar_profile)
            
            publications = []
            for pub in scholar_profile['publications'][:20]:
                try:
                    pub = scholarly.fill(pub)
                    publications.append({
                        'title': pub['bib'].get('title', ''),
                        'abstract': pub['bib'].get('abstract', ''),
                        'year': pub['bib'].get('pub_year', '')
                    })
                    time.sleep(random.uniform(1, 3))
                except Exception as e:
                    print(f"Error retrieving publication: {str(e)}")
                    continue
            
            return publications, scholar_profile['name']
        except Exception as e:
            print(f"Error processing URL {profile_url}: {str(e)}")
            return [], None

    def identify_research_areas(self, abstracts):
        """Identify key research areas using TF-IDF analysis."""
        if not abstracts:
            return "No data available"
            
        combined_text = ' '.join([abs for abs in abstracts if abs])
        
        vectorizer = TfidfVectorizer(max_features=15, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([combined_text])
        
        keywords = vectorizer.get_feature_names_out()
        top_keywords = [keywords[i] for i in tfidf_matrix.toarray()[0].argsort()[::-1][:15]]
        
        research_areas = []
        current_area = []
        
        for keyword in top_keywords:
            if len(current_area) < 3:
                current_area.append(keyword)
            else:
                research_areas.append(', '.join(current_area))
                current_area = [keyword]
        
        if current_area:
            research_areas.append(', '.join(current_area))
        
        return '; '.join(research_areas)

    def compute_research_diversity(self, abstracts):
        """Calculate research diversity using sentence embeddings."""
        try:
            model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cpu')
            
            embeddings = model.encode(abstracts)
            
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append(similarity)
            
            avg_similarity = np.mean(similarities) if similarities else 0
            
            if avg_similarity > 0.8:
                diversity = "Low Diversity"
            elif avg_similarity > 0.5:
                diversity = "Medium Diversity"
            else:
                diversity = "High Diversity"
            
            return avg_similarity, diversity
        except Exception as e:
            print(f"Error in diversity calculation: {e}")
            return 0, 0

    def create_research_visualization(self, text, scholar_name):
        """Generate and save research theme visualization."""
        if not text.strip():
            return
            
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Research Areas - {scholar_name}')
        
        os.makedirs('research_visualizations', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'research_visualizations/visualization_{scholar_name.replace(" ", "_")}_{timestamp}.png'
        plt.savefig(filename)
        plt.close()

    def save_analysis_results(self, output_file):
        """Save analysis results to Excel file."""
        try:
            with pd.ExcelWriter(output_file) as writer:
                self.publications_data.to_excel(writer, sheet_name='Publications', index=False)
                self.research_themes_data.to_excel(writer, sheet_name='Research Areas', index=False)
                self.diversity_metrics.to_excel(writer, sheet_name='Diversity Analysis', index=False)
            print(f"\nAnalysis complete! Results saved to: {output_file}")
            return True
        except PermissionError:
            print(f"\nError: Cannot save to {output_file}. Please ensure the file is not open.")
            return False

    def analyze_scholar_profiles(self):
        """Main method to analyze scholar profiles."""
        scholar_urls = [
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
        
        for profile_url in scholar_urls:
            print(f"Analyzing profile: {profile_url}")
            
            publications, scholar_name = self.fetch_scholar_publications(profile_url)
            
            if not publications or not scholar_name:
                print(f"No publications found for: {profile_url}")
                continue
            
            print(f"Processing {len(publications)} publications for {scholar_name}")
            
            for idx, pub in enumerate(publications, 1):
                self.publications_data = pd.concat([self.publications_data, pd.DataFrame({
                    'ID': [idx],
                    'Scholar Name': [scholar_name],
                    'Paper Title': [pub['title']],
                    'Abstract': [pub['abstract']]
                })], ignore_index=True)
            
            abstracts = [pub['abstract'] for pub in publications if pub['abstract']]
            if abstracts:
                combined_text = ' '.join(abstracts)
                self.create_research_visualization(combined_text, scholar_name)
                
                research_areas = self.identify_research_areas(abstracts)
                self.research_themes_data = pd.concat([self.research_themes_data, pd.DataFrame({
                    'Scholar': [scholar_name],
                    'Key Research Areas': [research_areas]
                })], ignore_index=True)
                
                similarity, diversity = self.compute_research_diversity(abstracts)
                self.diversity_metrics = pd.concat([self.diversity_metrics, pd.DataFrame({
                    'Scholar': [scholar_name],
                    'Similarity Index': [similarity],
                    'Diversity Level': [diversity]
                })], ignore_index=True)
            
            print(f"Completed analysis for {scholar_name}")
        
        output_file = self.generate_timestamped_filename()
        if not self.save_analysis_results(output_file):
            backup_file = f'academic_analysis_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
            if not self.save_analysis_results(backup_file):
                print("\nError: Could not save results. Please close any open Excel files and try again.")

if __name__ == "__main__":
    analyzer = AcademicProfileAnalyzer()
    analyzer.analyze_scholar_profiles() 