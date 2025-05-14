# Researcher-Profile-Mining-and-Analysis
Deep Learning subject case study

This tool automates the process of analyzing researchers' publications and research themes using Google Scholar data. It provides comprehensive analysis of academic profiles, research areas, and publication diversity.

## Features

- Fetches researcher profiles from Google Scholar
- Extracts top 20 publications for each researcher
- Generates word clouds of research themes
- Calculates research diversity scores
- Creates detailed Excel reports with multiple analysis sheets
- Object-oriented design for better maintainability and extensibility

## Setup

1. Install Python 3.8 or higher
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. First, create the input template:
   ```bash
   python create_input_template.py
   ```

2. Edit the generated `researchers.xlsx` file:
   - Replace the example names with actual researcher names
   - Save the file

3. Run the analysis:
   ```bash
   python academic_profile_analyzer.py
   ```

4. The script will generate:
   - `academic_analysis_[timestamp].xlsx`: Contains detailed analysis for each researcher
   - Research visualization images in the `research_visualizations` directory
   - A comprehensive analysis with multiple sheets

## Output

The `academic_analysis_[timestamp].xlsx` file contains three main sheets:

1. **Publications Sheet**
   - ID
   - Scholar Name
   - Paper Title
   - Abstract

2. **Research Areas Sheet**
   - Scholar
   - Key Research Areas (identified through TF-IDF analysis)

3. **Diversity Analysis Sheet**
   - Scholar
   - Similarity Index
   - Diversity Level (High/Medium/Low)

## Research Visualizations

- Word cloud images are saved in the `research_visualizations` directory
- Each visualization is named with the scholar's name and timestamp
- Images show the most prominent research themes and keywords

## Technical Details

- Uses TF-IDF vectorization for research theme identification
- Implements sentence transformers for diversity analysis
- Includes rate limiting and error handling for Google Scholar API
- Object-oriented design with the `AcademicProfileAnalyzer` class
- Automatic backup file generation if primary save fails

## Notes

- The script includes random delays to respect Google Scholar's rate limits
- Some researchers might not be found or might have incomplete data
- The script will skip researchers that can't be found and continue with others
- All output files include timestamps to prevent overwriting
- The analysis is performed using a smaller model for better performance

## Error Handling

- Graceful handling of API rate limits
- Automatic retry mechanisms for failed requests
- Backup file generation if primary save fails
- Detailed error logging for troubleshooting 