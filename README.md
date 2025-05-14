# Researcher-Profile-Mining-and-Analysis
Deep Learning subject case study

# Researcher Analysis Tool

This tool automates the process of analyzing researchers' publications and research themes using Google Scholar data.

## Features

- Fetches researcher profiles from Google Scholar
- Extracts top 20 publications for each researcher
- Generates word clouds of research themes
- Calculates research diversity scores
- Creates detailed Excel reports

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
   python researcher_analysis.py
   ```

4. The script will generate:
   - `researcher_analysis.xlsx`: Contains detailed analysis for each researcher
   - Word cloud images for each researcher
   - A summary sheet with diversity scores

## Output

The `researcher_analysis.xlsx` file contains:
- One sheet per researcher with their publications
- A summary sheet with diversity scores
- Each researcher's sheet includes:
  - S.No
  - Researcher Name
  - Title of the Paper
  - Abstract

## Notes

- The script includes delays to respect Google Scholar's rate limits
- Some researchers might not be found or might have incomplete data
- The script will skip researchers that can't be found and continue with others 