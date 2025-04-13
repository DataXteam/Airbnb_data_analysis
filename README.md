# Prague Airbnb Price Prediction Project

This project aims to predict Airbnb prices per night in Prague based on the Inside Airbnb dataset.

## Project Structure

```
├── data/
│   ├── interim/  # Intermediate data files
│   ├── processed/ # Final cleaned data ready for modeling
│   └── raw/      # Original downloaded data (place downloaded files here)
├── model/        # Saved model files (.joblib, etc.)
├── notebooks/    # Jupyter notebooks for analysis (e.g., EDA, modeling)
├── Report/       # Project report (PDF) and presentation slides
├── streamlit_app/ # Optional Streamlit application code
├── .gitignore    # Specifies intentionally untracked files
├── README.md     # This file
├── requirements.txt # Project dependencies
└── venv/         # Python virtual environment (created locally)
```

## Setup

1. Clone the repository:
   `git clone `
   `cd `
2. Create and activate the virtual environment:
   `python3 -m venv venv`
   `source venv/bin/activate`
3. Install dependencies:
   `uv pip install -r requirements.txt`
4. Download data from http://insideairbnb.com/get-the-data/ (Prague) and place `listings.csv.gz`, `calendar.csv.gz`, `reviews.csv.gz` into the `data/raw/` directory. Unzip them.

## Usage

- Explore the notebooks in the `notebooks/` directory.
- (Optional) Run the Streamlit app: `streamlit run streamlit_app/app.py`
