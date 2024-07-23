# NSE Stock Price Predictor

This project aims to predict stock prices using historical stock market data. It leverages various machine learning models to provide accurate predictions based on features such as moving averages.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Features](#features)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The Stock Price Predictor uses historical stock prices to predict future prices. The project includes data preprocessing, feature engineering, model training, and evaluation.

## Dataset

The dataset used is the "Indian Stock Market Master Data 24" from Kaggle, containing historical stock prices of major stocks in the Indian stock market.

## Installation

### Prerequisites

- Python 3.6+
- Pandas
- Numpy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

You can install the required packages using:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/stock-price-predictor.git
cd stock-price-predictor
```

### 2. Load the Dataset

Place the dataset in the `dataset` directory. Ensure the dataset file is named correctly, e.g., `ADANIENT.csv`.

### 3. Run Data Exploration and Preprocessing

Open the `notebooks` directory and run `EDA_Preprocessing.ipynb` to explore and preprocess the data.

### 4. Train the Model

Run the `Model_Training.ipynb` notebook to train the model. This will generate the trained model file `stock_price_model.pkl`.

### 5. Make Predictions

Use the `predict.py` script to make predictions on new data:

```bash
python predict.py --input path/to/new_data.csv --output path/to/save_predictions.csv
```

## Project Structure

```markdown
stock-price-predictor/
│
├── dataset/
│   └── ADANIENT.csv
│
├── notebooks/
│   ├── EDA_Preprocessing.ipynb
│   └── Model_Training.ipynb
│
├── scripts/
│   └── predict.py
│
├── stock_price_model.pkl
├── README.md
└── requirements.txt
```

## Features

- Data exploration and visualization
- Feature engineering (moving averages)
- Model training (Linear Regression)
- Model evaluation
- Prediction script for new data

## Model Training

The `Model_Training.ipynb` notebook covers:
- Splitting the data into training and testing sets
- Training a Linear Regression model
- Evaluating the model using metrics like MSE and MAE

## Evaluation

The model is evaluated based on:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Plotting actual vs. predicted prices

## Prediction

The `predict.py` script loads the trained model and makes predictions on new data. Ensure the new data has the same features (e.g., moving averages) as the training data.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License.
