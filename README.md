This is the Group Project Big Data Visualization in the lambton College

# Stroke Prediction App

## Description

This application predicts the likelihood of a stroke based on various input features such as age, hypertension, heart disease, and more. The prediction is generated using five machine learning models: Logistic Regression, Naive Bayes, SVM, Decision Tree, and Random Forest. The application also includes interpretability features using **LIME** to explain the predictions made by each model.

## Features
- Interactive form to input patient data for stroke prediction.
- Five machine learning models: Logistic Regression, Naive Bayes, SVM, Decision Tree, and Random Forest.
- **LIME** interpretability to explain predictions for each model.
- Embedded Power BI dashboard for additional data visualization.

## Project Structure
```
├── app.py                        # Main Streamlit app
├── models/                       # Pre-trained machine learning models
│   ├── Logistic_Regression.pkl   # Logistic Regression model
│   ├── Naive_Bayes.pkl           # Naive Bayes model
│   ├── SVM.pkl                   # SVM model
│   ├── Decision_Tree.pkl         # Decision Tree model
│   └── Random_Forest.pkl         # Random Forest model
├── data/                         # Dataset folder
│   └── train_2v-clean.csv        # Cleaned dataset
├── requirements.txt              # List of dependencies
├── README.md                     # Project documentation
```

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```bash
   cd project-directory
   ```

3. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # Activate the environment
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

Once the application is running, you can:
- Enter patient details such as age, hypertension status, heart disease, etc.
- View predictions from five different machine learning models.
- Explore model interpretability with **LIME** to understand which factors contributed to each model's prediction.
- Interact with the embedded Power BI dashboard for additional insights.

## Dependencies

- pandas
- numpy
- matplotlib
- tensorflow
- lime
- streamlit

All required dependencies are listed in the `requirements.txt` file. You can install them with:
```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgements

- [LIME: Local Interpretable Model-agnostic Explanations](https://github.com/marcotcr/lime)
- [Streamlit](https://www.streamlit.io/)
- Data used for training sourced from Kaggle
