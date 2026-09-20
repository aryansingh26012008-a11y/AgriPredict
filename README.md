# AgriPredict
AgriPredict.ai is a machine learning web application that recommends the most suitable crop using soil nutrients, temperature, humidity, pH, and rainfall. Built with Python, Flask, Pandas, and Scikit-learn, it delivers fast, data-driven insights through a modern interface.
# AgriPredict.ai

### Intelligent Crop Recommendation Using Machine Learning

AgriPredict.ai is a Flask-based machine learning web application that recommends the most suitable crop for cultivation based on soil nutrients and environmental conditions.

The system analyzes seven important agricultural parameters:

- Nitrogen level
- Phosphorus level
- Potassium level
- Temperature
- Humidity
- Soil pH
- Rainfall

Using a trained Random Forest classification model, the application processes the entered values and predicts the crop most likely to grow successfully in those conditions.

## Features

- Clean and responsive agricultural dashboard
- Machine learning-based crop recommendations
- Interactive input form
- Real-time prediction results
- Trained Random Forest model
- Soil and weather parameter analysis
- Flask-powered web interface
- Modern dark glassmorphism design
- Mobile-friendly layout

## How It Works

1. The user enters soil and environmental measurements.
2. The Flask application receives the form data.
3. The input values are converted into a structured dataset.
4. The trained machine learning pipeline scales the values.
5. The Random Forest classifier predicts the recommended crop.
6. The result is displayed through the web interface.

## Machine Learning Model

The model is trained using the `Crop_recommendation.csv` dataset. It uses:

- Min-Max Scaling for feature normalization
- Random Forest Classification for prediction
- Train-test splitting for model evaluation
- Scikit-learn Pipeline for organized preprocessing and prediction

The trained model is saved as `model.pkl` and loaded by the Flask application during startup.

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- HTML5
- CSS3
- Bootstrap 5
- Font Awesome

## Project Structure

```text
AgriPredict.ai/
│
├── app.py
├── crop.py
├── model.pkl
├── Crop_recommendation.csv
├── index.html
├── style.css
├── img.jpg
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open the application in your browser:

```text
http://127.0.0.1:5000
```

## Example Use Case

A farmer or agricultural advisor can enter the nutrient and weather conditions of a particular field. The application analyzes the data and provides a crop recommendation, helping users make more informed cultivation decisions.

## Future Improvements

- Add crop-wise cultivation guidance
- Include fertilizer recommendations
- Add weather API integration
- Store prediction history
- Improve model accuracy with larger datasets
- Add user authentication
- Deploy the application online
- Add charts for soil and climate analysis

## Important Note

This project is intended for educational and experimental purposes. The recommendations should be considered supportive insights and should be verified with agricultural experts and local farming knowledge before making real-world cultivation decisions.
