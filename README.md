# Gender and Age Detection using Synthetic Data

This project is a machine learning-based application that predicts a person's gender (Male/Female) and age group (Child/Adult/Senior) based on their height, weight, voice pitch, and BMI. It is trained entirely on synthetically generated data to simulate demographic effects, using Random Forest classifiers and enhanced feature engineering.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Project Highlights](#-project-highlights)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Technologies Used](#technologies-used)
- [Usage](#usage)
- [Results Sample](#-results-sample)
- [Contributing](#contributing)
- [License](#license)

---

## 🔍 Overview

The project demonstrates how machine learning models can be effectively trained on synthetic datasets to predict real-world features. It simulates demographic traits and their statistical dependencies, like:

- Gender differences in physical traits  
- Age-related changes in body and voice  
- Noise factors mimicking real-world data variance  

Users interact through a simple terminal-based CLI where they input their measurements, and the model returns a prediction with confidence levels.

---

## 🚀 Project Highlights

✅ Uses only basic Python libraries (no OpenCV, TensorFlow, or Keras)  
✅ Fully synthetic data generation – no need for real datasets  
✅ Interactive CLI for real-time predictions  
✅ Feature engineering using polynomial combinations and custom ratios  
✅ Trained using Scikit-learn's RandomForestClassifier

---

## 📸 Results Sample

### Input CLI
![Input CLI](image/Screenshot%202025-05-28%20121746.png)

![Input CLI](image/Screenshot%202025-05-28%20121757.png)

![Input CLI](image/Screenshot%202025-05-28%20121809.png)

### Processed Results Display
![Summary](image/Screenshot%202025-05-28%20121841.png)

---

## 💾 Installation

```bash
git clone https://github.com/chaudhary-pawan/Gender-Age_detection
cd Gender-Age_detection
pip install -r requirements.txt
