## ⚙️ How It Works

### 🧬 Data Generation
A synthetic dataset of 2000 samples is created with:
- Normally distributed **height**, **weight**, and **voice pitch**
- **BMI** calculated from height and weight
- **Gender** and **age group** labels with demographic adjustments

### 🧠 Feature Engineering
- Polynomial features up to **degree 2**
- **Height-to-weight ratio** and **pitch-to-height ratio** are computed and added to the feature set

### 🎯 Model Training
- Two separate `RandomForestClassifier` models are trained:
  - One for **gender prediction**
  - One for **age group prediction**

### 🧑‍💻 User Input & Prediction
- Users input their **height**, **weight**, and **voice pitch**
- The input is **scaled** and **transformed** using the same preprocessing as training data
- The model:
  - Predicts **gender** and **age group**
  - Provides **confidence scores** for each prediction
