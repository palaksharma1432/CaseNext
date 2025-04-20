# CaseNext
## Court Case Prioritization System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6%2B-yellow)

A machine learning system that predicts priority scores for court cases to optimize judicial scheduling and reduce backlog.

## ✨ Features

### CaseNext (Single Case Mode)
- Interactive command-line interface for single case prediction
- Processes individual cases with detailed feature input
- Displays prediction confidence levels
- Supports multiple predictions in one session

### CaseNext2 (Bulk Processing Mode)
- Processes entire datasets in CSV/Excel format
- Ranks cases by predicted priority score
- Options to view top 10/50/100 or all ranked cases
- Save results to CSV functionality
- Detailed model evaluation metrics

## 🛠️ Technical Implementation

### Machine Learning Models
- **Random Forest Classifier**
- **Logistic Regression**
- Ensemble approach combining both models' predictions

### Feature Processing
- Advanced data cleaning and validation
- Automated feature engineering
- Categorical feature encoding (OneHotEncoder)
- Numerical feature scaling (StandardScaler)

## 📂 Project Structure

```
court-case-prioritization/
├── CaseNext/                  # Single case prediction system <b>
│   ├── CaseNext(single case).py  # Main prediction script
│   └── CaseNext_rule.docx     # Format requirements
├── CaseNext2/                 # Bulk processing system
│   ├── CaseNext2.py           # Main processing script
│   └── sample_data/           # Example datasets
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/court-case-prioritization.git
cd court-case-prioritization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

**For single case predictions:**
```bash
python CaseNext/CaseNext(single\ case).py
```

**For bulk processing:**
```bash
python CaseNext2/CaseNext2.py
```

## 📊 Model Performance

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Random Forest       | 0.92     | 0.91      | 0.90   | 0.90     |
| Logistic Regression | 0.89     | 0.88      | 0.87   | 0.87     |
| Ensemble            | 0.93     | 0.92      | 0.91   | 0.91     |


## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or suggestions, please contact:  
Palak Sharma - 22cse079@gweca.ac.in <br>
Project Link: https://github.com/palaksharma1432/CaseNext <br>
linkedin: https://www.linkedin.com/in/palak-sharma-4799672b1/ <br>
```


