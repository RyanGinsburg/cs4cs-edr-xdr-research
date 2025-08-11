# CS4CS - Evaluating ML Performance in EDR and XDR Systems Against Common Cyber Threats

This repository contains a comprehensive cybersecurity analysis project focused on evaluating machine learning performance in EDR (Endpoint Detection and Response) and XDR (Extended Detection and Response) systems against common cyber threats.

## 📋 Project Overview

The main project analyzes the effectiveness of various machine learning models in detecting cyber threats using the LANL (Los Alamos National Laboratory) Comprehensive, Multi-Source Cyber-Security Events dataset. The analysis builds richer EDR and XDR datasets and evaluates multiple ML algorithms for threat detection.

## 🏗️ Project Structure

### Main Analysis Files
- **`lanl.py`** - Core dataset builder that processes LANL data to create EDR and XDR feature sets
- **`sort.py`** - Sorts and filters the master datasets for processing
- **`adjust.py`** - Adjusts dataset balance by sampling negative examples around positive events
- **`split.py`** - Creates balanced train/test splits from adjusted datasets
- **`eval.py`** - Machine learning evaluation pipeline with multiple algorithms
- **`results.html`** - Generated HTML report with analysis results
- **`styles.css`** - Styling for the HTML report

### Classwork Directory
Contains various educational exercises and assignments:

#### Reverse Engineering
- **`hello_world.c`** - Basic C program
- **`crackme.c`** - Password cracking challenge
- **`crackme.py`** - Python pattern printing exercises
- **`ReverseEngineering_Worksheet.txt`** - Educational worksheet

#### Other Exercises
- **`calculate.py`** - Mathematical calculations
- **`sherlock.py`** - Text analysis exercise
- **`test_2.txt`** - Linux CLI practice file

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/CS4CS.git
cd CS4CS
```

2. **Install dependencies:**
```bash\
cd project
pip install -r requirements.txt
```

### Data Requirements
The project expects LANL dataset files in a `lanl/` directory:
- `auth.txt.gz` - Authentication events
- `proc.txt.gz` - Process events  
- `flows.txt.gz` - Network flow data
- `dns.txt.gz` - DNS queries
- `redteam.txt.gz` - Attack labels

### Running the Complete Analysis Pipeline

**Important:** The analysis must be run in the following order:

1. **Build the initial datasets:**
```bash
cd project
python lanl.py
```
This creates the raw master datasets with EDR and XDR features.

2. **Sort and filter the datasets:**
```bash
python sort.py
```
This creates sorted and filtered versions of the master datasets.

3. **Adjust dataset balance:**
```bash
python adjust.py
```
This samples negative examples around positive events to create balanced datasets for training.

4. **Create train/test splits:**
```bash
python split.py
```
This creates final train/test splits from the adjusted datasets.

5. **Run ML evaluation:**
```bash
python eval.py
```
This generates `results.html` with comprehensive analysis results.

6. **View results:**
Open `results.html` in your web browser to see the complete analysis report.

## 🔄 Pipeline Overview

The analysis pipeline follows this data flow:

```
Raw LANL Data → lanl.py → Master Datasets
                    ↓
              sort.py → Sorted & Filtered Datasets  
                    ↓
             adjust.py → Balanced Datasets
                    ↓
              split.py → Train/Test Splits
                    ↓
               eval.py → ML Analysis & Results
```

## 🔍 Analysis Features

### Dataset Construction
- **Time-windowed features** (5-minute windows by default)
- **EDR features** from host logs (authentication, processes)
- **XDR features** combining EDR with network context (flows, DNS)
- **Contextual sampling** around malicious events (75-150 negative samples per side)
- **Temporal train/test splits** avoiding data leakage

### Machine Learning Models
- LightGBM Classifier
- Balanced Random Forest
- SGD Classifier  
- Isolation Forest (anomaly detection)

### Evaluation Metrics
- Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Confusion matrices
- Feature importance analysis

## 📊 Key Findings

The analysis reveals significant class imbalance challenges in cybersecurity datasets, with malicious samples representing less than 1% of the data. The project demonstrates various techniques for handling this imbalance and evaluating model performance in realistic security scenarios.

## 📁 Output Structure

```
project/
├── lanl_output/
│   ├── edr_master.csv                    # Initial EDR dataset
│   ├── xdr_master.csv                    # Initial XDR dataset
│   ├── edr_master_sorted_filtered.csv    # Sorted EDR dataset
│   ├── xdr_master_sorted_filtered.csv    # Sorted XDR dataset
│   ├── adjusted_master_edr.csv           # Balanced EDR dataset
│   ├── adjusted_master_xdr.csv           # Balanced XDR dataset
│   ├── edr_train_all.csv                 # EDR training data
│   ├── edr_test_all.csv                  # EDR test data
│   ├── xdr_train_all.csv                 # XDR training data
│   └── xdr_test_all.csv                  # XDR test data
├── results.html                          # Analysis report
└── styles.css                            # Report styling
```

## 📄 Documentation

For detailed methodology and findings, refer to:
- `Evaluating_ML_Performance_in_EDR_and_XDR_Systems_Against_Common_Cyber_Threats.pdf`
- Generated `results.html` report

## 🎓 Educational Components

The `classwork/` directory contains various cybersecurity and programming exercises suitable for learning:
- Reverse engineering fundamentals
- Binary analysis techniques
- Python programming exercises
- Linux command line practice

## 🔧 Dependencies

Key dependencies include:
- **pandas** (2.3.1) - Data manipulation and analysis
- **numpy** (2.3.1) - Numerical computing
- **scikit-learn** (1.7.1) - Machine learning library
- **matplotlib** (3.10.3) - Data visualization
- **seaborn** (0.13.2) - Statistical data visualization
- **kagglehub** (0.3.12) - Dataset downloading

See `requirements.txt` for complete dependency list.

## ⚙️ Configuration

The pipeline can be customized by modifying parameters in each script:

- **`adjust.py`**: MIN_CTX, MAX_CTX (negative sampling range)
- **`split.py`**: Train/test split ratios and temporal boundaries
- **`eval.py`**: ML model parameters and evaluation metrics

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is for educational and research purposes. Please ensure appropriate data use agreements when working with the LANL dataset.

## ⚠️ Important Notes

- **Run scripts in order**: The pipeline must be executed sequentially as each script depends on outputs from the previous step
- This project contains simulated cyber attack scenarios for educational purposes
- The LANL dataset should be used responsibly and in accordance with data use agreements
- Results are based on simulated data and may not reflect real-world attack patterns
- Always follow ethical guidelines when conducting cybersecurity research

## 🐛 Troubleshooting

**Common Issues:**
- **File not found errors**: Ensure you run scripts in the correct order
- **Memory issues**: The LANL dataset is large; ensure sufficient RAM
- **Missing dependencies**: Run `pip install -r requirements.txt`

## 📞 Support

If you encounter any issues or have questions about the project, please:
1. Check that you've run all pipeline steps in order
2. Review the documentation in the PDF file
3. Check the generated `results.html` report
4. Open an issue on GitHub with detailed information about your problem

---

**Note:** This project is part of a Computer Science Security course and is intended for
