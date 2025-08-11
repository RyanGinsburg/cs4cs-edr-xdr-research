# CS4CS - Evaluating_ML_Performance_in_EDR_and_XDR_Systems_Against_Common_Cyber_Threats

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

## Getting Started

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
```bash
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

## ⚙️ Configuration

The pipeline can be customized by modifying parameters in each script:

- **`adjust.py`**: MIN_CTX, MAX_CTX (negative sampling range)
- **`split.py`**: Train/test split ratios and temporal boundaries
- **`eval.py`**: ML model parameters and evaluation metrics


## ⚠️ Important Notes

- **Run scripts in order**: The pipeline must be executed sequentially as each script depends on outputs from the previous step
- This project contains simulated cyber attack scenarios for educational purposes
- The LANL dataset should be used responsibly and in accordance with data use agreements
- Results are based on simulated data and may not reflect real-world attack patterns
- Always follow ethical guidelines when conducting cybersecurity research
