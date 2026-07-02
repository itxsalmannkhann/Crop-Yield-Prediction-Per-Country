<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Python][python-shield]][python-url]
[![Scikit-Learn][sklearn-shield]][sklearn-url]
[![Pandas][pandas-shield]][pandas-url]
[![NumPy][numpy-shield]][numpy-url]
[![Jupyter][jupyter-shield]][jupyter-url]
[![License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <img src="https://img.icons8.com/color/96/wheat.png" alt="Logo" width="80" height="80">
  <h3 align="center">AgriYield Predictor</h3>

  <p align="center">
    A supervised machine learning regression model that predicts crop yield (hg/ha) for a given country, crop, year, rainfall, pesticide usage, and average temperature.
    <br />
    <br />
    <br />
    <a href="#usage">View Usage</a>
    &middot;
    <a href="#results">View Results</a>
    &middot;
    <a href="#contact">Contact</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#dataset">Dataset</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#1-exploratory-data-analysis-eda">Exploratory Data Analysis</a></li>
        <li><a href="#2-data-preprocessing">Data Preprocessing</a></li>
        <li><a href="#3-model-training--comparison">Model Training & Comparison</a></li>
        <li><a href="#4-making-predictions">Making Predictions</a></li>
      </ul>
    </li>
    <li><a href="#results">Results</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

**Crop Yield Prediction Per Country** is a supervised machine learning project that estimates crop yield, measured in **hg/ha** (hectograms per hectare), based on six input features: the **Year**, **Area** (country), crop **Item**, **average rainfall**, **pesticide usage**, and **average temperature**.

Since the target variable (`hg/ha_yield`) is numerical, this is framed as a **regression problem**. The notebook walks through the full machine learning pipeline:

* Cleaning and exploring a real-world agricultural dataset spanning **101 countries** and **10 crop types**
* Encoding categorical features and scaling numerical features with a single reusable `ColumnTransformer`
* Training and benchmarking five regression algorithms
* Selecting the best-performing model and wrapping it in a simple predictive function
* Serializing the final model and preprocessor with `pickle` so they can be reused without retraining

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Python][Python-badge]][Python-url]
* [![Jupyter][Jupyter-badge]][Jupyter-url]
* [![Pandas][Pandas-badge]][Pandas-url]
* [![NumPy][NumPy-badge]][NumPy-url]
* [![scikit-learn][sklearn-badge]][sklearn-url]
* [![Matplotlib][Matplotlib-badge]][Matplotlib-url]
* [![Seaborn][Seaborn-badge]][Seaborn-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DATASET -->
## Dataset

The project uses `yield_df.csv`, a global crop yield dataset with the following raw columns:

| Column | Description |
|---|---|
| `Area` | Country name |
| `Item` | Crop type (e.g. Maize, Rice, Wheat) |
| `Year` | Year of record |
| `hg/ha_yield` | **Target** — yield in hectograms per hectare |
| `average_rain_fall_mm_per_year` | Average annual rainfall (mm) |
| `pesticides_tonnes` | Pesticide usage (tonnes) |
| `avg_temp` | Average temperature (°C) |

**Cleaning steps applied:**

| Step | Result |
|---|---|
| Raw dataset shape | 28,242 rows × 7 columns |
| Dropped unnecessary `Unnamed: 0` index column | — |
| Missing values | None found |
| Duplicate rows removed | 2,310 rows dropped |
| Non-numeric values in `average_rain_fall_mm_per_year` removed | Rows with string/invalid values dropped |
| **Final clean dataset** | **25,932 rows** |
| Unique countries (`Area`) | 101 |
| Unique crop types (`Item`) | 10 |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

Follow these steps to run the notebook locally.

### Prerequisites

* Python 3.9+
* Jupyter Notebook or JupyterLab
* `yield_df.csv` placed in the same directory as the notebook

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/your_username/crop-yield-prediction.git
   cd crop-yield-prediction
   ```
2. (Optional) Create and activate a virtual environment
   ```sh
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. Install the required packages
   ```sh
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```
4. Launch Jupyter and open the notebook
   ```sh
   jupyter notebook crop_yield_prediction.ipynb
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

Run the notebook cells from top to bottom. The workflow is organized into four stages:

### 1. Exploratory Data Analysis (EDA)
* Inspects shape, data types, missing values, and duplicates
* Fixes the mixed-type `average_rain_fall_mm_per_year` column
* Visualizes total yield by country and by crop using Seaborn bar plots

### 2. Data Preprocessing
* Splits data into input features (`x`) and target (`y`)
* Performs an 80/20 train-test split (`random_state = 42`)
* Builds a single `ColumnTransformer` that applies:
  * `OneHotEncoder(drop='first')` to the categorical columns (`Area`, `Item`)
  * `StandardScaler()` to the numerical columns (`Year`, rainfall, pesticides, temperature)

### 3. Model Training & Comparison
Five regression models are trained and evaluated on the same train/test split:

```python
models = {
    'lr'  : LinearRegression(),
    'lss' : Lasso(),
    'rg'  : Ridge(),
    'knr' : KNeighborsRegressor(),
    'dtr' : DecisionTreeRegressor()
}
```

### 4. Making Predictions
A `prediction()` helper function takes the six raw inputs, runs them through the saved preprocessor, and returns the predicted yield from the trained K-Neighbors Regressor:

```python
prediction(Year=1990,
           average_rain_fall_mm_per_year=1485.0,
           pesticides_tonnes=121.0,
           avg_temp=16.37,
           Area='Albania',
           Item='Maize')
# => Predicted yield ≈ 28,118 hg/ha
```

The final model and preprocessor are saved as `knr.pkl` and `preprocessor.pkl` for reuse without retraining.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- RESULTS -->
## Results

All five models were evaluated using **Mean Squared Error (MSE)** and **R² Score** on the held-out test set:

| Model | MSE | R² Score |
|---|---|---|
| Linear Regression | 1,821,769,989.67 | 0.7486 |
| Lasso Regression | 1,822,352,597.41 | 0.7486 |
| Ridge Regression | 1,822,629,015.12 | 0.7485 |
| **K-Neighbors Regressor** | **124,881,646.98** | **0.9828** |
| Decision Tree Regressor | 469,772,132.27 | 0.9352 |

**Selected model:** `KNeighborsRegressor` — it produced the lowest MSE and the highest R² score among all candidates, making it the most accurate model for this dataset.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- PROJECT STRUCTURE -->
## Project Structure

```
crop-yield-prediction/
├── crop_yield_prediction.ipynb   # Main notebook: EDA, preprocessing, training, prediction
├── yield_df.csv                  # Raw dataset
├── knr.pkl                       # Serialized trained KNeighborsRegressor model
├── preprocessor.pkl              # Serialized ColumnTransformer (encoder + scaler)
└── README.md
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Data cleaning & EDA
- [x] Feature encoding & scaling pipeline
- [x] Multi-model training & comparison
- [x] Predictive function + model serialization
- [ ] Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- [ ] Cross-validation for more robust model evaluation
- [ ] Deploy as an interactive web app for live predictions
- [ ] Add more recent years of data

See the [open issues](https://github.com/itxsalmannkhann/crop-yield-prediction/issues) for a full list of proposed features.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are welcome and appreciated.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Salman Khan — AI/ML/DL Engineer — Abdul Wali Khan University Mardan — khanhackersalman@gmail.com

Project Link: https://github.com/itxsalmannkhann/Crop-Yield-Prediction-Per-Country/

Project Live Demo: https://agriyieldpredictor.streamlit.app/

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [scikit-learn Documentation](https://scikit-learn.org/stable/)
* [Pandas Documentation](https://pandas.pydata.org/docs/)
* [Seaborn Documentation](https://seaborn.pydata.org/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[python-shield]: https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[Python-badge]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[sklearn-shield]: https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white
[sklearn-url]: https://scikit-learn.org/
[sklearn-badge]: https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white
[pandas-shield]: https://img.shields.io/badge/Pandas-2.3.3-150458?style=for-the-badge&logo=pandas&logoColor=white
[pandas-url]: https://pandas.pydata.org/
[Pandas-badge]: https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white
[numpy-shield]: https://img.shields.io/badge/NumPy-2.3.5-013243?style=for-the-badge&logo=numpy&logoColor=white
[numpy-url]: https://numpy.org/
[NumPy-badge]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[jupyter-shield]: https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white
[jupyter-url]: https://jupyter.org/
[Jupyter-badge]: https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white
[Matplotlib-badge]: https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white
[Matplotlib-url]: https://matplotlib.org/
[Seaborn-badge]: https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge
[Seaborn-url]: https://seaborn.pydata.org/
[license-shield]: https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge
[license-url]: https://opensource.org/licenses/MIT
