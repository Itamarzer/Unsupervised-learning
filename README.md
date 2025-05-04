# 🧠 Unsupervised Learning Toolkit

In this project, we had a singular data set that contained audio of emotions that were transformed into a tabular setting. In the data, we performed dimensional reduction and clustering analysis using different methods and explained the best. We associated the different clusters to indicate different emotions to see how different emotions can be very similar, We tested whether several emotions can be found in one message and also whether the beginning and end of each message are noise that may interfere with the correct identification of the emotions. After that, we found anomalies in the data using anomaly detection tools and tested whether these anomalies could give us some insight into the emotion expression to improve our clustering. Lastly, we propose a visualization that best characterizes the clusters we optimized. We concluded that Fuzzy C-means with LLE dimensionality reduction is the overall best algorithm, and that there is a difference within one message, and that the beginning and end of the messages are mostly an important part of it for the emotion detection. You can find our project in here: https://www.overleaf.com/read/gwbrmngybvrr#80d67b.


---

## 🗒️ Notes

- The code executes a full unsupervised learning pipeline — including statistical analysis, dimensionality reduction, clustering, and evaluation of clustering performance.
- It automatically determines the optimal number of clusters using various methods (e.g., elbow method, silhouette score, statistical testing).
- All generated plots (e.g., cluster visualizations, elbow curves, dimensionality reduction plots) are saved locally within organized subdirectories in the repository.
- Output file paths are printed to the console for quick access to results and figures.
- The project is modular: you can run specific steps (like clustering or anomaly detection) independently by executing individual scripts.
- Make sure the input data is placed in the correct format and location as expected by the `data_import.py` module.
- Intermediate results and CSVs (e.g., mean scores) are also saved for further inspection and reproducibility.

---

## ‼️ Important Notice

- If you encounter any issues, bugs, or unexpected behavior while using this project, feel free to reach out for support.
- Contact: **nadavlisha@gmail.com** — I'll do my best to respond promptly and assist with any problems you face.
- When reporting issues, please include relevant details such as the error message, operating system, and steps to reproduce the issue. This helps speed up troubleshooting.

---
## 🛠️ Features

- **Clustering Algorithms**: Implementations of K-Means, DBSCAN, Spectral Clustering, etc.
- **Dimensionality Reduction**: Algorithms like PCA and t-SNE to reduce dimensions.
- **Anomaly Detection**: Scripts to identify outliers in datasets.
- **Visualization Tools**: Generate plots and elbow curves for better interpretability.
- **Statistical Tests**: Analyze cluster validity using statistical methods.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Itamarzer/Unsupervised-learning.git
cd Unsupervised-learning
```

### 2. install dependencies
```bash
pip install -r req.txt
```

### 3. Run the main script
```bash
pyton main_project.py
```

---
