#  House Price Prediction using XGBoost Stacking  

This project aims to predict house prices using an **XGBoost-based stacking model**, leveraging multiple regression algorithms for improved accuracy.  

---

##  Dataset  
The dataset used for this project is **California Housing Prices**, containing various features like:  
✔️ Median income  
✔️ Population  
✔️ House age  
✔️ Latitude & longitude  
✔️ Number of rooms, bedrooms, and households  

These features help predict **median house prices**.  

---

##  Methodology  

🔹 **Data Preprocessing** – Handling missing values, feature scaling, and encoding.  
🔹 **Feature Engineering** – Selecting the most relevant features for better performance.  
🔹 **Model Stacking** – Combining multiple regression models with **XGBoost** as the final estimator.  
🔹 **Evaluation Metrics** – Measuring model performance using MAE, RMSE, and R² Score.  

---

##  Results  

 **Mean Absolute Error (MAE):** 0.4088  
 **Root Mean Squared Error (RMSE):** 0.5895  
 **R² Score:** 0.7348  

---

##  Technologies Used  

-  **Python**  
-  **XGBoost, Scikit-Learn**  
-  **Pandas, NumPy, Matplotlib**  

---

##  Usage  

###  Clone the repository  
```bash
git clone https://github.com/nishu-2004/House-Price-Prediction-using-XGBoost-Stacking.git
```
###  Run the model  
```python
python train.py
```
