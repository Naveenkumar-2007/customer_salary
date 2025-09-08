# 💳 Customer Salary Prediction using ANN + FastAPI  

This project predicts a **customer's estimated salary** based on banking details using an **Artificial Neural Network (ANN)**.  
The model is trained on the **Churn_Modelling dataset**, and served via a **FastAPI** endpoint for real-time predictions.  

---

## 🚀 Features  

- 🔄 **Data Preprocessing**  
  - Removed irrelevant columns (`RowNumber`, `CustomerId`, `Surname`)  
  - Encoded **Gender** using `LabelEncoder`  
  - Encoded **Geography** using `OneHotEncoder`  
  - Standardized features with `StandardScaler`  

- 🧠 **Machine Learning Model (ANN)**  
  - Input layer with multiple numerical + categorical features  
  - Hidden layers with ReLU activation functions  
  - Output layer for regression (salary prediction)  
  - Optimizer: **Adam** | Loss: **Mean Absolute Error (MAE)**  
  - Metrics: **MAE**  

- ⚡ **Deployment with FastAPI**  
  - REST API for real-time predictions  
  - Loads trained ANN model + preprocessing objects  
  - Accepts raw input (categorical + numerical)  
  - Returns **Predicted Salary** in JSON  

---

## 📊 Input and Output Features  

### 🔹 Input Features (what the user provides)  

| Feature           | Type    | Description |
|-------------------|---------|-------------|
| **CreditScore**   | float   | Customer’s credit score |
| **Geography**     | string  | Country (`France`, `Spain`, `Germany`) |
| **Gender**        | string  | `Male` or `Female` |
| **Age**           | int     | Customer’s age |
| **Tenure**        | int     | Years with the bank |
| **Balance**       | float   | Current account balance |
| **NumOfProducts** | int     | Number of products (1, 2, 3, etc.) |
| **HasCrCard**     | int     | Whether customer has a credit card (`0` = No, `1` = Yes) |
| **IsActiveMember**| int     | Whether customer is active (`0` = No, `1` = Yes) |
| **Exited**        | int     | Whether customer left the bank (`0` = No, `1` = Yes) |

---

### 🔹 Output Feature  

| Output Feature         | Type   | Description |
|------------------------|--------|-------------|
| **Predicted Salary**   | float  | The model predicts the **Estimated Salary** of the customer (continuous value). |

---

## 🔄 How the Prediction Works  

1️⃣ **User sends data** (numerical + categorical features) in JSON format to the API.  

2️⃣ **Preprocessing applied**:  
   - Gender → Converted to `0` (Female) / `1` (Male)  
   - Geography → One-hot encoded (`France`, `Spain`, `Germany`)  
   - All features standardized using **StandardScaler**  

3️⃣ **ANN Model Prediction**:  
   - Processed input is passed through the **neural network**  
   - The ANN learns patterns from features and estimates salary  

4️⃣ **Output Returned**:  
   - Model outputs a **single float value** (predicted salary)  
   - API responds with JSON  


