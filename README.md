# Nightmare Detection — Backend

This repository contains the backend service for the Nightmare Detection project. It includes the model loading code, API endpoints, and environment setup for local development.

---

## 📁 Project Structure

```
backend/
│── venv/                # Python virtual environment
│── main.py              # Main application file (FastAPI)
│── model/               # Model weights and related files
│── requirements.txt     # Python dependencies
│── README.md            # Project documentation
```

---

## 🔧 1. Clone This Repository

```bash
git clone https://github.com/Pavana-karanth/Nightmare_Detection.git
cd Nightmare_Detection
```

---

## 🐍 2. Create and Activate Virtual Environment

### **Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

### **Mac/Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 📦 3. Install Dependencies

Make sure your virtual environment is activated.

```bash
pip install -r requirements.txt
```

---

## 🚀 4. Run the Backend Server

```bash
uvicorn main:app --reload
```

The server will start at:

```
http://127.0.0.1:8000
```

---

## 🌐 5. Test the API

Open in browser:

```
http://127.0.0.1:8000/docs
```

---

## 🔄 6. Git: Connecting Local Project to GitHub

If you created the folder first and then the GitHub repo:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/Pavana-karanth/Nightmare_Detection.git
git pull origin main --allow-unrelated-histories
git push -u origin main
```

---

## 📝 Notes

* Keep the virtual environment **inside the backend folder** if you prefer local isolation.
* Do **not** push the `venv/` folder; your `.gitignore` should include it.
* Update `requirements.txt` whenever you install new dependencies:

```bash
pip freeze > requirements.txt
```

---
