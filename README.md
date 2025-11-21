
````markdown
# Nightmare Detection – Local Setup Guide

This guide explains how to run the Backend (FastAPI) and Frontend (Next.js) locally.

---

## 1. Clone the Repository

```bash
git clone https://github.com/Pavana-karanth/Nightmare_Detection.git
cd Nightmare_Detection
````

---

# Backend Setup (FastAPI)

## 2. Create a Virtual Environment

Navigate to the Backend directory:

```bash
cd Backend
python -m venv venv
```

Activate the virtual environment:

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Create a `.env` File

Create a `.env` file inside the `Backend/` directory with the following content:

```
MODEL_PATH=./models/robust_deep_svdd.pth
PORT=8000
```

---

## 5. Run the Backend Server

Make sure you are inside the `Backend/` directory with the virtual environment activated:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend will run at:

```
http://127.0.0.1:8000/
```

---

# Frontend Setup (Next.js)

Open a new terminal window. Keep the backend running.

## 6. Install Node Modules

Navigate to the Frontend directory:

```bash
cd Frontend
npm install
```

---

## 7. Create a `.env.local` File

Inside the `Frontend/` directory, create a file named `.env.local` with the following content:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 8. Start the Frontend Development Server

From inside the `Frontend/` directory:

```bash
npm run dev
```

The frontend will run at:

```
http://localhost:3000/
```

---

# Additional Notes

### Node Version

If you face Node version issues, install and use a compatible version using nvm:

```bash
nvm install 18
nvm use 18
```

### Reinstalling Node Modules

If node modules break or are missing:

```bash
rm -rf node_modules package-lock.json
npm install
```

### Stopping Servers

Press:

```
CTRL + C
```

to stop the backend or frontend server.

---

# Project Structure Overview

```
Nightmare_Detection/
│
├── Backend/
│   ├── venv/
│   ├── main.py
│   ├── models/
│   ├── requirements.txt
│   └── .env
│
└── Frontend/
    ├── node_modules/
    ├── pages/
    ├── components/
    ├── public/
    └── .env.local
```

```
