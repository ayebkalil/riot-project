# 🚀 Frontend Launch Guide - For Professor Demo

## ⚠️ Current Status

**Node.js is NOT installed** on your system. You need it to run the React frontend.

---

## 🎯 Quick Setup (5 minutes)

### Step 1: Install Node.js

1. Download Node.js: **https://nodejs.org/** (click "LTS" version)
2. Run the installer (use default settings)
3. Restart PowerShell after installation

### Step 2: Install Dependencies

```powershell
cd "C:\Users\ayebk\OneDrive\Desktop\Riot Games Project\frontend\hextech-insights (1)"
npm install
```

This will take ~2-3 minutes to download all packages.

### Step 3: Launch Frontend

```powershell
npm run dev
```

The app will open at: **http://localhost:5173**

---

## 🎓 What to Show Professor

### **Frontend Features (Visual Demo):**

1. **Dashboard Overview** - Analytics and player stats
2. **Model Dashboard** - Shows all 4 ML models
   - Match Outcome Prediction (92% accuracy mock)
   - Rank Classification (88%)
   - Player Progression (85%)
   - Smurf Detection (95%)
3. **Predictions Page** - Team composition analyzer
4. **Profile Page** - Player match history

### **Important Notes for Demo:**

⚠️ **The frontend currently shows MOCK DATA only**
- All percentages are placeholder values
- No real API connection yet
- Models are not actually running

✅ **What IS Real:**
- The UI design and user experience
- The layout and navigation
- The visual representation of your project vision

---

## 💡 Recommended Demo Strategy

### Option A: Show Both Separately

1. **MLflow Dashboard** ← Show REAL model performance
   - Run: `mlflow ui --port 5000`
   - Open: http://localhost:5000
   - Show: Rank classifier with 65.21% accuracy, confusion matrix, feature importance

2. **Frontend Mockup** ← Show UI design
   - Run: `npm run dev` (in frontend folder)
   - Open: http://localhost:5173
   - Explain: "This is our UI design - API integration is next milestone"

### Option B: Show MLflow Only (If No Time for Node.js Setup)

Just demonstrate MLflow with your trained model:
```powershell
mlflow ui --port 5000
```

Then show the frontend **screenshots** from the `screen.png` files in:
- `frontend/dashboard_overview/screen.png`
- `frontend/ml_model_dashboard/screen.png`
- `frontend/match_outcome_prediction/screen.png`

---

## 🐛 Troubleshooting

### "npm: command not found"
→ Node.js not installed or not in PATH. Restart PowerShell after installing Node.js.

### Port 5173 already in use
```powershell
npm run dev -- --port 3000
```

### "Cannot find module"
```powershell
rm -r node_modules
npm install
```

---

## 📁 Project Architecture (To Explain)

```
Riot Games Project/
├── models/                          # ✅ 4 trained ML models
│   ├── 1_rank_tier_classifier/     # ✅ 65.21% accuracy
│   ├── 2_progression_regressor/
│   ├── 3_smurf_anomaly_detector/
│   └── 4_match_outcome_predictor/
│
├── data/                            # ✅ ~4,340 player dataset
│   └── processed/                   # CSV features ready for ML
│
├── frontend/                        # ✅ React + TypeScript UI
│   └── hextech-insights/           # Modern League of Legends themed
│
└── mlflow/                          # ✅ Experiment tracking
    └── mlruns/                      # All training runs logged
```

---

## ✅ What You've Accomplished (Talking Points)

### Data Pipeline ✅
- Collected 4,340+ player profiles from Riot API
- Built OP.GG web scraper for validation data
- Engineered 40+ gameplay features (KDA, CS/min, gold efficiency, etc.)

### Machine Learning ✅
- Trained 4 distinct models
- **Rank Tier Classifier**: 65.21% accuracy (4-class problem)
- Implemented MLflow for experiment tracking
- Generated professional visualizations (confusion matrix, feature importance)

### Frontend Development ✅
- Built modern React application
- League of Legends themed UI with Hextech aesthetics
- Responsive design with 4 main pages
- Ready for API integration

### Next Steps 📋
- Build FastAPI backend to serve models
- Connect frontend to real predictions
- Deploy to cloud (optional)

---

## 🎯 Key Message for Professor

**"We've built a complete ML pipeline with professional experiment tracking. The UI design is ready - we're now in the API integration phase to connect the frontend to our trained models."**

This shows:
- ✅ Strong ML fundamentals (training, evaluation, tracking)
- ✅ Professional development practices (MLflow, version control)
- ✅ Full-stack thinking (backend + frontend architecture)
- ✅ Clear roadmap for completion

---

**Good luck with your demo! 🚀**
