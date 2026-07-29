# NBA Team Simulator App
I created a webapp to simulate the win probability of any team. I used data gathered from kaggle.com for all 30 NBA teams data, ranging from 1996 to 2025.




```bash
# install requirements in project root
pip install -r requirements.txt

# 1. data pipeline (Run in R, in this sequential order)
R/01_clean_data.R
R/02_elo_ratings.R
R/03_zscore_features.R

# 2. train the model
python/train_model.py

# 3. start the app (serves both the API and the frontend at the same address)
uvicorn app.backend:app --reload --app-dir .
```

Then open **http://localhost:8000** — the frontend is served by the same
FastAPI process, so you don't need a separate dev server or to configure a
CORS origin. If you'd rather run the frontend separately (e.g. opening
`frontend/index.html` directly as a file), fill in the "API base" field at
the top of the page with `http://localhost:8000`.
