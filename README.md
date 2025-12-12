# DV Project Backend

FastAPI backend for the Data Visualization project analyzing lifestyle and fitness data.

## Features

- RESTful API for fitness and nutrition data analysis
- 20+ endpoints covering workouts, nutrition, exercises, and demographics
- CORS enabled for frontend integration
- Pandas-based data processing

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place `Final_data.csv` in the parent directory or set `DATA_PATH` environment variable

3. Run the server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Deployment on Render

1. Push code to GitHub
2. Create a new Web Service on Render
3. Connect your GitHub repository
4. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Set environment variables:
   - `FRONTEND_URL`: Your Vercel frontend URL (e.g., https://dv-frontend.vercel.app)
   - `DATA_PATH`: `./Final_data.csv`
6. Upload `Final_data.csv` to the Render service root directory

## Environment Variables

See `.env.example` for required environment variables.

## API Documentation

Once running, visit `/docs` for interactive API documentation.
