# Supabase Deployment Guide

This directory contains instructions for deploying the Sepsis Prediction API and frontend using Supabase.

## Option 1: Deploy API as Supabase Edge Function

Supabase Edge Functions allow you to deploy serverless API endpoints. However, Edge Functions have limitations:
- **Max execution time**: ~10 seconds (Pro plan may offer longer)
- **Memory**: Limited
- **No sklearn/LightGBM/XGBoost**: Edge Functions run on Deno, not Python, so ML libraries are not available

### Recommendation
Do NOT deploy the ML model as a Supabase Edge Function. The model requires Python ML libraries (scikit-learn, LightGBM, XGBoost) that cannot run in the Deno runtime.

## Option 2: Host Frontend on Supabase Storage + CDN

You can host the Streamlit frontend on Supabase Storage with a CDN.

### Steps

1. **Build Streamlit for static hosting** (experimental):
   ```bash
   pip install streamlit-static
   streamlit static app.py --output-dir dist/
   ```

2. **Upload to Supabase Storage**:
   ```bash
   # Install Supabase CLI
   npm install -g supabase

   # Login
   supabase login

   # Link project
   supabase link --project-ref YOUR_PROJECT_REF

   # Upload to storage bucket
   supabase storage upload dist/ frontend/ --bucket public
   ```

3. **Configure CDN**:
   - Go to Supabase Dashboard → Storage → frontend bucket → Settings
   - Enable CDN
   - Use the CDN URL in your application

## Option 3: Deploy API separately + Frontend on Supabase

1. Deploy the FastAPI backend to:
   - **Render** (free tier available)
   - **Railway** (free tier available)
   - **Fly.io** (free tier available)
   - **DigitalOcean App Platform**
   - **AWS ECS / Fargate**

2. Host the Streamlit frontend on Supabase Storage (Option 2)

3. Update the frontend's `API_URL` to point to your deployed backend

## Option 4: Use Supabase for data storage only

Use Supabase PostgreSQL to store:
- Prediction history
- Patient records
- Model metadata
- API keys for authentication

### Example: Store predictions in Supabase

```python
from supabase import create_client

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

# After prediction
supabase.table("predictions").insert({
    "patient_id": patient_id,
    "prediction": prediction,
    "probability": probability,
    "timestamp": datetime.now().isoformat()
}).execute()
```

## Supabase Schema

```sql
-- predictions table
CREATE TABLE predictions (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  patient_id TEXT,
  prediction TEXT,
  probability FLOAT,
  features JSONB,
  explanation TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- api_keys table for auth
CREATE TABLE api_keys (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  key_hash TEXT NOT NULL,
  name TEXT,
  is_active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Row Level Security
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role can read predictions" ON predictions
  FOR SELECT USING (auth.role() = 'service_role');

CREATE POLICY "Service role can insert predictions" ON predictions
  FOR INSERT WITH CHECK (auth.role() = 'service_role');
```

## Summary

| Component | Recommended Host |
|-----------|-----------------|
| FastAPI Backend | Render / Railway / Fly.io / Docker |
| Streamlit Frontend | Supabase Storage (static) or Render |
| ML Model | Packaged with API in Docker container |
| Database | Supabase PostgreSQL (for logging predictions) |

## Environment Variables for Supabase

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
API_KEY=sk-sepsis-2024-dev-key
GOOGLE_AI_API_KEY=your-gemini-key
```
