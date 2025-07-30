# ✅ 1. Use slim base image — good
FROM python:3.11-slim

# ✅ 2. Set working directory
WORKDIR /app

# ✅ 3. Install required system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ✅ 4. Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ✅ 5. Copy the app code
COPY . .

# ✅ 6. Environment variables
ENV PYTHONUNBUFFERED=1

# ✅ 7. Flask app port
EXPOSE 8000

# ✅ 8. Run via Gunicorn (make sure app is named correctly)
CMD ["gunicorn", "-b", "0.0.0.0:8000", "application:application"]
