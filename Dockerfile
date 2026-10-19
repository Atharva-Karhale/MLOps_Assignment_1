# Start with a lightweight computer running Python 3.11
FROM python:3.11-slim

# Set up a working folder inside the container
WORKDIR /app

# Copy the requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the rest of our files into the container
COPY . .

# Open port 5000 for the drive-thru window
EXPOSE 5000

# Tell the container to start the API when it turns on
CMD ["python", "app.py"]