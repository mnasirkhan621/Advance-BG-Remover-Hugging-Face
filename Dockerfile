# 1. Base Image
# We use a lightweight version of Python 3.10 running on Linux (Debian).
# "slim" means it excludes unnecessary tools, making the image smaller and faster to download.
FROM python:3.10-slim

# 2. Set Working Directory
# This creates a folder inside the container and sets it as the default location for subsequent commands.
WORKDIR /app

# 3. Install System Dependencies
# OpenCV (used in your code) requires 'libgl1'.
# We update the package list (apt-get update) and install it.
# '-y' automatically answers "yes" to prompts.
# 'rm -rf...' cleans up the temporary list to keep the image small.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy Dependencies File
# We copy ONLY requirements.txt first.
# Why? Docker caches steps. If you change your code (app.py) but not requirements.txt,
# Docker will skip the slow "pip install" step next time you build.
COPY requirements.txt .

# 5. Install Python Dependencies
# --no-cache-dir: Don't save downloaded files, saves space.
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code
# Now we copy the rest of your app code into the container.
COPY . .

# 7. Expose Port
# Tells Docker that this container listens on port 8080.
# (This is mostly for documentation, you still need to map ports when running).
EXPOSE 8070

# 8. Command to Run
# This is what executes when you type 'docker run'.
# host 0.0.0.0 is CRITICAL: it allows connections from outside the container (like your browser).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8070"]
