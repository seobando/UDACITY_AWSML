# Install Python packages
COPY requirements.txt* /tmp/requirements.txt
RUN if [ -f /tmp/requirements.txt ]; then \
        pip install --upgrade pip && \
        pip install --no-cache-dir -r /tmp/requirements.txt; \
    else \
        pip install --upgrade pip; \
    fi