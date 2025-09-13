# Project Title
Business Data Analysis App

## Description
This application provides business analysis tools, allowing users to analyze sales, finance, and purchase data through a web interface.

## Prerequisites
- Python 3.8 or higher
- PostgreSQL database
- Required Python packages listed in `requirements.txt`

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <project_directory>
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the database connection in `config/database.ini`:
   ```ini
   [database]
   host=localhost
   database=stream2
   user=postgres
   password=postgres
   port=5432
   ```
4. Sync all Files from da database to stream2:
   ```bash
   python db_sync/main.py

5. For creating Auth User Run the database setup script to create tables and default users:
   ```bash
   python auth/setup_db.py
   ```


   ```
   This will delete the existing database called stream2 and create a new one

## Running the Application
To start the Streamlit app, run:
```bash
streamlit run app.py
```

## Database Setup
Make sure to run `setup_db.py` after cloning the repository to create the necessary tables and default users. This should be done every month to ensure the database is up-to-date.

## User Credentials
Default users created:
- Admin User: `admin_user` / `admin123`
- Sales User: `sales_user` / `sales123`
- Finance User: `finance_user` / `finance123`
- Purchase User: `purchase_user` / `purchase123`


## Running the App 
- restore the da database 
- run python main.py from db_sync/main.py
- run auth/setup_db.py

## For Finding Any Problem