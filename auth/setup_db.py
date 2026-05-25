import sys
import os
import psycopg2
import bcrypt
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import get_db_params

load_dotenv()


def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def setup_auth_tables():
    conn = None
    cur = None
    try:
        db_params = get_db_params()
        # Connect to database
        print("Connecting to database...")
        print(f"Host: {db_params['host']}")
        print(f"Database: {db_params['dbname']}")
        print(f"User: {db_params['user']}")

        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        # Create users table
        print("Creating users table...")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username VARCHAR(50) PRIMARY KEY,
            password BYTEA NOT NULL,
            role VARCHAR(20) NOT NULL
        )
        """)
        
        # Create page_permissions table
        print("Creating page_permissions table...")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS page_permissions (
            role VARCHAR(20),
            page_name VARCHAR(100),
            PRIMARY KEY (role, page_name)
        )
        """)
        
        # Clear existing permissions to avoid duplicates
        print("Clearing existing permissions...")
        cur.execute("DELETE FROM page_permissions")
        
        # Insert default permissions
        print("Setting up default permissions...")
        cur.execute("""
            INSERT INTO page_permissions (role, page_name) VALUES
            ('admin', 'Home'),
            ('admin', 'Overall Sales Analysis'),
            ('admin', 'Overall Margin Analysis'),
            ('admin', 'Purchase Analysis'),
            ('admin', 'Collection Analysis'),
            ('admin', 'Distribution & Histograms'),
            ('admin', 'Descriptive Statistics'),
            ('admin', 'Basket Analysis'),
            ('admin', 'Financial Statements'),
            ('admin', 'Manufacturing Analysis'),
            ('admin', 'Accounting Analysis'),
            ('admin','Inventory Analysis'),
            ('admin','Customer Data View'),
            ('admin','Daily Sales Analysis'),
            ('admin','Target Management'),
            ('sales', 'Home'),
            ('sales', 'Overall Sales Analysis'),
            ('sales', 'Daily Sales Analysis'),
            ('sales','Collection Analysis'),
            ('sales', 'Basket Analysis'),
            ('sales','Customer Data View'),
            ('sales','Target Management'),
            ('SOP', 'Customer Data View'),
            ('crm', 'Collection Analysis'),
            ('crm', 'Overall Sales Analysis'),
            ('crm', 'Daily Sales Analysis'),
            ('crm', 'Customer Data View'),
            ('finance', 'Home'),
            ('finance', 'Overall Margin Analysis'),
            ('finance', 'Financial Statements'),
            ('finance', 'Collection Analysis'),
            ('finance', 'Accounting Analysis'),
            ('finance','Inventory Analysis'),
            ('purchase', 'Home'),
            ('purchase', 'Purchase Analysis'),
            ('purchase', 'Basket Analysis'),
            ('purchase', 'Distribution & Histograms'),
            ('purchase','Inventory Analysis'),
            ('admin', 'AR Analysis'),
            ('finance', 'AR Analysis'),
            ('hr', 'Target Management')
        """)
        
        # Create default users with their roles
        default_users = [
            (os.getenv('ADMIN_USERNAME'), os.getenv('ADMIN_PASSWORD'), 'admin'),
            (os.getenv('SALES_USERNAME'), os.getenv('SALES_PASSWORD'), 'sales'),
            (os.getenv('FINANCE_USERNAME'), os.getenv('FINANCE_PASSWORD'), 'finance'),
            (os.getenv('PURCHASE_USERNAME'), os.getenv('PURCHASE_PASSWORD'), 'purchase'),
            (os.getenv('CRM_USERNAME'), os.getenv('CRM_PASSWORD'), 'crm'),
            (os.getenv('SOP_USERNAME'), os.getenv('SOP_PASSWORD'), 'SOP'),
            (os.getenv('HR_USERNAME'), os.getenv('HR_PASSWORD'), 'hr'),
           
        ]
        
        # Clear existing users to avoid duplicates
        print("Clearing existing users...")
        cur.execute("DELETE FROM users")
        
        # Insert users with hashed passwords
        print("Creating default users...")
        for username, password, role in default_users:
            hashed_password = hash_password(password)
            cur.execute("""
            INSERT INTO users (username, password, role)
            VALUES (%s, %s, %s)
            """, (username, hashed_password, role))
        
        conn.commit()
        print("\nDatabase setup completed successfully!")
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error: {error}")
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    setup_auth_tables()