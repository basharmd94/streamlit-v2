import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from auth_utils import hash_password
import psycopg2
from db.db_utils import config

def create_user(username, password, role):
    conn = None
    try:
        # Connect to database
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        
        # Hash password
        hashed_password = hash_password(password)
        if isinstance(hashed_password, memoryview):
            hashed_password = bytes(hashed_password)
        
        # Insert or update user
        cur.execute("""
        INSERT INTO users (username, password, role)
        VALUES (%s, %s, %s)
        ON CONFLICT (username) 
        DO UPDATE SET password = EXCLUDED.password, role = EXCLUDED.role
        """, (username, psycopg2.Binary(hashed_password), role))
        
        conn.commit()
        print(f"Successfully created/updated user: {username} with role: {role}")
        
    except Exception as e:
        print(f"Error creating user: {e}")
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python create_user.py <username> <password> <role>")
        print("Available roles: admin, sales, finance, purchase, SOP")
        sys.exit(1)
        
    username = sys.argv[1]
    password = sys.argv[2]
    role = sys.argv[3]
    
    if role not in ['admin', 'sales', 'finance', 'purchase', 'SOP']:
        print("Invalid role. Available roles: admin, sales, finance, purchase, SOP")
        sys.exit(1)
        
    create_user(username, password, role)
