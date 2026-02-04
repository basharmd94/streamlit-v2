import os
import sys
import psycopg2
import bcrypt

# Manually define database credentials
db_params = {
    'host': 'localhost',
    'database': 'stream2',
    'user': 'postgres',
    'password': 'postgres',  # Replace with your actual password
    'port': '5432'
}

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def setup_auth_tables():
    conn = None
    cur = None
    try:
        # Connect to database
        print("Connecting to database...")
        print(f"Host: {db_params['host']}")
        print(f"Database: {db_params['database']}")
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
            ('admin', 'YOY Analysis'),
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
            ('sales', 'Home'),
            ('sales', 'Overall Sales Analysis'),
            ('sales', 'YOY Analysis'),
            ('sales', 'Basket Analysis'),
            ('sales','Customer Data View'),
            ('SOP', 'Customer Data View'),
            ('crm', 'Collection Analysis'),
            ('crm', 'Overall Sales Analysis'),
            ('crm', 'Customer Data View'),
            ('finance', 'Home'),
            ('finance', 'Overall Margin Analysis'),
            ('finance', 'Financial Statements'),
            ('finance', 'Collection Analysis'),
            ('finance', 'Accounting Analysis'),
            ('finance','Inventory Analysis'),
            ('purchase', 'Home'),
            ('purchase', 'Purchase Analysis'),
            ('purchase', 'YOY Analysis'),
            ('purchase', 'Distribution & Histograms'),
            ('purchase','Inventory Analysis')
        ON CONFLICT DO NOTHING
        """)
        
        # Create default users with their roles
        default_users = [
            ('admin_user', 'admin123', 'admin'),
            ('sales_user', 'sales123', 'sales'),
            ('finance_user', 'finance123', 'finance'),
            ('purchase_user', 'purchase123', 'purchase'),
            ('crm_user', 'crm3210', 'crm'),
            ('SOP_user', 'sop123', 'SOP')
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
            ON CONFLICT (username) 
            DO UPDATE SET password = EXCLUDED.password, role = EXCLUDED.role
            """, (username, hashed_password, role))
        
        conn.commit()
        print("\nDatabase setup completed successfully!")
        print("\nDefault Users Created:")
        print("1. Admin User:")
        print("   - Username: admin_user")
        print("   - Password: admin123")
        print("   - Access: All pages")
        print("\n2. Sales User:")
        print("   - Username: sales_user")
        print("   - Password: sales123")
        print("   - Access: Home, Overall Sales Analysis, YOY Analysis, Basket Analysis")
        print("\n3. Finance User:")
        print("   - Username: finance_user")
        print("   - Password: finance123")
        print("   - Access: Home, Overall Margin Analysis, Financial Statements, Collection Analysis")
        print("\n4. Purchase User:")
        print("   - Username: purchase_user")
        print("   - Password: purchase123")
        print("   - Access: Home, Purchase Analysis, YOY Analysis, Distribution & Histograms")
        print("\n5. CRM User:")
        print("   - Username: crm_user")
        print("   - Password: crm3210")
        print("   - Access: Home, Collection Analysis, Overall Sales Analysis")
        print("\n6. SOP User:")
        print("   - Username: SOP_user")
        print("   - Password: sop123")
        print("   - Access: Home, Customer Data View")
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