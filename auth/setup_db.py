import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
import psycopg2
import bcrypt
from config.settings import get_db_params


def hash_password(password):
    if password is None:
        raise ValueError("Password is missing (None). Ensure *_PASSWORD is set (for example via the project .env file).")
    if isinstance(password, bytes):
        password_bytes = password
    else:
        password_bytes = str(password).encode("utf-8")
    return bcrypt.hashpw(password_bytes, bcrypt.gensalt())

def _load_dotenv_file(dotenv_path: Path) -> bool:
    if not dotenv_path.exists():
        return False

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            continue

        if key not in os.environ:
            os.environ[key] = value

    return True

# Roles to scan. For each prefix, setup_db looks for PREFIX_USERNAME_1,
# PREFIX_USERNAME_2, ... until the env var is absent. Add as many numbered
# entries as needed in .env — no code change required.
_ROLES = [
    ("ADMIN",      "admin"),
    ("SALES",      "sales"),
    ("FINANCE",    "finance"),
    ("PURCHASE",   "purchase"),
    ("CRM",        "crm"),
    ("SOP",        "SOP"),
    ("HR",         "HR"),
    ("PRODUCTION", "production"),
]


def _build_default_users():
    default_users = []

    for prefix, role in _ROLES:
        i = 1
        while True:
            user_env = f"{prefix}_USERNAME_{i}"
            pass_env = f"{prefix}_PASSWORD_{i}"
            username = os.getenv(user_env, "").strip()
            password = os.getenv(pass_env, "").strip()

            # Stop scanning this role when both vars are absent
            if not username and not password:
                break

            # One present without the other is a misconfiguration
            if not username or not password:
                missing = user_env if not username else pass_env
                raise ValueError(
                    f"Incomplete entry for {role} user #{i}: {missing} is missing. "
                    "Both USERNAME and PASSWORD must be set together."
                )

            default_users.append((username, password, role))
            i += 1

    if not default_users:
        raise ValueError(
            "No users found. Add at least one entry (e.g. ADMIN_USERNAME_1 / ADMIN_PASSWORD_1) "
            "to your .env file."
        )

    return default_users

def setup_auth_tables():
    conn = None
    cur = None
    try:
        project_root = Path(__file__).resolve().parent.parent
        dotenv_path = project_root / ".env"
        if _load_dotenv_file(dotenv_path):
            print(f"Loaded environment variables from: {dotenv_path}")
        else:
            print(f"Note: .env file not found at: {dotenv_path} (using process environment variables only)")

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
            -- Admin (full access)
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
            ('admin', 'Inventory Analysis'),
            ('admin', 'Customer Data View'),
            ('admin', 'Daily Sales Analysis'),
            ('admin', 'Target Management'),
            ('admin', 'AR Analysis'),
            ('admin', 'Customer Support'),
            ('admin', 'Marketing Analysis'),
            -- Sales
            ('sales', 'Home'),
            ('sales', 'Overall Sales Analysis'),
            ('sales', 'Daily Sales Analysis'),
            ('sales', 'Collection Analysis'),
            ('sales', 'Basket Analysis'),
            ('sales', 'Customer Data View'),
            ('sales', 'Target Management'),
            ('sales', 'Customer Support'),
            ('sales', 'Marketing Analysis'),
            -- Finance
            ('finance', 'Home'),
            ('finance', 'Overall Margin Analysis'),
            ('finance', 'Financial Statements'),
            ('finance', 'Collection Analysis'),
            ('finance', 'Accounting Analysis'),
            ('finance', 'Inventory Analysis'),
            ('finance', 'Manufacturing Analysis'),
            ('finance', 'AR Analysis'),
            -- Purchase
            ('purchase', 'Home'),
            ('purchase', 'Purchase Analysis'),
            ('purchase', 'Basket Analysis'),
            ('purchase', 'Distribution & Histograms'),
            ('purchase', 'Inventory Analysis'),
            ('purchase', 'Manufacturing Analysis'),
            -- CRM
            ('crm', 'Home'),
            ('crm', 'Overall Sales Analysis'),
            ('crm', 'Daily Sales Analysis'),
            ('crm', 'Collection Analysis'),
            ('crm', 'Customer Data View'),
            ('crm', 'Target Management'),
            ('crm', 'Customer Support'),
            ('crm', 'Marketing Analysis'),
            -- SOP
            ('SOP', 'Customer Data View'),
            -- HR
            ('HR', 'Home'),
            ('HR', 'Target Management'),
            -- Production
            ('production', 'Home'),
            ('production', 'Overall Sales Analysis'),
            ('production', 'Daily Sales Analysis'),
            ('production', 'Manufacturing Analysis'),
            ('production', 'Inventory Analysis')
        """)

        default_users = _build_default_users()

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
        print("\nDefault Users Created:")
        for username, _, role in default_users:
            print(f" - {role}: {username}")
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
