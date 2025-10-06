import streamlit as st
import bcrypt
import psycopg2
from db.db_utils import config
import extra_streamlit_components as stx
import json

def init_auth():
    """Initialize authentication state"""
    # Initialize session state variables if they don't exist
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    # Important: Do NOT auto-authenticate from cookies to prevent cross-user leakage

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed):
    try:
        if isinstance(hashed, memoryview):
            hashed = bytes(hashed)
        return bcrypt.checkpw(password.encode('utf-8'), hashed)
    except Exception as e:
        print(f"Password check error: {e}")
        return False

def login(username, password):
    conn = None
    try:
        # Connect to database
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        
        # Get user data
        cur.execute("""
        SELECT username, password, role 
        FROM users 
        WHERE username = %s
        """, (username,))
        
        result = cur.fetchone()
        
        if result:
            stored_password = result[1]
            if check_password(password, stored_password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.user_role = result[2]
                
                return True
        return False
        
    except Exception as e:
        print(f"Login error: {e}")
        return False
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def logout():
    try:
        # Clear session state
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.user_role = None
    except Exception as e:
        print(f"Logout error: {e}")

def check_page_access(page_name):
    if not st.session_state.authenticated:
        return False
    
    conn = None
    try:
        # Connect to database
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        
        # Check permission
        cur.execute("""
        SELECT COUNT(*) 
        FROM page_permissions 
        WHERE role = %s AND page_name = %s
        """, (st.session_state.user_role, page_name))
        
        result = cur.fetchone()
        return result[0] > 0 if result else False
        
    except Exception as e:
        print(f"Permission check error: {e}")
        return False
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def render_login_page():
    # Custom CSS for the login page
    st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            background-color: white;
        }
        .login-title {
            text-align: center;
            color: #2c3e50;
            font-size: 2rem;
            margin-bottom: 2rem;
            font-weight: bold;
        }
        .stButton>button {
            width: 100%;
            background-color: #2c3e50;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            font-weight: bold;
            margin-top: 1rem;
        }
        .stButton>button:hover {
            background-color: #34495e;
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
            border: 1px solid #bdc3c7;
            padding: 0.5rem;
        }
        .login-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .login-footer {
            text-align: center;
            margin-top: 1rem;
            color: #7f8c8d;
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Center the content vertically
    st.markdown("<br>" * 3, unsafe_allow_html=True)

    # Create three columns for horizontal centering
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        # Login header with logo
        st.markdown("""
            <div class="login-header">
                <h1 class="login-title">Business Analysis</h1>
                <p>Please sign in to continue</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Initialize form state if needed
        if 'login_form' not in st.session_state:
            st.session_state.login_form = {
                'username': '',
                'password': ''
            }
        
        # Login form with unique keys and state management
        username = st.text_input(
            "Username",
            key="login_username",
            value=st.session_state.login_form['username'],
            placeholder="Enter your username"
        )
        password = st.text_input(
            "Password",
            key="login_password",
            value=st.session_state.login_form['password'],
            type="password",
            placeholder="Enter your password"
        )
        
        # Update form state
        st.session_state.login_form['username'] = username
        st.session_state.login_form['password'] = password
        
        if st.button("Sign In", key="login_button"):
            if login(username, password):
                st.success("Login successful!")
                # Clear form state after successful login
                st.session_state.login_form = {'username': '', 'password': ''}
                # Use new API if available; fallback for older Streamlit
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()
            else:
                st.error("Invalid username or password")
        
        # Footer
        st.markdown("""
            <div class="login-footer">
                <p> 2025 Business Analysis. All rights reserved.</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
