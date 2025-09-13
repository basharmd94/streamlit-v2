-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password BYTEA NOT NULL,
    role VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create page_permissions table
CREATE TABLE IF NOT EXISTS page_permissions (
    id SERIAL PRIMARY KEY,
    role VARCHAR(20) NOT NULL,
    page_name VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(role, page_name)
);

-- Insert some default roles and permissions
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
    ('sales', 'Home'),
    ('sales', 'Overall Sales Analysis'),
    ('sales', 'YOY Analysis'),
    ('sales', 'Basket Analysis'),
    ('finance', 'Home'),
    ('finance', 'Overall Margin Analysis'),
    ('finance', 'Financial Statements'),
    ('finance', 'Collection Analysis')
ON CONFLICT DO NOTHING;
