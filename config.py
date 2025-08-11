"""
Configuration file for LangChain integration
Contains API keys, model settings, and vector store configuration
"""

import os
from typing import Optional

# ============= LLM Configuration =============
# Model choice: 'openai' or 'google' or 'anthropic'
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai')

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.3'))

# Google Vertex AI Configuration
GOOGLE_PROJECT_ID = os.getenv('GOOGLE_PROJECT_ID', '')
GOOGLE_LOCATION = os.getenv('GOOGLE_LOCATION', 'us-central1')
GOOGLE_MODEL = os.getenv('GOOGLE_MODEL', 'text-bison')

# Anthropic Configuration
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229')

# ============= Vector Store Configuration =============
ENABLE_VECTOR_STORE = os.getenv('ENABLE_VECTOR_STORE', 'true').lower() == 'true'
VECTOR_STORE_PATH = os.getenv('VECTOR_STORE_PATH', 'data/vector_store')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-ada-002')

# ============= Output Configuration =============
REPORTS_DIR = os.getenv('REPORTS_DIR', 'reports')
MARKDOWN_DIR = os.path.join(REPORTS_DIR, 'markdown')
HTML_DIR = os.path.join(REPORTS_DIR, 'html')

# ============= LangChain Configuration =============
# Maximum token length for summaries
MAX_SUMMARY_TOKENS = int(os.getenv('MAX_SUMMARY_TOKENS', '1000'))

# Retry configuration
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
RETRY_DELAY = float(os.getenv('RETRY_DELAY', '1.0'))

# ============= Health Data Configuration =============
# Thresholds for anomaly detection
HEALTH_THRESHOLDS = {
    'hr': {'low': 60, 'high': 100, 'critical_low': 40, 'critical_high': 150},
    'bp_systolic': {'low': 90, 'high': 140, 'critical_low': 70, 'critical_high': 180},
    'bp_diastolic': {'low': 60, 'high': 90, 'critical_low': 40, 'critical_high': 120},
    'spo2': {'low': 95, 'high': 100, 'critical_low': 88, 'critical_high': 100},
    'temp': {'low': 97.0, 'high': 99.5, 'critical_low': 95.0, 'critical_high': 103.0},
    'steps': {'low': 5000, 'high': 15000, 'critical_low': 1000, 'critical_high': 30000},
    'sleep_hours': {'low': 6, 'high': 9, 'critical_low': 4, 'critical_high': 12}
}

# Data quality thresholds
DATA_QUALITY_THRESHOLDS = {
    'good': {'completeness': 0.9, 'anomaly_ratio': 0.05},
    'acceptable': {'completeness': 0.7, 'anomaly_ratio': 0.15},
    'poor': {'completeness': 0.5, 'anomaly_ratio': 0.3}
}

# ============= Validation Functions =============
def validate_config() -> dict:
    """Validate configuration and return status."""
    issues = []
    
    # Check LLM provider and API keys
    if LLM_PROVIDER == 'openai' and not OPENAI_API_KEY:
        issues.append("OpenAI API key not provided")
    elif LLM_PROVIDER == 'google' and (not GOOGLE_PROJECT_ID or not GOOGLE_LOCATION):
        issues.append("Google Cloud project ID and location required")
    elif LLM_PROVIDER == 'anthropic' and not ANTHROPIC_API_KEY:
        issues.append("Anthropic API key not provided")
    
    # Check if unsupported provider
    if LLM_PROVIDER not in ['openai', 'google', 'anthropic']:
        issues.append(f"Unsupported LLM provider: {LLM_PROVIDER}")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'provider': LLM_PROVIDER,
        'vector_store_enabled': ENABLE_VECTOR_STORE
    }

def get_model_config() -> dict:
    """Get model configuration based on provider."""
    config = validate_config()
    
    if not config['valid']:
        raise ValueError(f"Configuration issues: {config['issues']}")
    
    if LLM_PROVIDER == 'openai':
        return {
            'provider': 'openai',
            'model': OPENAI_MODEL,
            'api_key': OPENAI_API_KEY,
            'temperature': OPENAI_TEMPERATURE
        }
    elif LLM_PROVIDER == 'google':
        return {
            'provider': 'google',
            'model': GOOGLE_MODEL,
            'project_id': GOOGLE_PROJECT_ID,
            'location': GOOGLE_LOCATION
        }
    elif LLM_PROVIDER == 'anthropic':
        return {
            'provider': 'anthropic',
            'model': ANTHROPIC_MODEL,
            'api_key': ANTHROPIC_API_KEY
        }

# ============= Example Environment Setup =============
def print_env_example():
    """Print example environment variables."""
    print("""
Example .env file:

# Choose your LLM provider
LLM_PROVIDER=openai

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_TEMPERATURE=0.3

# Google Vertex AI Configuration (alternative)
# LLM_PROVIDER=google
# GOOGLE_PROJECT_ID=your_project_id
# GOOGLE_LOCATION=us-central1
# GOOGLE_MODEL=text-bison

# Anthropic Configuration (alternative)
# LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=your_anthropic_api_key

# Vector Store Configuration
ENABLE_VECTOR_STORE=true
VECTOR_STORE_PATH=data/vector_store

# Output Configuration
REPORTS_DIR=reports
    """)

if __name__ == "__main__":
    # Test configuration
    config = validate_config()
    print("Configuration Status:")
    print(f"Valid: {config['valid']}")
    print(f"Provider: {config['provider']}")
    print(f"Vector Store: {config['vector_store_enabled']}")
    
    if not config['valid']:
        print(f"Issues: {config['issues']}")
        print("\n" + "="*50)
        print_env_example()
