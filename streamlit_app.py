#!/usr/bin/env python3
"""
Streamlit Dashboard App Runner
Launch the Portfolio Risk Management Dashboard
"""

import streamlit as st
import sys
import os

# Add the main module to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the enhanced system
from main import (
    create_enhanced_portfolio, 
    EnhancedRiskManagementSystem,
    RiskManagementDashboard
)
    st.set_page_config(
        page_title="Portfolio Risk Management System",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
def main():
    """Main Streamlit application entry point"""
    # Page configuration

    # Initialize session state
    if 'risk_system' not in st.session_state:
        portfolio = create_enhanced_portfolio()
        st.session_state.risk_system = EnhancedRiskManagementSystem(portfolio)
        st.session_state.dashboard = RiskManagementDashboard(st.session_state.risk_system)
    
    # Run the dashboard
    st.session_state.dashboard.run_dashboard()

if __name__ == "__main__":
    main()
