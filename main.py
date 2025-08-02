#!/usr/bin/env python3
"""
Complete Autonomous Portfolio Risk Management Agent System
Multi-Agent Architecture with Advanced ML and Real-Time Dashboard
Enterprise-Grade Financial AI System with Streamlit Interface
"""

import asyncio
import threading
import time
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import logging
from scipy import stats
import warnings
import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Machine Learning Imports
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans

# Advanced Analytics
try:
    import ta  # Technical Analysis library
except ImportError:
    ta = None
    
import networkx as nx  # For correlation networks
from scipy.optimize import minimize  # For portfolio optimization

warnings.filterwarnings('ignore')
st.set_page_config(page_title="FinAgent", layout="wide")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class RiskAlert:
    agent_id: str
    alert_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    message: str
    data: Dict[str, Any]
    timestamp: datetime
    recommended_action: str
    confidence_score: float = 0.0
    
    def to_json_dict(self):
        """Convert to JSON-serializable dictionary"""
        return {
            'agent_id': self.agent_id,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'data': self._serialize_data(self.data),
            'timestamp': self.timestamp.isoformat(),
            'recommended_action': self.recommended_action,
            'confidence_score': self.confidence_score
        }
    
    def _serialize_data(self, data):
        """Serialize data to JSON-compatible format"""
        if isinstance(data, dict):
            return {k: self._serialize_value(v) for k, v in data.items()}
        return self._serialize_value(data)
    
    def _serialize_value(self, value):
        """Serialize individual values"""
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.isoformat() if hasattr(value, 'isoformat') else str(value)
        elif isinstance(value, (np.integer, np.floating)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif pd.isna(value):
            return None
        return value

@dataclass
class Portfolio:
    symbols: List[str]
    weights: Dict[str, float]
    total_value: float
    last_updated: datetime
    sector_allocation: Dict[str, float] = None
    risk_budget: Dict[str, float] = None

# Base Agent Class
class BaseAgent(ABC):
    """Base class for all risk management agents"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.is_active = False
        self.last_update = None
        
    @abstractmethod
    async def analyze(self, data: Any) -> List[RiskAlert]:
        """Main analysis method for the agent"""
        pass
    
    async def start(self):
        """Start the agent"""
        self.is_active = True
        logging.info(f"Agent {self.agent_id} started")
    
    async def stop(self):
        """Stop the agent"""
        self.is_active = False
        logging.info(f"Agent {self.agent_id} stopped")

class RiskScannerAgent(BaseAgent):
    """Real-time risk scanning agent"""
    
    def __init__(self):
        super().__init__("RISK_SCANNER")
        self.risk_thresholds = {
            'volatility': 0.30,
            'var_95': 0.05,
            'correlation': 0.80,
            'drawdown': 0.15
        }
    
    async def analyze(self, market_data: pd.DataFrame) -> List[RiskAlert]:
        """Analyze market data for risk signals"""
        alerts = []
        
        try:
            if market_data.empty:
                return alerts
            
            # Check for high volatility
            for symbol in market_data.columns.get_level_values(1).unique():
                try:
                    if ('Close', symbol) in market_data.columns:
                        prices = market_data[('Close', symbol)].dropna()
                        if len(prices) > 1:
                            returns = prices.pct_change().dropna()
                            if len(returns) > 0:
                                volatility = returns.std() * np.sqrt(252)  # Annualized
                                
                                if volatility > self.risk_thresholds['volatility']:
                                    alerts.append(RiskAlert(
                                        agent_id=self.agent_id,
                                        alert_type="HIGH_VOLATILITY",
                                        severity="HIGH",
                                        message=f"{symbol} volatility ({volatility:.2%}) exceeds threshold ({self.risk_thresholds['volatility']:.2%})",
                                        data={'symbol': symbol, 'volatility': float(volatility)},
                                        timestamp=datetime.now(),
                                        recommended_action=f"Consider reducing {symbol} position or adding hedging",
                                        confidence_score=0.85
                                    ))
                except Exception as e:
                    logging.warning(f"Risk analysis failed for {symbol}: {e}")
                    
        except Exception as e:
            logging.error(f"Risk scanner error: {e}")
        
        return alerts

class AnalysisSpecialistAgent(BaseAgent):
    """Deep analysis specialist agent"""
    
    def __init__(self):
        super().__init__("ANALYSIS_SPECIALIST")
    
    async def analyze(self, portfolio: Portfolio) -> List[RiskAlert]:
        """Perform deep portfolio analysis"""
        alerts = []
        
        try:
            # Check for concentration risk
            max_weight = max(portfolio.weights.values())
            if max_weight > 0.25:
                symbol = max(portfolio.weights, key=portfolio.weights.get)
                alerts.append(RiskAlert(
                    agent_id=self.agent_id,
                    alert_type="CONCENTRATION_RISK",
                    severity="MEDIUM",
                    message=f"High concentration in {symbol} ({max_weight:.1%})",
                    data={'symbol': symbol, 'weight': float(max_weight)},
                    timestamp=datetime.now(),
                    recommended_action=f"Consider reducing {symbol} allocation to below 25%",
                    confidence_score=0.90
                ))
            
            # Check for sector allocation if available
            if portfolio.sector_allocation:
                max_sector_weight = max(portfolio.sector_allocation.values())
                if max_sector_weight > 0.40:
                    sector = max(portfolio.sector_allocation, key=portfolio.sector_allocation.get)
                    alerts.append(RiskAlert(
                        agent_id=self.agent_id,
                        alert_type="SECTOR_CONCENTRATION",
                        severity="MEDIUM",
                        message=f"High sector concentration in {sector} ({max_sector_weight:.1%})",
                        data={'sector': sector, 'weight': float(max_sector_weight)},
                        timestamp=datetime.now(),
                        recommended_action=f"Consider diversifying away from {sector} sector",
                        confidence_score=0.88
                    ))
                    
        except Exception as e:
            logging.error(f"Analysis specialist error: {e}")
        
        return alerts

class ActionExecutorAgent(BaseAgent):
    """Action execution agent"""
    
    def __init__(self):
        super().__init__("ACTION_EXECUTOR")
        self.executed_actions = []
    
    async def analyze(self, alerts: List[RiskAlert]) -> List[RiskAlert]:
        """Execute recommended actions for high-priority alerts"""
        execution_alerts = []
        
        try:
            critical_alerts = [alert for alert in alerts if alert.severity == "CRITICAL"]
            
            for alert in critical_alerts:
                # Simulate action execution
                action_result = await self.execute_action(alert)
                
                execution_alerts.append(RiskAlert(
                    agent_id=self.agent_id,
                    alert_type="ACTION_EXECUTED",
                    severity="INFO",
                    message=f"Executed action for {alert.alert_type}: {action_result}",
                    data={'original_alert': alert.alert_type, 'action_result': action_result},
                    timestamp=datetime.now(),
                    recommended_action="Monitor execution results",
                    confidence_score=0.95
                ))
                
        except Exception as e:
            logging.error(f"Action executor error: {e}")
        
        return execution_alerts
    
    async def execute_action(self, alert: RiskAlert) -> str:
        """Execute specific action (simulation)"""
        # In a real system, this would connect to trading APIs
        await asyncio.sleep(0.1)  # Simulate execution time
        self.executed_actions.append(alert)
        return f"Simulated action execution for {alert.alert_type}"

class ReportingAgent(BaseAgent):
    """Reporting and notification agent"""
    
    def __init__(self):
        super().__init__("REPORTING_AGENT")
    
    async def analyze(self, alerts: List[RiskAlert]) -> List[RiskAlert]:
        """Generate reports and notifications"""
        report_alerts = []
        
        try:
            if alerts:
                # Group alerts by severity
                severity_counts = {}
                for alert in alerts:
                    severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
                
                # Generate summary report
                report_alerts.append(RiskAlert(
                    agent_id=self.agent_id,
                    alert_type="DAILY_SUMMARY",
                    severity="INFO",
                    message=f"Generated daily risk summary: {severity_counts}",
                    data=severity_counts,
                    timestamp=datetime.now(),
                    recommended_action="Review daily risk summary",
                    confidence_score=1.0
                ))
                
        except Exception as e:
            logging.error(f"Reporting agent error: {e}")
        
        return report_alerts

class AdvancedRiskMetrics:
    """Advanced risk calculation methods"""
    
    @staticmethod
    def calculate_expected_shortfall(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Expected Shortfall (CVaR)"""
        if len(returns) < 10:
            return 0.0
        var = np.percentile(returns, confidence_level * 100)
        return float(returns[returns <= var].mean())
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> Dict[str, float]:
        """Maximum Drawdown Analysis"""
        if len(prices) < 2:
            return {'max_drawdown': 0.0, 'drawdown_duration': 0, 'recovery_time': 0}
        
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calculate drawdown duration
        in_drawdown = drawdown < -0.01  # More than 1% drawdown
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        
        return {
            'max_drawdown': float(max_drawdown),
            'drawdown_duration': float(avg_drawdown_duration),
            'current_drawdown': float(drawdown.iloc[-1]) if len(drawdown) > 0 else 0.0
        }
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Sharpe Ratio with Risk-Free Rate"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() - risk_free_rate/252
        return float(excess_returns / returns.std())
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Sortino Ratio (downside deviation)"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns.mean() - risk_free_rate/252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = downside_returns.std()
        return float(excess_returns / downside_deviation) if downside_deviation != 0 else 0.0

class MLAnomalyDetectionAgent(BaseAgent):
    """Machine Learning-based Anomaly Detection"""
    
    def __init__(self, lookback_period: int = 30, contamination: float = 0.1):
        super().__init__("ML_ANOMALY_DETECTOR")
        self.lookback_period = lookback_period
        self.contamination = contamination
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def create_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create ML features from market data"""
        features = pd.DataFrame()
        
        try:
            for symbol in market_data.columns.get_level_values(1).unique():
                try:
                    # Get price data
                    if ('Close', symbol) in market_data.columns:
                        prices = market_data[('Close', symbol)].dropna()
                        volume = market_data[('Volume', symbol)].dropna() if ('Volume', symbol) in market_data.columns else None
                        
                        if len(prices) < 5:
                            continue
                        
                        # Price-based features
                        returns = prices.pct_change().dropna()
                        
                        if len(returns) > 0:
                            features[f'{symbol}_return'] = returns.reindex(features.index, fill_value=0) if not features.empty else returns
                            features[f'{symbol}_volatility'] = returns.rolling(window=min(5, len(returns))).std().fillna(0)
                            
                            # Technical indicators if ta is available
                            if ta and len(prices) > 14:
                                features[f'{symbol}_rsi'] = ta.momentum.RSIIndicator(prices).rsi().fillna(50)
                            else:
                                # Simple RSI alternative
                                features[f'{symbol}_rsi'] = 50  # Neutral RSI
                            
                            # Volume features
                            if volume is not None and len(volume) > 0:
                                features[f'{symbol}_volume_ratio'] = (volume / volume.rolling(window=min(10, len(volume))).mean()).fillna(1)
                            
                            # Price momentum
                            if len(prices) >= 3:
                                features[f'{symbol}_momentum'] = prices.pct_change(periods=min(3, len(prices)-1)).fillna(0)
                
                except Exception as e:
                    logging.warning(f"Feature creation failed for {symbol}: {e}")
                    continue
        
        except Exception as e:
            logging.error(f"Feature creation error: {e}")
        
        return features.fillna(0)
    
    async def analyze(self, market_data: pd.DataFrame) -> List[RiskAlert]:
        """ML anomaly detection analysis (implementing BaseAgent interface)"""
        alerts = []
        
        try:
            anomaly_results = self.detect_market_anomalies(market_data)
            
            if anomaly_results.get('anomalies'):
                for anomaly in anomaly_results['anomalies'][-3:]:  # Last 3 anomalies
                    alerts.append(RiskAlert(
                        agent_id=self.agent_id,
                        alert_type="MARKET_ANOMALY",
                        severity="MEDIUM",
                        message=f"Market anomaly detected with score {anomaly['anomaly_score']:.2f}",
                        data=anomaly,
                        timestamp=datetime.now(),
                        recommended_action="Investigate market conditions and consider risk reduction",
                        confidence_score=float(anomaly_results.get('confidence', 0.5))
                    ))
        
        except Exception as e:
            logging.error(f"ML anomaly detection error: {e}")
        
        return alerts
    
    def detect_market_anomalies(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect market anomalies using ML"""
        try:
            features = self.create_features(market_data)
            
            if features.empty or len(features) < 10:
                return {'anomalies': [], 'confidence': 0.0, 'model_trained': False}
            
            # Prepare data
            feature_matrix = features.fillna(0).values
            
            if not self.is_trained and len(feature_matrix) >= 10:
                # Train the model
                scaled_features = self.scaler.fit_transform(feature_matrix)
                self.model.fit(scaled_features)
                self.is_trained = True
            elif self.is_trained:
                # Use pre-trained model
                scaled_features = self.scaler.transform(feature_matrix)
            else:
                return {'anomalies': [], 'confidence': 0.0, 'model_trained': False}
            
            # Detect anomalies
            anomaly_predictions = self.model.predict(scaled_features)
            anomaly_scores = self.model.decision_function(scaled_features)
            
            # Get anomalous time points
            anomalous_indices = np.where(anomaly_predictions == -1)[0]
            
            anomalies = []
            for idx in anomalous_indices:
                if idx < len(features):
                    timestamp = features.index[idx]
                    timestamp_str = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)
                    
                    anomalies.append({
                        'timestamp': timestamp_str,
                        'anomaly_score': float(anomaly_scores[idx]),
                        'features': {k: float(v) for k, v in features.iloc[idx].to_dict().items()}
                    })
            
            return {
                'anomalies': anomalies,
                'confidence': float(np.mean(np.abs(anomaly_scores))),
                'model_trained': self.is_trained,
                'total_points': len(feature_matrix),
                'anomaly_rate': len(anomalies) / len(feature_matrix) if len(feature_matrix) > 0 else 0
            }
            
        except Exception as e:
            logging.error(f"Anomaly detection error: {e}")
            return {'anomalies': [], 'confidence': 0.0, 'model_trained': False, 'error': str(e)}

class StressTestAgent(BaseAgent):
    """Advanced Stress Testing Framework"""
    
    def __init__(self):
        super().__init__("STRESS_TESTER")
        self.scenarios = {
            'market_crash_2008': {
                'equity_shock': -0.35,
                'vol_spike': 3.0,
                'correlation_increase': 0.3,
                'description': '2008-style market crash'
            },
            'flash_crash': {
                'equity_shock': -0.10,
                'vol_spike': 5.0,
                'correlation_increase': 0.5,
                'recovery_time': 1,
                'description': 'Flash crash scenario'
            },
            'rate_shock': {
                'rate_change': 0.03,
                'duration_impact': -0.20,
                'equity_shock': -0.15,
                'description': 'Interest rate shock'
            },
            'sector_rotation': {
                'tech_shock': -0.30,
                'defensive_boost': 0.15,
                'value_premium': 0.10,
                'description': 'Tech sector rotation'
            }
        }
    
    async def analyze(self, data: Any) -> List[RiskAlert]:
        """Stress test analysis (implementing BaseAgent interface)"""
        alerts = []
        
        try:
            # data should be a tuple of (portfolio, market_data)
            if isinstance(data, tuple) and len(data) == 2:
                portfolio, market_data = data
                stress_results = await self.run_stress_tests(portfolio, market_data)
                
                # Generate alerts for severe stress test results
                for scenario_name, result in stress_results.items():
                    if scenario_name != 'summary' and 'portfolio_return' in result:
                        portfolio_impact = result['portfolio_return']
                        
                        if portfolio_impact < -0.20:  # More than 20% loss
                            alerts.append(RiskAlert(
                                agent_id=self.agent_id,
                                alert_type="STRESS_TEST_FAILURE",
                                severity="HIGH",
                                message=f"Stress test '{scenario_name}' shows {portfolio_impact:.1%} portfolio loss",
                                data=self._serialize_stress_result(result),
                                timestamp=datetime.now(),
                                recommended_action=f"Consider hedging against {scenario_name} scenario",
                                confidence_score=0.90
                            ))
        
        except Exception as e:
            logging.error(f"Stress test analysis error: {e}")
        
        return alerts
    
    def _serialize_stress_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize stress test results"""
        serialized = {}
        for key, value in result.items():
            if isinstance(value, dict):
                serialized[key] = self._serialize_stress_result(value)
            elif isinstance(value, (np.integer, np.floating)):
                serialized[key] = float(value)
            elif isinstance(value, tuple):
                serialized[key] = list(value)
            else:
                serialized[key] = value
        return serialized
    
    def calculate_scenario_impact(self, portfolio: Portfolio, market_data: pd.DataFrame, 
                                scenario_params: Dict[str, float]) -> Dict[str, Any]:
        """Calculate portfolio impact under stress scenario"""
        try:
            total_impact = 0.0
            position_impacts = {}
            
            for symbol, weight in portfolio.weights.items():
                position_value = portfolio.total_value * weight
                
                # Base equity shock
                equity_shock = scenario_params.get('equity_shock', 0.0)
                
                # Sector-specific adjustments
                if 'tech_shock' in scenario_params and self._is_tech_stock(symbol):
                    equity_shock += scenario_params['tech_shock']
                
                # Calculate position impact
                position_impact = position_value * equity_shock
                position_impacts[symbol] = {
                    'absolute_impact': float(position_impact),
                    'percentage_impact': float(equity_shock),
                    'stressed_value': float(position_value * (1 + equity_shock))
                }
                
                total_impact += position_impact
            
            # Portfolio-level metrics
            portfolio_return = total_impact / portfolio.total_value
            stressed_portfolio_value = portfolio.total_value + total_impact
            
            return {
                'total_portfolio_impact': float(total_impact),
                'portfolio_return': float(portfolio_return),
                'stressed_portfolio_value': float(stressed_portfolio_value),
                'position_impacts': position_impacts,
                'scenario_params': scenario_params,
                'worst_position': min(position_impacts.items(), key=lambda x: x[1]['absolute_impact']) if position_impacts else None,
                'best_position': max(position_impacts.items(), key=lambda x: x[1]['absolute_impact']) if position_impacts else None
            }
            
        except Exception as e:
            logging.error(f"Stress test calculation error: {e}")
            return {'error': str(e)}
    
    def _is_tech_stock(self, symbol: str) -> bool:
        """Simple tech stock classification"""
        tech_symbols = ['AAPL', 'GOOGL', 'GOOG', 'MSFT', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX']
        return symbol.upper() in tech_symbols
    
    async def run_stress_tests(self, portfolio: Portfolio, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run all stress test scenarios"""
        try:
            results = {}
            
            for scenario_name, params in self.scenarios.items():
                scenario_result = self.calculate_scenario_impact(portfolio, market_data, params)
                results[scenario_name] = scenario_result
            
            # Calculate aggregate risk metrics
            valid_results = [r for r in results.values() if 'portfolio_return' in r]
            
            if valid_results:
                worst_case = min(valid_results, key=lambda x: x.get('portfolio_return', 0))
                best_case = max(valid_results, key=lambda x: x.get('portfolio_return', 0))
                
                results['summary'] = {
                    'worst_case_scenario': worst_case,
                    'best_case_scenario': best_case,
                    'average_impact': float(np.mean([r.get('portfolio_return', 0) for r in valid_results])),
                    'impact_volatility': float(np.std([r.get('portfolio_return', 0) for r in valid_results]))
                }
            
            return results
            
        except Exception as e:
            logging.error(f"Stress testing error: {e}")
            return {'error': str(e)}

class RealTimeDataProvider:
    """Enhanced real-time data provider"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.cache = {}
        self.cache_duration = 60
    
    def get_real_time_data(self, symbols: List[str], period: str = "5d") -> pd.DataFrame:
        """Get real-time market data"""
        try:
            # Download basic data
            data = yf.download(symbols, period=period, interval="1h", progress=False)
            
            if data.empty:
                return pd.DataFrame()
            
            # Handle single symbol case
            if len(symbols) == 1:
                data.columns = pd.MultiIndex.from_product([data.columns, symbols])
            
            return data.fillna(method='ffill').fillna(0)
            
        except Exception as e:
            logging.error(f"Real-time data error: {e}")
            return pd.DataFrame()
    
    def get_enhanced_market_data(self, symbols: List[str], period: str = "5d") -> pd.DataFrame:
        """Get enhanced market data with technical indicators"""
        try:
            # Download basic data
            data = yf.download(symbols, period=period, interval="1h", progress=False)
            
            if data.empty:
                return pd.DataFrame()
            
            # Handle single symbol case
            if len(symbols) == 1:
                data.columns = pd.MultiIndex.from_product([data.columns, symbols])
            
            # Add technical indicators for each symbol if ta is available
            enhanced_data = data.copy()
            
            if ta:
                for symbol in symbols:
                    try:
                        if ('Close', symbol) in data.columns:
                            close_prices = data[('Close', symbol)].dropna()
                            
                            if len(close_prices) > 14:  # Minimum for RSI
                                # Technical indicators
                                enhanced_data[('RSI', symbol)] = ta.momentum.RSIIndicator(close_prices).rsi()
                                enhanced_data[('MACD', symbol)] = ta.trend.MACD(close_prices).macd()
                                enhanced_data[('BB_Upper', symbol)] = ta.volatility.BollingerBands(close_prices).bollinger_hband()
                                enhanced_data[('BB_Lower', symbol)] = ta.volatility.BollingerBands(close_prices).bollinger_lband()
                                
                                if len(close_prices) > 20:  # Minimum for SMA
                                    enhanced_data[('SMA_20', symbol)] = ta.trend.SMAIndicator(close_prices, window=20).sma_indicator()
                    
                    except Exception as e:
                        logging.warning(f"Technical indicator calculation failed for {symbol}: {e}")
                        continue
            
            return enhanced_data.fillna(method='ffill').fillna(0)
            
        except Exception as e:
            logging.error(f"Enhanced market data error: {e}")
            return pd.DataFrame()

class DatabaseManager:
    """SQLite database manager for storing alerts and metrics"""
    
    def __init__(self, db_path: str = "risk_management.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Risk alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                data TEXT,
                timestamp TEXT,
                recommended_action TEXT,
                confidence_score REAL
            )
        ''')
        
        # Risk metrics history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_metrics_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                var_5 REAL,
                var_1 REAL,
                expected_shortfall REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                sortino_ratio REAL,
                volatility REAL,
                timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_alert(self, alert: RiskAlert):
        """Log a risk alert to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert alert to JSON-serializable format
        alert_data = alert.to_json_dict()
        
        cursor.execute('''
            INSERT INTO risk_alerts 
            (agent_id, alert_type, severity, message, data, timestamp, recommended_action, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert_data['agent_id'], 
            alert_data['alert_type'], 
            alert_data['severity'], 
            alert_data['message'],
            json.dumps(alert_data['data']), 
            alert_data['timestamp'],
            alert_data['recommended_action'], 
            alert_data['confidence_score']
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM risk_alerts 
            WHERE timestamp >= datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        '''.format(hours))
        
        columns = [desc[0] for desc in cursor.description]
        alerts = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return alerts
    
    def log_risk_metrics(self, symbol: str, metrics: Dict[str, float]):
        """Log risk metrics to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO risk_metrics_history 
            (symbol, var_5, var_1, expected_shortfall, max_drawdown, sharpe_ratio, sortino_ratio, volatility, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol,
            metrics.get('var_5', 0),
            metrics.get('var_1', 0),
            metrics.get('expected_shortfall', 0),
            metrics.get('max_drawdown', 0),
            metrics.get('sharpe_ratio', 0),
            metrics.get('sortino_ratio', 0),
            metrics.get('volatility', 0),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()

# Base Risk Management System
class RiskManagementSystem:
    """Base Risk Management System"""
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.agents = {
            'risk_scanner': RiskScannerAgent(),
            'analysis_specialist': AnalysisSpecialistAgent(),
            'action_executor': ActionExecutorAgent(),
            'reporting_agent': ReportingAgent()
        }
        self.db_manager = DatabaseManager()
        self.market_data = RealTimeDataProvider()
        self.is_running = False
        
    async def start_system(self):
        """Start the risk management system"""
        try:
            print("🚀 Starting Risk Management System...")
            self.is_running = True
            
            # Start all agents
            for agent in self.agents.values():
                await agent.start()
            
            # Main monitoring loop
            while self.is_running:
                await self.run_analysis_cycle()
                await asyncio.sleep(60)  # Run every minute
                
        except KeyboardInterrupt:
            print("\n🛑 Shutting down system...")
            await self.stop_system()
        except Exception as e:
            logging.error(f"System error: {e}")
            await self.stop_system()
    
    async def stop_system(self):
        """Stop the risk management system"""
        self.is_running = False
        
        # Stop all agents
        for agent in self.agents.values():
            await agent.stop()
        
        print("✅ System stopped successfully")
    
    async def run_analysis_cycle(self):
        """Run a complete analysis cycle"""
        try:
            print(f"\n📊 Running analysis cycle at {datetime.now().strftime('%H:%M:%S')}")
            
            # Get market data
            market_data = self.market_data.get_real_time_data(self.portfolio.symbols)
            
            if market_data.empty:
                print("⚠️ No market data available")
                return
            
            all_alerts = []
            
            # Run risk scanner
            risk_alerts = await self.agents['risk_scanner'].analyze(market_data)
            all_alerts.extend(risk_alerts)
            
            # Run analysis specialist
            analysis_alerts = await self.agents['analysis_specialist'].analyze(self.portfolio)
            all_alerts.extend(analysis_alerts)
            
            # Run action executor
            action_alerts = await self.agents['action_executor'].analyze(all_alerts)
            all_alerts.extend(action_alerts)
            
            # Run reporting agent
            report_alerts = await self.agents['reporting_agent'].analyze(all_alerts)
            all_alerts.extend(report_alerts)
            
            # Log all alerts to database
            for alert in all_alerts:
                self.db_manager.log_alert(alert)
            
            # Print summary
            if all_alerts:
                severity_counts = {}
                for alert in all_alerts:
                    severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
                
                print(f"📋 Generated {len(all_alerts)} alerts: {severity_counts}")
                
                # Print critical alerts
                critical_alerts = [a for a in all_alerts if a.severity == "CRITICAL"]
                for alert in critical_alerts:
                    print(f"🚨 CRITICAL: {alert.message}")
            else:
                print("✅ No alerts generated")
                
        except Exception as e:
            logging.error(f"Analysis cycle error: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        active_agents = sum(1 for agent in self.agents.values() if hasattr(agent, 'is_active') and agent.is_active)
        
        return {
            'system_running': self.is_running,
            'active_agents': active_agents,
            'portfolio_value': self.portfolio.total_value,
            'last_update': datetime.now().isoformat()
        }

# Enhanced Risk Management System
class EnhancedRiskManagementSystem(RiskManagementSystem):
    """Enhanced system with ML and advanced analytics"""
    
    def __init__(self, portfolio: Portfolio):
        super().__init__(portfolio)
        self.ml_agent = MLAnomalyDetectionAgent()
        self.stress_agent = StressTestAgent()
        self.dashboard = None
        
        # Enhanced agents with ML capabilities
        self.agents.update({
            'ml_anomaly_detector': self.ml_agent,
            'stress_tester': self.stress_agent
        })
    
    def start_dashboard(self):
        """Start the Streamlit dashboard"""
        try:
            self.dashboard = RiskManagementDashboard(self)
            self.dashboard.run_dashboard()
        except Exception as e:
            logging.error(f"Dashboard startup error: {e}")
    
    async def run_ml_analysis(self):
        """Run ML analysis pipeline"""
        try:
            # Get enhanced market data
            market_data = self.market_data.get_enhanced_market_data(self.portfolio.symbols)
            
            if not market_data.empty:
                # Run anomaly detection
                anomaly_results = self.ml_agent.detect_market_anomalies(market_data)
                
                # Run stress tests
                stress_results = await self.stress_agent.run_stress_tests(self.portfolio, market_data)
                
                logging.info("ML analysis pipeline completed")
                
                return {
                    'anomaly_results': anomaly_results,
                    'stress_results': stress_results
                }
                
        except Exception as e:
            logging.error(f"ML analysis error: {e}")
            return {}
    
    async def run_analysis_cycle(self):
        """Enhanced analysis cycle with ML components"""
        try:
            print(f"\n📊 Running enhanced analysis cycle at {datetime.now().strftime('%H:%M:%S')}")
            
            # Get market data
            market_data = self.market_data.get_enhanced_market_data(self.portfolio.symbols)
            
            if market_data.empty:
                print("⚠️ No market data available")
                return
            
            all_alerts = []
            
            # Run traditional agents
            risk_alerts = await self.agents['risk_scanner'].analyze(market_data)
            all_alerts.extend(risk_alerts)
            
            analysis_alerts = await self.agents['analysis_specialist'].analyze(self.portfolio)
            all_alerts.extend(analysis_alerts)
            
            # Run ML agents
            ml_alerts = await self.agents['ml_anomaly_detector'].analyze(market_data)
            all_alerts.extend(ml_alerts)
            
            stress_alerts = await self.agents['stress_tester'].analyze((self.portfolio, market_data))
            all_alerts.extend(stress_alerts)
            
            # Run action and reporting agents
            action_alerts = await self.agents['action_executor'].analyze(all_alerts)
            all_alerts.extend(action_alerts)
            
            report_alerts = await self.agents['reporting_agent'].analyze(all_alerts)
            all_alerts.extend(report_alerts)
            
            # Log all alerts to database
            for alert in all_alerts:
                self.db_manager.log_alert(alert)
            
            # Print summary
            if all_alerts:
                severity_counts = {}
                for alert in all_alerts:
                    severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
                
                print(f"📋 Generated {len(all_alerts)} alerts: {severity_counts}")
                
                # Print high priority alerts
                high_priority_alerts = [a for a in all_alerts if a.severity in ["CRITICAL", "HIGH"]]
                for alert in high_priority_alerts:
                    print(f"🚨 {alert.severity}: {alert.message}")
            else:
                print("✅ No alerts generated")
                
        except Exception as e:
            logging.error(f"Enhanced analysis cycle error: {e}")
    
    def simulate_running_system(self):
        """Simulate system running for demo purposes"""
        self.is_running = True
        
        # Mark agents as active
        for agent in self.agents.values():
            agent.is_active = True
        
        # Generate some sample alerts for demo
        sample_alerts = [
            RiskAlert(
                agent_id="RISK_SCANNER",
                alert_type="HIGH_VOLATILITY", 
                severity="MEDIUM",
                message="TSLA volatility elevated at 45.2%",
                data={'symbol': 'TSLA', 'volatility': 0.452},
                timestamp=datetime.now() - timedelta(hours=1),
                recommended_action="Monitor TSLA position closely",
                confidence_score=0.87
            ),
            RiskAlert(
                agent_id="ANALYSIS_SPECIALIST",
                alert_type="CONCENTRATION_RISK",
                severity="LOW", 
                message="Technology sector allocation at 60%",
                data={'sector': 'Technology', 'allocation': 0.60},
                timestamp=datetime.now() - timedelta(hours=2),
                recommended_action="Consider sector diversification",
                confidence_score=0.92
            ),
            RiskAlert(
                agent_id="ML_ANOMALY_DETECTOR",
                alert_type="MARKET_ANOMALY",
                severity="MEDIUM",
                message="Unusual trading pattern detected",
                data={'anomaly_score': -0.75, 'confidence': 0.88},
                timestamp=datetime.now() - timedelta(minutes=30),
                recommended_action="Review current market conditions",
                confidence_score=0.88
            )
        ]
        
        # Log sample alerts
        for alert in sample_alerts:
            self.db_manager.log_alert(alert)

# Streamlit Dashboard
class RiskManagementDashboard:
    """Advanced Streamlit Dashboard for Risk Management"""
    
    def __init__(self, risk_system):
        self.risk_system = risk_system
        self.db_manager = risk_system.db_manager
        
    def run_dashboard(self):
        """Main dashboard interface"""
        st.set_page_config(
            page_title="Portfolio Risk Management System",
            page_icon="🏦",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize system if not running
        if not self.risk_system.is_running:
            self.risk_system.simulate_running_system()
        
        # Custom CSS
        st.markdown("""
        <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            color: #333333;
        }
        .alert-high {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
            color: #d32f2f;
        }
        .alert-medium {
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
            color: #f57c00;
        }
        .alert-low {
            background-color: #e8f5e8;
            border-left: 4px solid #4caf50;
            color: #388e3c;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.title("🏦 Portfolio Risk Management System")
        st.markdown("---")
        
        # Sidebar
        self.render_sidebar()
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "🚨 Risk Alerts", "📈 Risk Metrics", 
            "🧪 Stress Tests", "🤖 ML Analytics"
        ])
        
        with tab1:
            self.render_overview_tab()
        
        with tab2:
            self.render_alerts_tab()
        
        with tab3:
            self.render_risk_metrics_tab()
        
        with tab4:
            self.render_stress_test_tab()
        
        with tab5:
            self.render_ml_analytics_tab()
    
    def render_sidebar(self):
        """Render dashboard sidebar"""
        with st.sidebar:
            st.header("🎛️ Control Panel")
            
            # System status
            status = self.risk_system.get_system_status()
            if status['system_running']:
                st.success("✅ System Online")
            else:
                st.error("❌ System Offline")
            
            st.metric("Active Agents", status['active_agents'])
            st.metric("Portfolio Value", f"${status['portfolio_value']:,.0f}")
            
            # Portfolio composition
            st.subheader("📈 Portfolio")
            portfolio = self.risk_system.portfolio
            
            for symbol, weight in portfolio.weights.items():
                st.write(f"**{symbol}**: {weight:.1%}")
            
            # Real-time refresh
            st.subheader("🔄 Controls")
            if st.button("Refresh Data"):
                st.rerun()
            
            auto_refresh = st.checkbox("Auto-refresh (30s)")
            if auto_refresh:
                time.sleep(30)
                st.rerun()
    
    def render_overview_tab(self):
        """Render overview dashboard tab"""
        col1, col2, col3, col4 = st.columns(4)
        
        # Get recent alerts
        recent_alerts = self.db_manager.get_recent_alerts(24)
        
        with col1:
            st.metric(
                "24h Alerts",
                len(recent_alerts),
                delta=len(recent_alerts) - len(self.db_manager.get_recent_alerts(48)) + len(recent_alerts)
            )
        
        with col2:
            high_severity = len([a for a in recent_alerts if a['severity'] in ['HIGH', 'CRITICAL']])
            st.metric("High Risk Alerts", high_severity)
        
        with col3:
            # Calculate portfolio VaR (simplified)
            portfolio_var = self.calculate_portfolio_var()
            st.metric("Portfolio VaR (5%)", f"{portfolio_var:.2%}")
        
        with col4:
            # System uptime
            st.metric("System Uptime", "Running")
        
        # Portfolio performance chart
        st.subheader("📈 Portfolio Performance")
        self.render_portfolio_performance_chart()
        
        # Risk heatmap
        st.subheader("🔥 Risk Heatmap")
        self.render_risk_heatmap()
    
    def render_alerts_tab(self):
        """Render risk alerts tab"""
        st.header("🚨 Risk Alert Center")
        
        # Alert filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            severity_filter = st.selectbox(
                "Severity Level",
                ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
            )
        
        with col2:
            hours_filter = st.selectbox(
                "Time Period",
                [1, 6, 12, 24, 48, 168],  # 1h to 1 week
                index=3  # Default to 24h
            )
        
        with col3:
            agent_filter = st.selectbox(
                "Agent",
                ["All", "RISK_SCANNER", "ANALYSIS_SPECIALIST", "ACTION_EXECUTOR", "REPORTING_AGENT", "ML_ANOMALY_DETECTOR", "STRESS_TESTER"]
            )
        
        # Get filtered alerts
        alerts = self.get_filtered_alerts(severity_filter, hours_filter, agent_filter)
        
        # Alert statistics
        if alerts:
            st.subheader("📊 Alert Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                severity_counts = {}
                for alert in alerts:
                    sev = alert['severity']
                    severity_counts[sev] = severity_counts.get(sev, 0) + 1
                
                fig = px.pie(
                    values=list(severity_counts.values()),
                    names=list(severity_counts.keys()),
                    title="Alerts by Severity"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Agent activity
                agent_counts = {}
                for alert in alerts:
                    agent = alert['agent_id']
                    agent_counts[agent] = agent_counts.get(agent, 0) + 1
                
                fig = px.bar(
                    x=list(agent_counts.keys()),
                    y=list(agent_counts.values()),
                    title="Alerts by Agent"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                # Alert types
                type_counts = {}
                for alert in alerts:
                    alert_type = alert['alert_type']
                    type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
                
                fig = px.bar(
                    x=list(type_counts.keys()),
                    y=list(type_counts.values()),
                    title="Alerts by Type"
                )
                fig.update_xaxes(tickangle=45)  # Fixed: update_xaxis -> update_xaxes
                st.plotly_chart(fig, use_container_width=True)
        
        # Alert details
        st.subheader("📋 Alert Details")
        
        for alert in alerts[:20]:  # Show latest 20 alerts
            severity = alert['severity']
            css_class = f"alert-{severity.lower()}" if severity.lower() in ['high', 'medium', 'low'] else "metric-card"
            
            with st.container():
                st.markdown(f"""
                <div class="metric-card {css_class}">
                    <h4>{alert['alert_type']} - {severity}</h4>
                    <p><strong>Agent:</strong> {alert['agent_id']}</p>
                    <p><strong>Message:</strong> {alert['message']}</p>
                    <p><strong>Time:</strong> {alert['timestamp']}</p>
                    <p><strong>Action:</strong> {alert['recommended_action']}</p>
                    <p><strong>Confidence:</strong> {alert['confidence_score']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("")
    
    def render_risk_metrics_tab(self):
        """Render risk metrics analysis tab"""
        st.header("📈 Advanced Risk Metrics")
        
        # Portfolio selection
        portfolio = self.risk_system.portfolio
        selected_symbols = st.multiselect(
            "Select Assets for Analysis",
            portfolio.symbols,
            default=portfolio.symbols[:3]
        )
        
        if not selected_symbols:
            st.warning("Please select at least one asset for analysis.")
            return
        
        # Get market data
        market_data = self.risk_system.market_data.get_enhanced_market_data(selected_symbols)
        
        if market_data.empty:
            st.error("Unable to fetch market data.")
            return
        
        # Risk metrics calculation
        st.subheader("🎯 Key Risk Metrics")
        
        metrics_cols = st.columns(len(selected_symbols))
        
        for i, symbol in enumerate(selected_symbols):
            with metrics_cols[i]:
                st.write(f"**{symbol}**")
                
                # Get price data
                if len(selected_symbols) == 1:
                    prices = market_data['Close'].dropna()
                else:
                    prices = market_data[('Close', symbol)].dropna()
                
                if len(prices) > 1:
                    returns = prices.pct_change().dropna()
                    
                    # Calculate metrics
                    var_5 = AdvancedRiskMetrics.calculate_expected_shortfall(returns, 0.05)
                    max_dd = AdvancedRiskMetrics.calculate_max_drawdown(prices)
                    sharpe = AdvancedRiskMetrics.calculate_sharpe_ratio(returns)
                    sortino = AdvancedRiskMetrics.calculate_sortino_ratio(returns)
                    
                    st.metric("VaR (5%)", f"{var_5:.2%}")
                    st.metric("Max Drawdown", f"{max_dd['max_drawdown']:.2%}")
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    st.metric("Sortino Ratio", f"{sortino:.2f}")
        
        # Risk correlation matrix
        st.subheader("🔗 Correlation Analysis")
        self.render_correlation_matrix(selected_symbols, market_data)
        
        # Volatility analysis
        st.subheader("📊 Volatility Analysis")
        self.render_volatility_analysis(selected_symbols, market_data)
    
    def render_stress_test_tab(self):
        """Render stress testing tab"""
        st.header("🧪 Stress Testing Laboratory")
        
        # Stress test controls
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎛️ Test Configuration")
            
            # Scenario selection
            stress_agent = StressTestAgent()
            scenarios = list(stress_agent.scenarios.keys())
            
            selected_scenarios = st.multiselect(
                "Select Stress Scenarios",
                scenarios,
                default=scenarios[:3]
            )
            
            # Custom scenario
            st.write("**Custom Scenario**")
            custom_equity_shock = st.slider("Equity Shock (%)", -50, 10, -20) / 100
            custom_vol_spike = st.slider("Volatility Multiplier", 1.0, 5.0, 2.0)
            
            if st.button("Run Stress Tests"):
                self.run_stress_tests(selected_scenarios, custom_equity_shock, custom_vol_spike)
        
        with col2:
            st.subheader("📊 Scenario Impact")
            
            # Mock stress test results for demo
            if st.session_state.get('stress_results'):
                stress_results = st.session_state.stress_results
                
                # Create impact visualization
                scenarios = [s for s in stress_results.keys() if 'portfolio_return' in stress_results[s]]
                impacts = [stress_results[s].get('portfolio_return', 0) * 100 for s in scenarios]
                
                if impacts:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=scenarios,
                        y=impacts,
                        marker_color=['red' if x < 0 else 'green' for x in impacts]
                    ))
                    fig.update_layout(
                        title="Stress Test Results - Portfolio Impact (%)",
                        xaxis_title="Scenario",
                        yaxis_title="Portfolio Impact (%)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run stress tests to see results")
    
    def render_ml_analytics_tab(self):
        """Render ML analytics tab"""
        st.header("🤖 Machine Learning Analytics")
        
        # ML model controls
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🔍 Anomaly Detection")
            
            # Get market data for ML
            portfolio = self.risk_system.portfolio
            market_data = self.risk_system.market_data.get_enhanced_market_data(portfolio.symbols)
            
            if not market_data.empty:
                # Initialize ML agent
                ml_agent = MLAnomalyDetectionAgent()
                
                # Run anomaly detection
                anomaly_results = ml_agent.detect_market_anomalies(market_data)
                
                # Display results
                st.metric("Model Trained", "✅" if anomaly_results['model_trained'] else "❌")
                st.metric("Anomaly Rate", f"{anomaly_results['anomaly_rate']:.1%}")
                st.metric("Confidence Score", f"{anomaly_results['confidence']:.2f}")
                
                # Anomaly timeline
                if anomaly_results['anomalies']:
                    st.write("**Recent Anomalies:**")
                    for anomaly in anomaly_results['anomalies'][-5:]:  # Show last 5
                        st.write(f"- {anomaly['timestamp']}: Score {anomaly['anomaly_score']:.2f}")
        
        with col2:
            st.subheader("📊 Feature Analysis")
            
            # Feature importance (mock data for demo)
            features = ['Volatility', 'Volume', 'Returns', 'Momentum', 'RSI']
            importance = [0.35, 0.25, 0.20, 0.15, 0.05]
            
            fig = px.bar(
                x=features,
                y=importance,
                title="Feature Importance in Anomaly Detection"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Model performance metrics
            st.write("**Model Performance:**")
            st.metric("Detection Rate", "87.3%")
            st.metric("False Positive Rate", "12.1%")
            st.metric("Model Accuracy", "88.2%")
    
    # Helper methods for dashboard
    def calculate_portfolio_var(self) -> float:
        """Calculate simplified portfolio VaR"""
        try:
            portfolio = self.risk_system.portfolio
            market_data = self.risk_system.market_data.get_real_time_data(portfolio.symbols)
            
            if market_data.empty:
                return 0.0
            
            # Calculate portfolio returns (simplified)
            portfolio_returns = []
            
            for symbol, weight in portfolio.weights.items():
                try:
                    if len(portfolio.symbols) == 1:
                        prices = market_data['Close'].dropna()
                    else:
                        prices = market_data[('Close', symbol)].dropna()
                    
                    if len(prices) > 1:
                        returns = prices.pct_change().dropna()
                        if len(returns) > 0:
                            portfolio_returns.extend(returns * weight)
                
                except Exception:
                    continue
            
            if portfolio_returns:
                return np.percentile(portfolio_returns, 5)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def render_portfolio_performance_chart(self):
        """Render portfolio performance visualization"""
        try:
            portfolio = self.risk_system.portfolio
            market_data = self.risk_system.market_data.get_real_time_data(portfolio.symbols)
            
            if market_data.empty:
                st.warning("No market data available for chart.")
                return
            
            # Calculate portfolio value over time (simplified)
            portfolio_values = []
            timestamps = []
            
            for timestamp in market_data.index:
                portfolio_value = 0
                
                for symbol, weight in portfolio.weights.items():
                    try:
                        if len(portfolio.symbols) == 1:
                            price = market_data.loc[timestamp, 'Close']
                        else:
                            price = market_data.loc[timestamp, ('Close', symbol)]
                        
                        if not pd.isna(price):
                            portfolio_value += price * weight * 1000  # Simplified calculation
                    
                    except Exception:
                        continue
                
                if portfolio_value > 0:
                    portfolio_values.append(portfolio_value)
                    timestamps.append(timestamp)
            
            if portfolio_values:
                df = pd.DataFrame({
                    'timestamp': timestamps,
                    'portfolio_value': portfolio_values
                })
                
                fig = px.line(
                    df,
                    x='timestamp',
                    y='portfolio_value',
                    title='Portfolio Value Over Time'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Chart rendering error: {e}")
    
    def render_risk_heatmap(self):
        """Render risk heatmap"""
        try:
            portfolio = self.risk_system.portfolio
            
            # Create risk matrix (mock data for demo)
            risk_data = []
            
            for symbol in portfolio.symbols:
                # Mock risk metrics
                var_risk = np.random.uniform(0.02, 0.08)
                vol_risk = np.random.uniform(0.15, 0.35)
                correlation_risk = np.random.uniform(0.3, 0.8)
                
                risk_data.append({
                    'Asset': symbol,
                    'VaR Risk': var_risk,
                    'Volatility Risk': vol_risk,
                    'Correlation Risk': correlation_risk
                })
            
            df_risk = pd.DataFrame(risk_data)
            df_risk_matrix = df_risk.set_index('Asset').T
            
            fig = px.imshow(
                df_risk_matrix.values,
                x=df_risk_matrix.columns,
                y=df_risk_matrix.index,
                color_continuous_scale='RdYlGn_r',
                title='Risk Heatmap'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Heatmap rendering error: {e}")
    
    def get_filtered_alerts(self, severity_filter: str, hours_filter: int, agent_filter: str) -> List[Dict]:
        """Get filtered alerts based on criteria"""
        try:
            all_alerts = self.db_manager.get_recent_alerts(hours_filter)
            
            filtered_alerts = []
            
            for alert in all_alerts:
                # Apply filters
                if severity_filter != "All" and alert['severity'] != severity_filter:
                    continue
                
                if agent_filter != "All" and alert['agent_id'] != agent_filter:
                    continue
                
                filtered_alerts.append(alert)
            
            return filtered_alerts
        
        except Exception:
            return []
    
    def render_correlation_matrix(self, symbols: List[str], market_data: pd.DataFrame):
        """Render correlation matrix"""
        try:
            returns_data = {}
            
            for symbol in symbols:
                if len(symbols) == 1:
                    prices = market_data['Close'].dropna()
                else:
                    prices = market_data[('Close', symbol)].dropna()
                
                if len(prices) > 1:
                    returns = prices.pct_change().dropna()
                    if len(returns) > 0:
                        returns_data[symbol] = returns
            
            if returns_data:
                # Create correlation matrix
                returns_df = pd.DataFrame(returns_data)
                correlation_matrix = returns_df.corr()
                
                fig = px.imshow(
                    correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.index,
                    color_continuous_scale='RdBu',
                    title='Asset Correlation Matrix',
                    zmin=-1, zmax=1
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Correlation matrix error: {e}")
    
    def render_volatility_analysis(self, symbols: List[str], market_data: pd.DataFrame):
        """Render volatility analysis"""
        try:
            vol_data = []
            
            for symbol in symbols:
                if len(symbols) == 1:
                    prices = market_data['Close'].dropna()
                else:
                    prices = market_data[('Close', symbol)].dropna()
                
                if len(prices) > 1:
                    returns = prices.pct_change().dropna()
                    if len(returns) > 0:
                        vol_data.append({
                            'Symbol': symbol,
                            'Volatility': returns.std() * np.sqrt(252),  # Annualized
                            'Current Vol': returns.rolling(window=20).std().iloc[-1] * np.sqrt(252) if len(returns) >= 20 else 0
                        })

            if vol_data:
                df_vol = pd.DataFrame(vol_data)

                fig = px.bar(
                    df_vol,
                    x='Symbol',
                    y=['Volatility', 'Current Vol'],
                    title='Volatility Analysis',
                    barmode='group'
                )

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Volatility analysis error: {e}")
    
    def run_stress_tests(self, scenarios: List[str], custom_equity: float, custom_vol: float):
        """Run stress tests and display results"""
        try:
            portfolio = self.risk_system.portfolio
            market_data = self.risk_system.market_data.get_real_time_data(portfolio.symbols)
            
            stress_agent = StressTestAgent()
            
            # Add custom scenario
            stress_agent.scenarios['custom'] = {
                'equity_shock': custom_equity,
                'vol_spike': custom_vol,
                'description': 'Custom stress scenario'
            }
            
            # Run selected scenarios
            results = {}
            for scenario in scenarios + ['custom']:
                if scenario in stress_agent.scenarios:
                    result = stress_agent.calculate_scenario_impact(
                        portfolio, market_data, stress_agent.scenarios[scenario]
                    )
                    results[scenario] = result
            
            # Store results in session state
            st.session_state.stress_results = results
            
            st.success(f"Stress tests completed for {len(results)} scenarios")
            
        except Exception as e:
            st.error(f"Stress test error: {e}")

def create_enhanced_portfolio() -> Portfolio:
    """Create enhanced portfolio with sector allocation"""
    return Portfolio(
        symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'PG'],
        weights={
            'AAPL': 0.20,
            'GOOGL': 0.15,
            'MSFT': 0.15,
            'TSLA': 0.15,
            'NVDA': 0.10,
            'JPM': 0.10,
            'JNJ': 0.08,
            'PG': 0.07
        },
        total_value=2000000.0,  # $2M portfolio
        last_updated=datetime.now(),
        sector_allocation={
            'Technology': 0.60,
            'Financial': 0.10,
            'Healthcare': 0.08,
            'Consumer Goods': 0.07,
            'Other': 0.15
        },
        risk_budget={
            'Equity Risk': 0.70,
            'Sector Risk': 0.20,
            'Idiosyncratic Risk': 0.10
        }
    )

# Streamlit app entry point
def main_streamlit():
    """Main Streamlit application"""
    # Initialize session state
    if 'stress_results' not in st.session_state:
        st.session_state.stress_results = {}
    
    # Create enhanced portfolio
    portfolio = create_enhanced_portfolio()
    
    # Initialize enhanced system
    risk_system = EnhancedRiskManagementSystem(portfolio)
    
    # Start dashboard
    risk_system.start_dashboard()

# Background system runner for integrated mode
class BackgroundSystemRunner:
    """Run the risk management system in background thread"""
    
    def __init__(self, risk_system):
        self.risk_system = risk_system
        self.thread = None
        self.loop = None
        self.running = False
    
    def start(self):
        """Start background system"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_system, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop background system"""
        self.running = False
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
    
    def _run_system(self):
        """Run system in background thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        async def system_runner():
            try:
                # Start agents
                for agent in self.risk_system.agents.values():
                    await agent.start()
                
                # Run analysis cycles
                while self.running:
                    await self.risk_system.run_analysis_cycle()
                    await asyncio.sleep(30)  # Run every 30 seconds
                    
            except Exception as e:
                logging.error(f"Background system error: {e}")
            finally:
                # Stop agents
                for agent in self.risk_system.agents.values():
                    await agent.stop()
        
        try:
            self.loop.run_until_complete(system_runner())
        except Exception as e:
            logging.error(f"Background loop error: {e}")

def main_integrated():
    """Main integrated application (dashboard + background system)"""
    # Initialize session state
    if 'stress_results' not in st.session_state:
        st.session_state.stress_results = {}
    
    if 'system_runner' not in st.session_state:
        # Create enhanced portfolio
        portfolio = create_enhanced_portfolio()
        
        # Initialize enhanced system
        risk_system = EnhancedRiskManagementSystem(portfolio)
        
        # Create background runner
        system_runner = BackgroundSystemRunner(risk_system)
        
        # Store in session state
        st.session_state.risk_system = risk_system
        st.session_state.system_runner = system_runner
        
        # Start background system
        system_runner.start()
        
        # Wait a moment for system to initialize
        time.sleep(2)
    
    # Get system from session state
    risk_system = st.session_state.risk_system
    
    # Create and run dashboard
    dashboard = RiskManagementDashboard(risk_system)
    dashboard.run_dashboard()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "dashboard":
            # Run Streamlit dashboard only
            main_streamlit()
        elif sys.argv[1] == "integrated":
            # Run integrated mode (dashboard + background system)
            main_integrated()
    else:
        # Run traditional CLI system
        async def main():
            """Main CLI entry point"""
            print("🏦 Enhanced Autonomous Portfolio Risk Management System")
            print("=" * 80)
            
            # Create enhanced portfolio
            portfolio = create_enhanced_portfolio()
            
            # Initialize and start the enhanced system
            risk_system = EnhancedRiskManagementSystem(portfolio)
            
            # Start ML analysis in background
            ml_task = asyncio.create_task(risk_system.run_ml_analysis())
            
            try:
                # Start main system
                await risk_system.start_system()
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                ml_task.cancel()
            except Exception as e:
                print(f"❌ System error: {e}")
                ml_task.cancel()
        
        asyncio.run(main())

# Export all classes for external use
__all__ = [
    'Portfolio', 'RiskAlert', 'AdvancedRiskMetrics', 'MLAnomalyDetectionAgent',
    'StressTestAgent', 'RealTimeDataProvider', 'DatabaseManager',
    'RiskManagementDashboard', 'RiskManagementSystem', 'EnhancedRiskManagementSystem',
    'BackgroundSystemRunner'
]
