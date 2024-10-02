import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
from streamlit_option_menu import option_menu
import base64
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
import io
from sklearn.preprocessing import MinMaxScaler

base_xgb_model = joblib.load('xgb_model.joblib')
# API key (consider using environment variables for security)
API_KEY = "5KX601WBN1NA9XAN"

# Function to fetch data from Alpha Vantage
def fetch_data(url):
    # Send a GET request to the provided URL
    response = requests.get(url)
    # If the request is successful (status code 200), return the JSON data
    if response.status_code == 200:
        return response.json()
    # If the request fails, display an error message and return None
    else:
        st.error(f"Error fetching data: {response.status_code}")
        return None

# Function to get company overview
import logging

logging.basicConfig(level=logging.INFO)

import yfinance as yf

# Function to prepare features for the XGBoost model
def prepare_features(df):
    # Calculate the percentage change in price and create lag features
    df.loc[:, 'returns'] = df['price'].pct_change()
    df.loc[:, 'lag_1'] = df['price'].shift(1)
    df.loc[:, 'lag_2'] = df['price'].shift(2)
    df.loc[:, 'lag_3'] = df['price'].shift(3)
    # Return the prepared features
    return df[['lag_1', 'lag_2', 'lag_3', 'returns']]

# Function to fine-tune the XGBoost model for a specific stock
def fine_tune_model(symbol):
    # Fetch historical weekly data for the given stock symbol
    df = yf.download(symbol, start='2010-01-01', end=pd.Timestamp.now().strftime('%Y-%m-%d'), interval='1wk')
    df = df['Adj Close'].to_frame('price')
    
    # If the data is empty, return None, None
    if df.empty:
        return None, None
    
    # Prepare features for the XGBoost model
    X = prepare_features(df)
    y = df['price']
    X = X.dropna()
    y = y.iloc[3:]  # Align y with X after creating lag features
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a new model with the same parameters as the base model
    new_model = XGBRegressor(**base_xgb_model.get_params())
    
    # Fine-tune the model by fitting it to the training data
    new_model.fit(X_train, y_train)
    
    # Evaluate the model's performance on the testing set
    y_pred = new_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Return the fine-tuned model and its RMSE
    return new_model, rmse

def get_drive_service():
    creds_json = {
        "type": "service_account",
        "project_id": "the-eye-437304",
        "client_email": "the-eye@the-eye-437304.iam.gserviceaccount.com",
        "private_key": "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_HERE\n-----END PRIVATE KEY-----\n"
    }
    creds = Credentials.from_service_account_info(creds_json)
    return build('drive', 'v3', credentials=creds)

@st.cache_resource
def download_model(file_id, output_path):
    if not os.path.exists(output_path):
        service = get_drive_service()
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")
        fh.seek(0)
        with open(output_path, 'wb') as f:
            f.write(fh.read())
    return output_path

# Updated file ID and service account email
sarima_model_file_id = '1o5790y30fby6p5Vfpx6o1umQbSL8hVwY'
service_account_email = 'the-eye@the-eye-437304.iam.gserviceaccount.com'

# Remove or comment out SARIMA model loading
# sarima_model_path = download_model(sarima_model_file_id, 'sarima_model.joblib')
# sarima_model = joblib.load(sarima_model_path)

# Function to predict stock prices using XGBoost and SARIMA models
def predict_stock_price(symbol, weeks=52):
    # Fine-tune the XGBoost model for the specific stock
    xgb_model, rmse = fine_tune_model(symbol)
    
    # If the XGBoost model is None, return None, None
    if xgb_model is None:
        return None, None
    
    # Fetch the most recent data for prediction
    df = yf.download(symbol, start='2010-01-01', end=pd.Timestamp.now().strftime('%Y-%m-%d'), interval='1wk')
    df = df['Adj Close'].to_frame('price')
    X = prepare_features(df)
    X = X.dropna()
    
    # Make XGBoost predictions
    xgb_pred = []
    current_data = X.iloc[-3:].values.flatten()

    for _ in range(weeks):
        features = np.array([[current_data[0], current_data[1], current_data[2], 
                              (current_data[0] - current_data[1]) / current_data[1]]])
        pred = xgb_model.predict(features)[0]
        xgb_pred.append(pred)
        current_data = np.roll(current_data, -1)
        current_data[-1] = pred
    
    # Make SARIMA predictions
    # sarima_pred = sarima_model.forecast(steps=weeks)
    
    # Average the predictions
    # avg_pred = [(x + y) / 2 for x, y in zip(xgb_pred, sarima_pred)]
    
    # Create a DataFrame with dates and predictions
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=7), periods=weeks, freq='W')
    forecast_df = pd.DataFrame({
        'Date': future_dates, 
        'XGB_Predicted_Price': xgb_pred,
        # 'SARIMA_Predicted_Price': sarima_pred,
        # 'Avg_Predicted_Price': avg_pred
    })
    forecast_df.set_index('Date', inplace=True)
    
    # Return the forecast DataFrame and the RMSE of the XGBoost model
    return forecast_df, rmse

# Function to display stock price predictions
def display_stock_prediction(symbol):
    st.subheader(f"Stock Price Prediction: {symbol}")
    
    # Set a fixed number of weeks for prediction
    weeks = 52  # You can adjust this value as needed
    
    with st.spinner("Generating prediction..."):
        predictions, rmse = predict_stock_price(symbol, weeks)
    
    if predictions is not None:
        # Get historical data for comparison
        historical_data = yf.download(symbol, start='2023-01-01', end=pd.Timestamp.now().strftime('%Y-%m-%d'), interval='1wk')
        historical_data = historical_data['Adj Close'].to_frame('price')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['price'], mode='lines', name='Historical Price'))
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions['XGB_Predicted_Price'], mode='lines', name='XGBoost Prediction'))
        # fig.add_trace(go.Scatter(x=predictions.index, y=predictions['SARIMA_Predicted_Price'], mode='lines', name='SARIMA Prediction'))
        # fig.add_trace(go.Scatter(x=predictions.index, y=predictions['Avg_Predicted_Price'], mode='lines', name='Average Prediction'))
        fig.update_layout(title=f"{symbol} Stock Price - Historical and Predicted", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)
        
        st.write("Predicted Prices:")
        st.dataframe(predictions)
        
        # Display prediction summary
        st.write("\nPrediction Summary:")
        st.write(f"Mean (XGBoost Prediction): ${predictions['XGB_Predicted_Price'].mean():.2f}")
        st.write(f"Min (XGBoost Prediction): ${predictions['XGB_Predicted_Price'].min():.2f}")
        st.write(f"Max (XGBoost Prediction): ${predictions['XGB_Predicted_Price'].max():.2f}")
        st.write(f"XGBoost Model RMSE: ${rmse:.2f}")
        
        st.warning("Please note that these predictions are based on historical data and a combination of XGBoost and SARIMA models. "
                   "They should not be used as the sole basis for investment decisions. Always conduct thorough "
                   "research and consult with a financial advisor before making investment choices.")
    else:
        st.write("Unable to make predictions for this stock.")

# Function to get company overview data using yfinance
def get_company_overview(symbol):
    try:
        # Fetch company data using yfinance
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Extract relevant information
        overview = {
            'Name': info.get('longName', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'MarketCapitalization': info.get('marketCap', 'N/A'),
            'PERatio': info.get('trailingPE', 'N/A'),
            'DividendYield': info.get('dividendYield', 'N/A'),
            '52WeekHigh': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52WeekLow': info.get('fiftyTwoWeekLow', 'N/A'),
            'EPS': info.get('trailingEps', 'N/A'),
            'Beta': info.get('beta', 'N/A'),
            'ProfitMargin': info.get('profitMargins', 'N/A'),
            'Description': info.get('longBusinessSummary', 'N/A')
        }
        # Return the overview data
        return overview
    except Exception as e:
        # Handle any errors that occur during data fetching
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

# Function to display company overview information
def display_company_overview(symbol):
    # Get the company overview data
    overview = get_company_overview(symbol)
    if overview:
        st.subheader(f"Company Overview: {overview['Name']}")
        
        # Real-time price
        real_time_price = get_real_time_price(symbol)
        if real_time_price:
            st.metric("Current Stock Price", f"${real_time_price:.2f}")
        
        # Display company information in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Sector:** {overview['Sector']}")
            st.write(f"**Industry:** {overview['Industry']}")
            st.write(f"**Market Cap:** ${overview['MarketCapitalization']:,}")
        with col2:
            st.write(f"**P/E Ratio:** {overview['PERatio']}")
            st.write(f"**Dividend Yield:** {overview['DividendYield']}")
            st.write(f"**52 Week High/Low:** ${overview['52WeekHigh']} / ${overview['52WeekLow']}")
        with col3:
            st.write(f"**EPS:** ${overview['EPS']}")
            st.write(f"**Beta:** {overview['Beta']}")
            st.write(f"**Profit Margin:** {overview['ProfitMargin']}")
        
        # Display company description in an expandable section
        with st.expander("Company Description"):
            st.write(overview['Description'])
    else:
        st.warning(f"Unable to fetch company overview data for {symbol}")

# Function to get real-time stock price
def get_real_time_price(symbol):
    # Fetch the most recent data for the given symbol
    stock = yf.Ticker(symbol)
    data = stock.history(period="1d")
    if not data.empty:
        # Return the closing price of the most recent day
        return data['Close'].iloc[-1]
    return None

# Function to get historical stock data
def get_historical_data(symbol, period='1y'):
    # Fetch historical data for the given symbol and period
    stock = yf.Ticker(symbol)
    data = stock.history(period=period)
    return data

# Function to display company overview
def display_company_overview(symbol):
    overview = get_company_overview(symbol)
    if overview:
        st.subheader(f"Company Overview: {overview.get('Name', symbol)}")
        
        # Real-time price
        real_time_price = get_real_time_price(symbol)
        if real_time_price:
            st.metric("Current Stock Price", f"${real_time_price:.2f}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Sector:** {overview.get('Sector', 'N/A')}")
            st.write(f"**Industry:** {overview.get('Industry', 'N/A')}")
            st.write(f"**Market Cap:** ${overview.get('MarketCapitalization', 'N/A')}")
        with col2:
            st.write(f"**P/E Ratio:** {overview.get('PERatio', 'N/A')}")
            st.write(f"**Dividend Yield:** {overview.get('DividendYield', 'N/A')}")
            st.write(f"**52 Week High/Low:** ${overview.get('52WeekHigh', 'N/A')} / ${overview.get('52WeekLow', 'N/A')}")
        with col3:
            st.write(f"**EPS:** ${overview.get('EPS', 'N/A')}")
            st.write(f"**Beta:** {overview.get('Beta', 'N/A')}")
            st.write(f"**Profit Margin:** {overview.get('ProfitMargin', 'N/A')}")
        
        with st.expander("Company Description"):
            st.write(overview.get('Description', 'N/A'))

# Function to display stock price chart
def display_stock_chart(symbol):
    st.subheader(f"Stock Price Chart: {symbol}")
    # Allow users to select the time period for the chart
    period = st.selectbox("Select Time Period", ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'])
    data = get_historical_data(symbol, period)
    
    if not data.empty:
        # Create a candlestick chart with a 20-day moving average line
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index,
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'],
                                     name='Price'))
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(window=20).mean(), name='20 Day MA'))
        fig.update_layout(title=f"{symbol} Stock Price", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)
    else:
        st.write("No historical data available.")

# Function to display financial charts
def display_financial_charts(symbol):
    st.subheader("Financial Charts")
    
    # Fetch financial data using yfinance
    stock = yf.Ticker(symbol)
    income_stmt = stock.financials
    balance_sheet = stock.balance_sheet
    cash_flow = stock.cashflow
    
    # Create tabs for Income Statement, Balance Sheet, and Cash Flow
    tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
    
    with tab1:
        if not income_stmt.empty:
            # Create a bar chart for Total Revenue and Net Income
            fig = go.Figure()
            fig.add_trace(go.Bar(x=income_stmt.columns, y=income_stmt.loc['Total Revenue'], name="Total Revenue"))
            fig.add_trace(go.Bar(x=income_stmt.columns, y=income_stmt.loc['Net Income'], name="Net Income"))
            fig.update_layout(title="Annual Revenue and Net Income", barmode='group')
            st.plotly_chart(fig)
        else:
            st.write("No income statement data available.")
    
    with tab2:
        if not balance_sheet.empty:
            # Create a bar chart for Total Assets and Total Liabilities
            fig = go.Figure()
            fig.add_trace(go.Bar(x=balance_sheet.columns, y=balance_sheet.loc['Total Assets'], name="Total Assets"))
            fig.add_trace(go.Bar(x=balance_sheet.columns, y=balance_sheet.loc['Total Liabilities Net Minority Interest'], name="Total Liabilities"))
            fig.update_layout(title="Annual Assets and Liabilities", barmode='group')
            st.plotly_chart(fig)
        else:
            st.write("No balance sheet data available.")
    
    with tab3:
        if not cash_flow.empty:
            # Create a bar chart for Operating Cash Flow, Investing Cash Flow, and Financing Cash Flow
            fig = go.Figure()
            fig.add_trace(go.Bar(x=cash_flow.columns, y=cash_flow.loc['Operating Cash Flow'], name="Operating Cash Flow"))
            fig.add_trace(go.Bar(x=cash_flow.columns, y=cash_flow.loc['Investing Cash Flow'], name="Investing Cash Flow"))
            fig.add_trace(go.Bar(x=cash_flow.columns, y=cash_flow.loc['Financing Cash Flow'], name="Financing Cash Flow"))
            fig.update_layout(title="Annual Cash Flows", barmode='group')
            st.plotly_chart(fig)
        else:
            st.write("No cash flow data available.")

# Function to manage portfolio
def manage_portfolio():
    st.markdown("<h1 style='text-align: center;'>Portfolio Management</h1>", unsafe_allow_html=True)
    
    # Initialize portfolio in session state if it doesn't exist
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = pd.DataFrame(columns=['Symbol', 'Shares', 'Purchase Price', 'Purchase Date'])
    
    # Add stock to portfolio
    with st.expander("Add New Stock to Portfolio", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            symbol = st.text_input("Stock Symbol:")
        with col2:
            shares = st.number_input("Number of Shares:", min_value=0.0, step=0.01)
        with col3:
            price = st.number_input("Purchase Price:", min_value=0.0, step=0.01)
        with col4:
            purchase_date = st.date_input("Purchase Date:")
        
        if st.button("Add to Portfolio"):
            new_stock = pd.DataFrame({'Symbol': [symbol], 'Shares': [shares], 'Purchase Price': [price], 'Purchase Date': [purchase_date]})
            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_stock], ignore_index=True)
            st.success(f"Added {shares} shares of {symbol} to your portfolio.")
    
    # Display and edit portfolio
    st.subheader("Current Portfolio")
    if not st.session_state.portfolio.empty:
        edited_df = st.data_editor(st.session_state.portfolio, num_rows="dynamic")
        if st.button("Save Changes"):
            st.session_state.portfolio = edited_df
            st.success("Portfolio updated successfully!")
    else:
        st.info("Your portfolio is empty. Add some stocks to get started!")
    
    # Calculate and display portfolio performance
    if not st.session_state.portfolio.empty:
        st.subheader("Portfolio Performance")
        total_value = 0
        total_cost = 0
        performance_data = []
        
        for _, row in st.session_state.portfolio.iterrows():
            current_price = get_real_time_price(row['Symbol'])
            if current_price:
                value = current_price * row['Shares']
                cost = row['Purchase Price'] * row['Shares']
                total_value += value
                total_cost += cost
                performance_data.append({
                    'Symbol': row['Symbol'],
                    'Current Price': current_price,
                    'Value': value,
                    'Cost': cost,
                    'Gain/Loss': value - cost,
                    'Gain/Loss %': (value - cost) / cost * 100
                })
        
        performance_df = pd.DataFrame(performance_data)
        
        # Display performance metrics
        col1, col2, col3 = st.columns(3)
        total_gain_loss = total_value - total_cost
        total_gain_loss_percent = (total_gain_loss / total_cost) * 100
        col1.metric("Total Portfolio Value", f"${total_value:.2f}", f"${total_gain_loss:.2f}")
        col2.metric("Total Cost", f"${total_cost:.2f}")
        col3.metric("Total Return", f"{total_gain_loss_percent:.2f}%")
        
        # Display performance table
        st.dataframe(performance_df.style.format({
            'Current Price': '${:.2f}',
            'Value': '${:.2f}',
            'Cost': '${:.2f}',
            'Gain/Loss': '${:.2f}',
            'Gain/Loss %': '{:.2f}%'
        }))
        
        # Portfolio Composition Pie Chart
        st.subheader("Portfolio Composition")
        fig = px.pie(performance_df, values='Value', names='Symbol', title='Portfolio Allocation')
        st.plotly_chart(fig)
    
    # Export Portfolio
    st.subheader("Export Portfolio")
    if st.button("Export to CSV"):
        csv = st.session_state.portfolio.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="my_portfolio.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)

def display_company_analysis(symbol):
    display_company_overview(symbol)
    display_stock_chart(symbol)
    display_financial_charts(symbol)
    display_stock_prediction(symbol)

def calculate_ema(data, period=20):
    return data.ewm(span=period, adjust=False).mean()

def predict_with_ema(data, days_to_predict=30):
    ema = calculate_ema(data['Close'])
    last_ema = ema.iloc[-1]
    
    # Calculate the average daily change in EMA
    ema_daily_change = ema.diff().mean()
    
    future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=days_to_predict)
    future_ema = [last_ema + i * ema_daily_change for i in range(1, days_to_predict + 1)]
    
    return pd.Series(future_ema, index=future_dates)

def create_features(data):
    data['day'] = data.index.dayofweek
    data['month'] = data.index.month
    data['year'] = data.index.year
    data['day_of_year'] = data.index.dayofyear
    return data

def train_xgb_model(data):
    features = ['day', 'month', 'year', 'day_of_year']
    X = data[features]
    y = data['Close']
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return model

def predict_with_xgb(model, data, days_to_predict=30):
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_to_predict)
    future_features = pd.DataFrame({
        'day': future_dates.dayofweek,
        'month': future_dates.month,
        'year': future_dates.year,
        'day_of_year': future_dates.dayofyear
    })
    predictions = model.predict(future_features)
    return pd.Series(predictions, index=future_dates)

def handle_outliers(predictions, last_known_price, threshold=100):
    """
    Handle outliers in predictions.
    Replace values below threshold with the last known price.
    """
    return predictions.apply(lambda x: max(x, last_known_price) if x < threshold else x)

def main():
    st.title('Stock Price Predictor')

    ticker = st.text_input('Enter Stock Ticker (e.g., AAPL, GOOGL)')
    start_date = st.date_input('Start Date', value=pd.to_datetime('2020-01-01'))
    end_date = st.date_input('End Date', value=pd.to_datetime('today'))

    prediction_days = st.slider('Number of days to predict', min_value=1, max_value=90, value=30)
    ema_period = st.slider('EMA Period', min_value=5, max_value=100, value=20)

    if st.button('Predict'):
        if ticker and start_date and end_date:
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if not data.empty:
                st.subheader('Historical Data')
                st.dataframe(data)

                last_known_price = data['Close'].iloc[-1]

                # EMA Prediction
                ema = calculate_ema(data['Close'], period=ema_period)
                future_ema = predict_with_ema(data, days_to_predict=prediction_days)
                future_ema = handle_outliers(future_ema, last_known_price)

                # XGBoost Prediction
                data_with_features = create_features(data)
                xgb_model = train_xgb_model(data_with_features)
                future_xgb = predict_with_xgb(xgb_model, data_with_features, days_to_predict=prediction_days)
                future_xgb = handle_outliers(future_xgb, last_known_price)

                # Average Prediction
                future_avg = (future_ema + future_xgb) / 2

                # Plotting
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical'))
                fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA'))
                fig.add_trace(go.Scatter(x=future_ema.index, y=future_ema, mode='lines', name='EMA Prediction', line=dict(dash='dash')))
                fig.add_trace(go.Scatter(x=future_xgb.index, y=future_xgb, mode='lines', name='XGB Prediction', line=dict(dash='dot')))
                fig.add_trace(go.Scatter(x=future_avg.index, y=future_avg, mode='lines', name='Average Prediction', line=dict(color='red')))
                fig.update_layout(title=f'{ticker} Stock Price and Predictions', xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig)

                # Display prediction results
                st.subheader('Predictions')
                predictions_df = pd.DataFrame({
                    'Date': future_avg.index,
                    'EMA Prediction': future_ema.values,
                    'XGB Prediction': future_xgb.values,
                    'Average Prediction': future_avg.values
                })
                st.dataframe(predictions_df)

                # Display outlier information
                st.subheader('Outlier Information')
                outlier_count = (predictions_df == 'Outlier').sum()
                st.write(f"Number of outliers (predictions below $100):")
                st.write(f"EMA Prediction: {outlier_count['EMA Prediction']}")
                st.write(f"XGB Prediction: {outlier_count['XGB Prediction']}")
                st.write(f"Average Prediction: {outlier_count['Average Prediction']}")

            else:
                st.error('No data available for the selected date range.')
        else:
            st.error('Please enter all required fields.')

# Main application function
def main():
    # Set up the Streamlit page configuration
    st.set_page_config(page_title="Advanced Stock Data Visualizer", layout="wide")
    
    # Apply custom CSS for improved UI
    st.markdown("""
    <style>
    :root {
        --background-color: #1e2a3a;
        --text-color: #ffffff;
        --card-background: #2c3e50;
        --sidebar-background: #0e1621;
    }
    
    @media (prefers-color-scheme: light) {
        :root {
            --background-color: #f0f2f5;
            --text-color: #333333;
            --card-background: #ffffff;
            --sidebar-background: #e0e0e0;
        }
    }
    
    body {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    .reportview-container {
        background: var(--background-color);
        color: var(--text-color);
    }
    
    .sidebar .sidebar-content {
        background: var(--sidebar-background);
    }
    
    .Widget>label {
        color: var(--text-color);
        font-weight: bold;
    }
    
    .stRadio>div {
        flex-direction: row;
        align-items: center;
    }
    
    .stRadio>div>label {
        margin-right: 15px;
    }
    
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    
    .medium-font {
        font-size: 18px !important;
    }
    
    .card {
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        background-color: var(--card-background);
        color: var(--text-color);
    }
    
    .feature-card {
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        background-color: var(--card-background);
        color: var(--text-color);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .feature-icon {
        font-size: 2.5em;
        margin-bottom: 10px;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-color);
    }
    
    .stMetric {
        background-color: var(--card-background);
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    
    <script>
    function updateTheme() {
        const isDarkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        document.body.classList.toggle('dark-mode', isDarkMode);
    }
    
    updateTheme();
    window.matchMedia('(prefers-color-scheme: dark)').addListener(updateTheme);
    </script>
    """, unsafe_allow_html=True)
    
    # Set the main title of the application
    st.title("Advanced Stock Data Visualizer")

    # Create a sidebar for navigation using option_menu
    with st.sidebar:
        selected = option_menu(
            "Navigation",
            ["Home", "Company Analysis", "Portfolio Management", "Market Overview"],
            icons=['house', 'graph-up', 'wallet2', 'globe'],
            menu_icon="cast",
            default_index=0,
        )

    # Home page
    if selected == "Home":
        st.title("Welcome to the Advanced Stock Data Visualizer")
        st.markdown("<p class='medium-font'>Your one-stop solution for comprehensive stock market analysis and portfolio management.</p>", unsafe_allow_html=True)
        
        # Introduction
        st.markdown("""
        <div class='card'>
        <h2>Discover the Power of Data-Driven Investing</h2>
        <p>Our Advanced Stock Data Visualizer provides you with cutting-edge tools and insights to make informed investment decisions. Whether you're a seasoned investor or just starting out, our platform offers a range of powerful features to enhance your investment strategy.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features overview
        st.subheader("Explore Our Features")
        col1, col2 = st.columns(2)
        
        features = [
            {
                "icon": "🔍",
                "title": "Company Analysis",
                "description": "Dive deep into company financials, stock performance, and key metrics to make data-driven investment decisions."
            },
            {
                "icon": "💼",
                "title": "Portfolio Management",
                "description": "Track your investments, analyze performance, and optimize your portfolio with our advanced management tools."
            },
            {
                "icon": "📊",
                "title": "Interactive Charts",
                "description": "Visualize stock data with customizable, interactive charts to identify trends and patterns easily."
            },
            {
                "icon": "🌐",
                "title": "Market Overview",
                "description": "Stay informed with real-time market data, sector performance, and interactive heatmaps."
            }
        ]
        
        for i, feature in enumerate(features):
            with col1 if i % 2 == 0 else col2:
                st.markdown(f"""
                <div class='feature-card'>
                    <div class='feature-icon'>{feature['icon']}</div>
                    <h3>{feature['title']}</h3>
                    <p>{feature['description']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Market Overview
        st.subheader("Today's Market Snapshot")
        market_indices = ['^GSPC', '^DJI', '^IXIC', '^RUT']  # S&P 500, Dow Jones, NASDAQ, Russell 2000
        index_names = ["S&P 500", "Dow Jones", "NASDAQ", "Russell 2000"]
        cols = st.columns(4)
        for index, name, col in zip(market_indices, index_names, cols):
            price = get_real_time_price(index)
            if price:
                col.metric(name, f"${price:.2f}")
        
        # Call-to-action
        st.markdown("""
        <div class='card' style='background-color: #3498db;'>
        <h2>Ready to Start?</h2>
        <p>Use the sidebar to navigate through different sections of the app and unlock the full potential of your investments!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Disclaimer
        st.markdown("""
        <div style='font-size: 12px; color: #bdc3c7;'>
        <p><strong>Disclaimer:</strong> This application is for informational purposes only. It is not intended to be investment advice. 
        Always do your own research and consult with a qualified financial advisor before making investment decisions.</p>
        </div>
        """, unsafe_allow_html=True)

    elif selected == "Company Analysis":
        st.markdown("<h1 style='text-align: center;'>Company Analysis</h1>", unsafe_allow_html=True)
        
        # Search bar and button in the same row
        col1, col2 = st.columns([3, 1])
        with col1:
            symbol = st.text_input("Enter company symbol for analysis (e.g., AAPL, GOOGL):", key="symbol_input")
        with col2:
            search_button = st.button("Search")

        if symbol and (search_button or symbol != st.session_state.get('last_searched')):
            st.session_state['last_searched'] = symbol
            with st.spinner(f"Analyzing {symbol}..."):
                display_company_analysis(symbol)
        
        # Add more content
        st.markdown("---")
        st.subheader("Popular Stocks")
        popular_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "FB"]
        cols = st.columns(len(popular_stocks))
        for i, stock in enumerate(popular_stocks):
            with cols[i]:
                if st.button(stock):
                    symbol = stock
                    with st.spinner(f"Fetching data for {symbol}..."):
                        # Fetch data using yfinance
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        
                        # Display some basic information
                        st.write(f"**{info['longName']}** ({symbol})")
                        st.write(f"Current Price: ${info['currentPrice']:.2f}")
                        st.write(f"52 Week High: ${info['fiftyTwoWeekHigh']:.2f}")
                        st.write(f"52 Week Low: ${info['fiftyTwoWeekLow']:.2f}")
                        
                        # Fetch historical data
                        hist = ticker.history(period="1mo")
                        
                        # Create a line chart of closing prices
                        fig = px.line(hist, x=hist.index, y="Close", title=f"{symbol} Stock Price (Last Month)")
                        st.plotly_chart(fig)

    # Portfolio Management page
    elif selected == "Portfolio Management":
        # Provide tools for users to manage their stock portfolio
        manage_portfolio()

    # Market Overview page
    elif selected == "Market Overview":
        st.markdown("<h1 style='text-align: center;'>Market Overview</h1>", unsafe_allow_html=True)

        # Market Summary
        st.subheader("Market Summary")
        col1, col2, col3, col4 = st.columns(4)
        indices = ['^GSPC', '^DJI', '^IXIC', '^RUT']
        index_names = ['S&P 500', 'Dow Jones', 'NASDAQ', 'Russell 2000']
        
        for index, name, col in zip(indices, index_names, [col1, col2, col3, col4]):
            ticker = yf.Ticker(index)
            data = ticker.history(period="1d")
            if not data.empty:
                current_price = data['Close'].iloc[-1]
                previous_close = data['Close'].iloc[-2] if len(data) > 1 else data['Open'].iloc[-1]
                change = (current_price - previous_close) / previous_close * 100
                col.metric(name, f"${current_price:.2f}", f"{change:.2f}%")

        # Major Indices Performance Chart
        st.subheader("Major Indices Performance (Last 6 Months)")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        df_indices = yf.download(indices, start=start_date, end=end_date)['Adj Close']
        df_indices_norm = df_indices / df_indices.iloc[0] * 100

        fig_indices = px.line(df_indices_norm, x=df_indices_norm.index, y=df_indices_norm.columns,
                              labels={'value': 'Normalized Price', 'variable': 'Index'},
                              title='Major Indices Performance (Normalized to 100)')
        st.plotly_chart(fig_indices, use_container_width=True)

        # Sector Performance
        st.subheader("Sector Performance (Last Month)")
        sectors = ['XLK', 'XLV', 'XLF', 'XLY', 'XLI', 'XLE', 'XLB', 'XLU', 'XLRE', 'XLP', 'XLC']
        sector_names = ['Technology', 'Healthcare', 'Financials', 'Consumer Discretionary', 'Industrials',
                        'Energy', 'Materials', 'Utilities', 'Real Estate', 'Consumer Staples', 'Communication Services']
        
        df_sectors = yf.download(sectors, period="1mo")['Adj Close']
        sector_returns = df_sectors.pct_change().iloc[-1].sort_values(ascending=False)
        sector_returns.index = sector_names

        fig_sectors = go.Figure(go.Bar(
            x=sector_returns.values * 100,
            y=sector_returns.index,
            orientation='h',
            marker=dict(color=['green' if x >= 0 else 'red' for x in sector_returns.values])
        ))
        fig_sectors.update_layout(title='Sector Performance (Last Month)', xaxis_title='Percentage Change', yaxis_title='Sector')
        st.plotly_chart(fig_sectors, use_container_width=True)

        # Top Movers
        st.subheader("Top Movers (S&P 500)")

        # Function to get S&P 500 components
        def get_sp500_components():
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            table = pd.read_html(url, header=0)[0]
            return table['Symbol'].tolist()

        # Get S&P 500 components
        sp500_components = get_sp500_components()

        # Fetch data for all S&P 500 stocks
        try:
            df_sp500 = yf.download(sp500_components, period="5d")['Adj Close']
            df_sp500_returns = df_sp500.pct_change().iloc[-1].sort_values(ascending=False)

            # Filter out any NaN values
            df_sp500_returns = df_sp500_returns.dropna()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div style='background-color: #2c3e50; padding: 20px; border-radius: 10px;'>
                <h4 style='text-align: center;'>Top Gainers</h4>
                <ol>
                """, unsafe_allow_html=True)
                for symbol, change in df_sp500_returns.head(5).items():
                    current_price = df_sp500.loc[df_sp500.index[-1], symbol]
                    st.markdown(f"<li>{symbol}: ${current_price:.2f} (+{change*100:.2f}%)</li>", unsafe_allow_html=True)
                st.markdown("</ol></div>", unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div style='background-color: #2c3e50; padding: 20px; border-radius: 10px;'>
                <h4 style='text-align: center;'>Top Losers</h4>
                <ol>
                """, unsafe_allow_html=True)
                for symbol, change in df_sp500_returns.tail(5).items():
                    current_price = df_sp500.loc[df_sp500.index[-1], symbol]
                    st.markdown(f"<li>{symbol}: ${current_price:.2f} ({change*100:.2f}%)</li>", unsafe_allow_html=True)
                st.markdown("</ol></div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred while fetching S&P 500 data: {str(e)}")
            st.write("Unable to display Top Movers at this time.")

        # Market News (You might want to integrate a news API here)
        st.subheader("Latest Market News")
        st.markdown("""
        <div style='background-color: #2c3e50; padding: 20px; border-radius: 10px;'>
        <ul>
            <li><strong>Market Update:</strong> Major indices show mixed performance amid economic data releases.</li>
            <li><strong>Earnings Season:</strong> Key companies report quarterly results, impacting market sentiment.</li>
            <li><strong>Economic Indicators:</strong> Latest economic reports provide insights into market direction.</li>
            <li><strong>Global Events:</strong> International developments influence domestic market trends.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # Display footer information
    st.sidebar.write("Powered by Alpha Vantage and Yahoo Finance APIs")

# Run the main application
if __name__ == "__main__":
    main()
