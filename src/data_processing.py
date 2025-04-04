# src/data_processing.py
import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        
    def load_data(self):
        """Load data from CSV file"""
        self.data = pd.read_csv(self.data_path)
        return self.data
    
    def clean_data(self):
        """Clean and preprocess the data"""
        if self.data is None:
            self.load_data()
            
        # Handle missing values
        self.data = self.data.fillna({
            'children': 0,
            'country': 'unknown',
            'agent': 0,
            'company': 0
        })
        
        # Convert date columns to datetime
        self.data['reservation_status_date'] = pd.to_datetime(self.data['reservation_status_date'])
        self.data['arrival_date'] = pd.to_datetime(self.data['arrival_date_year'].astype(str) + '-' + 
                                                 self.data['arrival_date_month'] + '-' + 
                                                 self.data['arrival_date_day_of_month'].astype(str))
        
        # Calculate revenue
        self.data['total_nights'] = self.data['stays_in_weekend_nights'] + self.data['stays_in_week_nights']
        self.data['revenue'] = self.data['adr'] * self.data['total_nights']
        
        # Calculate lead time in days
        self.data['lead_time_days'] = self.data['lead_time']
        
        # Clean up country codes if needed
        # You might want to map country codes to country names
        
        return self.data
    
    def get_processed_data(self):
        """Get the processed data"""
        if self.data is None:
            self.load_data()
            self.clean_data()
        return self.data