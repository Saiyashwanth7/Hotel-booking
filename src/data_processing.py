# src/data_processing.py
import pandas as pd

MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}


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
        df = self.data

        # Missing values (the CSV uses the text NULL, which pandas reads as NaN)
        df["children"] = df["children"].fillna(0)
        df["country"] = df["country"].fillna("unknown")
        df["agent"] = df["agent"].fillna(0)
        df["company"] = df["company"].fillna(0)

        # Dates
        df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"])
        df["arrival_date"] = pd.to_datetime(
            dict(
                year=df["arrival_date_year"],
                month=df["arrival_date_month"].map(MONTHS),
                day=df["arrival_date_day_of_month"],
            )
        )

        # Derived columns
        df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
        df["revenue"] = df["adr"] * df["total_nights"]  # ADR x nights, includes canceled bookings
        df["lead_time_days"] = df["lead_time"]

        self.data = df
        return self.data

    def get_processed_data(self):
        """Get the processed data (loads and cleans on first call)"""
        if self.data is None:
            self.load_data()
            self.clean_data()
        return self.data
