# src/analytics.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

class BookingAnalytics:
    def __init__(self, data):
        self.data = data
        
    def revenue_trends(self):
        """Calculate revenue trends over time"""
        # Group by month and sum revenue
        revenue_by_month = self.data.groupby(pd.Grouper(key='arrival_date', freq='M'))['revenue'].sum().reset_index()
        revenue_by_month['month_year'] = revenue_by_month['arrival_date'].dt.strftime('%b %Y')
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(revenue_by_month['arrival_date'], revenue_by_month['revenue'], marker='o')
        plt.title('Revenue Trends Over Time')
        plt.xlabel('Date')
        plt.ylabel('Revenue')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        return {
            'chart': base64.b64encode(image_png).decode('utf-8'),
            'data': revenue_by_month[['month_year', 'revenue']].to_dict('records')
        }
    
    def cancellation_rate(self):
        """Calculate cancellation rate"""
        cancellation_counts = self.data['is_canceled'].value_counts()
        total_bookings = len(self.data)
        cancellation_rate = (cancellation_counts.get(1, 0) / total_bookings) * 100
        
        # Create pie chart
        plt.figure(figsize=(8, 8))
        plt.pie([cancellation_counts.get(0, 0), cancellation_counts.get(1, 0)], 
                labels=['Confirmed', 'Canceled'], 
                autopct='%1.1f%%',
                colors=['#4CAF50', '#F44336'])
        plt.title('Booking Cancellation Rate')
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        return {
            'chart': base64.b64encode(image_png).decode('utf-8'),
            'rate': cancellation_rate,
            'total_bookings': total_bookings,
            'canceled_bookings': cancellation_counts.get(1, 0)
        }
    
    def geographical_distribution(self):
        """Calculate geographical distribution of bookings"""
        country_counts = self.data['country'].value_counts().reset_index()
        country_counts.columns = ['country', 'count']
        
        # Create bar chart for top 15 countries
        top_countries = country_counts.head(15)
        plt.figure(figsize=(12, 8))
        sns.barplot(x='count', y='country', data=top_countries)
        plt.title('Top 15 Countries by Number of Bookings')
        plt.xlabel('Number of Bookings')
        plt.ylabel('Country')
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        return {
            'chart': base64.b64encode(image_png).decode('utf-8'),
            'data': country_counts.to_dict('records')
        }
    
    def lead_time_distribution(self):
        """Calculate lead time distribution"""
        # Create histogram of lead times
        plt.figure(figsize=(12, 6))
        sns.histplot(self.data['lead_time_days'], bins=30, kde=True)
        plt.title('Distribution of Booking Lead Time')
        plt.xlabel('Lead Time (Days)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        
        # Calculate statistics
        lead_time_stats = {
            'mean': self.data['lead_time_days'].mean(),
            'median': self.data['lead_time_days'].median(),
            'min': self.data['lead_time_days'].min(),
            'max': self.data['lead_time_days'].max()
        }
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        return {
            'chart': base64.b64encode(image_png).decode('utf-8'),
            'stats': lead_time_stats
        }
    
    def additional_analytics(self):
        """Additional analytics - guest composition"""
        # Guest composition analysis
        guest_composition = self.data.groupby(['adults', 'children', 'babies']).size().reset_index()
        guest_composition.columns = ['adults', 'children', 'babies', 'count']
        guest_composition = guest_composition.sort_values('count', ascending=False).head(10)
        
        # Create bar chart
        plt.figure(figsize=(12, 6))
        sns.barplot(x='count', y=guest_composition.apply(lambda x: f"A:{x['adults']}, C:{x['children']}, B:{x['babies']}", axis=1), data=guest_composition)
        plt.title('Top 10 Guest Compositions')
        plt.xlabel('Number of Bookings')
        plt.ylabel('Guest Composition (Adults, Children, Babies)')
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        return {
            'chart': base64.b64encode(image_png).decode('utf-8'),
            'data': guest_composition.to_dict('records')
        }
    
    def generate_all_analytics(self):
        """Generate all analytics"""
        return {
            'revenue_trends': self.revenue_trends(),
            'cancellation_rate': self.cancellation_rate(),
            'geographical_distribution': self.geographical_distribution(),
            'lead_time_distribution': self.lead_time_distribution(),
            'additional_analytics': self.additional_analytics()
        }