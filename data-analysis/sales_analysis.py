"""
Sales Data Analysis Dashboard

This module provides comprehensive analysis of retail sales data including
trend analysis, product performance, and regional insights.

Author: Gert Coetser
Date: March 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
from typing import Tuple, Dict


class SalesAnalyzer:
    """Analyzes retail sales data and generates insights."""
    
    def __init__(self, data_path: str = None):
        """
        Initialize the SalesAnalyzer.
        
        Args:
            data_path: Path to CSV file. If None, generates sample data.
        """
        self.df = None
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        if data_path:
            self.load_data(data_path)
        else:
            self.generate_sample_data()
    
    def generate_sample_data(self, n_records: int = 1000) -> None:
        """
        Generate sample sales data for demonstration.
        
        Args:
            n_records: Number of records to generate
        """
        np.random.seed(42)
        
        # Generate date range
        start_date = datetime(2024, 1, 1)
        dates = [start_date + timedelta(days=x) for x in range(365)]
        
        # Product and region lists
        products = [
            'Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones',
            'Webcam', 'USB Cable', 'Desk Lamp', 'Chair', 'Notebook'
        ]
        regions = ['North', 'South', 'East', 'West', 'Central']
        
        # Generate records
        data = {
            'date': np.random.choice(dates, n_records),
            'product': np.random.choice(products, n_records),
            'region': np.random.choice(regions, n_records),
            'quantity': np.random.randint(1, 10, n_records),
            'unit_price': np.random.uniform(10, 500, n_records).round(2)
        }
        
        self.df = pd.DataFrame(data)
        self.df['total_sales'] = self.df['quantity'] * self.df['unit_price']
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        print(f"✓ Generated {n_records} sample sales records")
    
    def load_data(self, path: str) -> None:
        """
        Load sales data from CSV file.
        
        Args:
            path: Path to CSV file
        """
        self.df = pd.read_csv(path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        print(f"✓ Loaded data from {path}")
    
    def clean_data(self) -> None:
        """Clean and preprocess the data."""
        initial_rows = len(self.df)
        
        # Remove duplicates
        self.df.drop_duplicates(inplace=True)
        
        # Remove rows with missing values
        self.df.dropna(inplace=True)
        
        # Remove invalid sales (negative or zero)
        self.df = self.df[self.df['total_sales'] > 0]
        
        rows_removed = initial_rows - len(self.df)
        print(f"✓ Data cleaned: {rows_removed} rows removed")
    
    def analyze_trends(self) -> pd.DataFrame:
        """
        Analyze sales trends over time.
        
        Returns:
            DataFrame with monthly sales aggregation
        """
        monthly_sales = self.df.groupby(
            self.df['date'].dt.to_period('M')
        ).agg({
            'total_sales': 'sum',
            'quantity': 'sum'
        }).reset_index()
        
        monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()
        
        return monthly_sales
    
    def top_products(self, n: int = 10) -> pd.DataFrame:
        """
        Identify top performing products.
        
        Args:
            n: Number of top products to return
            
        Returns:
            DataFrame with top products
        """
        product_sales = self.df.groupby('product').agg({
            'total_sales': 'sum',
            'quantity': 'sum'
        }).sort_values('total_sales', ascending=False).head(n)
        
        return product_sales
    
    def regional_analysis(self) -> pd.DataFrame:
        """
        Analyze sales by region.
        
        Returns:
            DataFrame with regional sales summary
        """
        regional_sales = self.df.groupby('region').agg({
            'total_sales': ['sum', 'mean', 'count']
        }).round(2)
        
        regional_sales.columns = ['total_sales', 'avg_sale', 'num_transactions']
        
        return regional_sales.sort_values('total_sales', ascending=False)
    
    def generate_summary_statistics(self) -> Dict:
        """
        Generate comprehensive summary statistics.
        
        Returns:
            Dictionary containing summary statistics
        """
        stats = {
            'total_revenue': self.df['total_sales'].sum(),
            'total_transactions': len(self.df),
            'avg_transaction_value': self.df['total_sales'].mean(),
            'median_transaction_value': self.df['total_sales'].median(),
            'total_units_sold': self.df['quantity'].sum(),
            'unique_products': self.df['product'].nunique(),
            'date_range': f"{self.df['date'].min().date()} to {self.df['date'].max().date()}"
        }
        
        return stats
    
    def plot_trends(self, monthly_sales: pd.DataFrame) -> None:
        """
        Create and save sales trend visualization.
        
        Args:
            monthly_sales: DataFrame with monthly sales data
        """
        plt.figure(figsize=(12, 6))
        plt.plot(monthly_sales['date'], monthly_sales['total_sales'], 
                marker='o', linewidth=2, markersize=6)
        plt.title('Monthly Sales Trend', fontsize=16, fontweight='bold')
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Total Sales ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_path = self.output_dir / 'sales_trend.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved sales trend chart to {output_path}")
    
    def plot_top_products(self, product_sales: pd.DataFrame) -> None:
        """
        Create and save top products visualization.
        
        Args:
            product_sales: DataFrame with product sales data
        """
        plt.figure(figsize=(10, 6))
        colors = sns.color_palette('viridis', len(product_sales))
        plt.barh(product_sales.index, product_sales['total_sales'], color=colors)
        plt.title('Top 10 Products by Revenue', fontsize=16, fontweight='bold')
        plt.xlabel('Total Sales ($)', fontsize=12)
        plt.ylabel('Product', fontsize=12)
        plt.tight_layout()
        
        output_path = self.output_dir / 'top_products.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved top products chart to {output_path}")
    
    def plot_regional_analysis(self, regional_sales: pd.DataFrame) -> None:
        """
        Create and save regional analysis visualization.
        
        Args:
            regional_sales: DataFrame with regional sales data
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Total sales by region
        colors = sns.color_palette('Set2', len(regional_sales))
        ax1.bar(regional_sales.index, regional_sales['total_sales'], color=colors)
        ax1.set_title('Total Sales by Region', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Total Sales ($)', fontsize=11)
        ax1.tick_params(axis='x', rotation=45)
        
        # Average sale by region
        ax2.bar(regional_sales.index, regional_sales['avg_sale'], color=colors)
        ax2.set_title('Average Transaction Value by Region', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Average Sale ($)', fontsize=11)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'regional_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved regional analysis chart to {output_path}")
    
    def export_summary(self, stats: Dict, regional_sales: pd.DataFrame) -> None:
        """
        Export summary statistics to CSV.
        
        Args:
            stats: Dictionary of summary statistics
            regional_sales: DataFrame with regional sales data
        """
        # Create summary DataFrame
        summary_df = pd.DataFrame([stats])
        
        output_path = self.output_dir / 'summary_report.csv'
        summary_df.to_csv(output_path, index=False)
        print(f"✓ Saved summary report to {output_path}")
        
        # Also save regional breakdown
        regional_path = self.output_dir / 'regional_breakdown.csv'
        regional_sales.to_csv(regional_path)
        print(f"✓ Saved regional breakdown to {regional_path}")
    
    def run_full_analysis(self) -> None:
        """Execute complete analysis pipeline."""
        print("\n" + "="*50)
        print("SALES DATA ANALYSIS")
        print("="*50 + "\n")
        
        # Clean data
        self.clean_data()
        
        # Perform analyses
        print("\nPerforming analysis...")
        monthly_sales = self.analyze_trends()
        top_products = self.top_products(10)
        regional_sales = self.regional_analysis()
        stats = self.generate_summary_statistics()
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        self.plot_trends(monthly_sales)
        self.plot_top_products(top_products)
        self.plot_regional_analysis(regional_sales)
        
        # Export results
        print("\nExporting results...")
        self.export_summary(stats, regional_sales)
        
        # Print summary to console
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(f"Total Revenue:        ${stats['total_revenue']:,.2f}")
        print(f"Total Transactions:   {stats['total_transactions']:,}")
        print(f"Avg Transaction:      ${stats['avg_transaction_value']:,.2f}")
        print(f"Median Transaction:   ${stats['median_transaction_value']:,.2f}")
        print(f"Total Units Sold:     {stats['total_units_sold']:,}")
        print(f"Unique Products:      {stats['unique_products']}")
        print(f"Date Range:           {stats['date_range']}")
        print("="*50 + "\n")
        
        print("✓ Analysis complete! Check the 'output' directory for charts and reports.\n")


def main():
    """Main execution function."""
    # Create analyzer with sample data
    analyzer = SalesAnalyzer()
    
    # Run full analysis
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
