python3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

class DegradationVisualizer:
    def __init__(self, analyzer, output_dir='results/figures'):
        self.analyzer = analyzer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_degradation_curves(self, save=True):
        """Plot degradation curves for all temperatures"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        temperatures = self.analyzer.df['temperature_c'].unique()
        colors = sns.color_palette("viridis", len(temperatures))
        
        for i, temp in enumerate(sorted(temperatures)):
            temp_data = self.analyzer.df[self.analyzer.df['temperature_c'] == temp]
            
            # Linear scale
            ax1.plot(temp_data['time_weeks'], temp_data['retention_percent'], 
                    'o-', color=colors[i], label=f'{temp}°C', markersize=6)
            
            # Log scale
            ax2.semilogy(temp_data['time_weeks'], temp_data['retention_percent'], 
                        'o-', color=colors[i], label=f'{temp}°C', markersize=6)
        
        # Formatting
        for ax in [ax1, ax2]:
            ax.set_xlabel('Time (weeks)', fontsize=12)
            ax.set_ylabel('Molecular Weight Retention (%)', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        ax1.set_title('Degradation Curves - Linear Scale')
        ax2.set_title('Degradation Curves - Log Scale')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'degradation_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_arrhenius(self, save=True):
        """Create Arrhenius plot"""
        temperatures_k = np.array([temp + 273.15 for temp in self.analyzer.rate_constants.keys()])
        ln_k = np.array([np.log(data['k']) for data in self.analyzer.rate_constants.values()])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Data points
        ax.plot(1/temperatures_k, ln_k, 'bo', markersize=8, label='Data')
        
        # Fitted line
        x_fit = np.linspace(1/temperatures_k.min(), 1/temperatures_k.max(), 100)
        slope = -self.analyzer.activation_energy * 1000 / 8.314
        intercept = np.log(self.analyzer.pre_exponential)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, 'r-', linewidth=2, label='Arrhenius Fit')
        
        ax.set_xlabel('1/T (K⁻¹)', fontsize=12)
        ax.set_ylabel('ln(k)', fontsize=12)
        ax.set_title(f'Arrhenius Plot\nEₐ = {self.analyzer.activation_energy:.1f} kJ/mol')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(self.output_dir / 'arrhenius_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction(self, temp_celsius=25, max_years=15, save=True):
        """Plot long-term prediction"""
        time_years = np.linspace(0, max_years, 1000)
        retention = []
        
        for t in time_years:
            if t == 0:
                retention.append(100)
            else:
                retention.append(self.analyzer.predict_long_term(temp_celsius, t))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_years, retention, 'b-', linewidth=3, label=f'{temp_celsius}°C prediction')
        ax.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='80% threshold')
        
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Molecular Weight Retention (%)', fontsize=12)
        ax.set_title(f'Long-term Degradation Prediction at {temp_celsius}°C')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        if save:
            plt.savefig(self.output_dir / f'prediction_{temp_celsius}C.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        self.plot_degradation_curves()
        self.plot_arrhenius()
        self.plot_prediction()
        print(f"All plots saved to {self.output_dir}")

# Usage
if __name__ == "__main__":
    from kinetic_analysis import KineticAnalyzer
    
    analyzer = KineticAnalyzer()
    analyzer.fit_temperature_curves()
    analyzer.arrhenius_analysis()
    
    visualizer = DegradationVisualizer(analyzer)
    visualizer.generate_all_plots()
