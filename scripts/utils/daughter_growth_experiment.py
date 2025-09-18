#!/usr/bin/env python3
"""
CRAM 딸핵종 증가 검증 실험 시스템

목표: 5개 핵심 부모-딸 체인에서 딸핵종의 초기 증가를 확인
- Sr-92 → Y-92
- I-135 → Xe-135  
- Ba-140 → La-140
- Zr-95 → Nb-95
- Mo-99 → Tc-99m
"""

import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from pathlib import Path

class DaughterGrowthExperiment:
    def __init__(self):
        self.base_dir = Path("/home/jrpark/ldm-CRAM")
        self.exp_dir = self.base_dir / "experiments" / "daughter_growth"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Core test chains (parent → daughter)
        self.test_chains = [
            {"name": "Sr92_Y92", "parent": "Sr-92", "daughter": "Y-92", "parent_idx": 12, "daughter_idx": 13},
            {"name": "I135_Xe135", "parent": "I-135", "daughter": "Xe-135", "parent_idx": 38, "daughter_idx": 39},
            {"name": "Ba140_La140", "parent": "Ba-140", "daughter": "La-140", "parent_idx": 43, "daughter_idx": 44},
            {"name": "Zr95_Nb95", "parent": "Zr-95", "daughter": "Nb-95", "parent_idx": 15, "daughter_idx": 16},
            {"name": "Mo99_Tc99m", "parent": "Mo-99", "daughter": "Tc-99m", "parent_idx": 18, "daughter_idx": 19}
        ]
        
        # Load A matrix for theoretical calculations
        self.A_matrix = self.load_A_matrix()
        
    def load_A_matrix(self):
        """Load A60.csv matrix for theoretical calculations"""
        try:
            A = np.loadtxt(self.base_dir / "cram" / "A60.csv", delimiter=',')
            print(f"✅ Loaded A matrix: {A.shape}")
            return A
        except Exception as e:
            print(f"❌ Failed to load A matrix: {e}")
            return None
    
    def load_nuclide_data(self):
        """Load nuclide data from existing config"""
        nuclides = []
        config_file = self.base_dir / "input" / "nuclides_daughter_stress_test.txt"
        
        with open(config_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    name = parts[0]
                    lambda_val = float(parts[1])
                    initial_ratio = float(parts[2])
                    nuclides.append({
                        'name': name,
                        'lambda': lambda_val,
                        'initial_ratio': initial_ratio
                    })
        
        return nuclides
    
    def create_test_case_config(self, chain, parent_initial=1.0):
        """Create nuclide config file for specific test case"""
        nuclides = self.load_nuclide_data()
        
        # Reset all to 0
        for nuc in nuclides:
            nuc['initial_ratio'] = 0.0
        
        # Set parent to specified initial value
        for nuc in nuclides:
            if nuc['name'] == chain['parent']:
                nuc['initial_ratio'] = parent_initial
                break
        
        # Write config file
        config_file = self.exp_dir / f"{chain['name']}_config.txt"
        with open(config_file, 'w') as f:
            for nuc in nuclides:
                f.write(f"{nuc['name']},{nuc['lambda']},{nuc['initial_ratio']}\n")
        
        return config_file
    
    def calculate_theoretical_growth_rate(self, chain):
        """Calculate theoretical initial growth rate dC_j/dt|_0 for daughter"""
        if self.A_matrix is None:
            return None
            
        parent_idx = chain['parent_idx']
        daughter_idx = chain['daughter_idx']
        
        # Production rate from parent: a_ji * C_i(0) where C_i(0) = 1.0
        production_rate = self.A_matrix[daughter_idx, parent_idx] * 1.0
        
        # Decay rate of daughter: -λ_j * C_j(0) where C_j(0) = 0.0
        decay_rate = self.A_matrix[daughter_idx, daughter_idx] * 0.0
        
        # Net initial growth rate
        net_rate = production_rate + decay_rate  # decay_rate is 0 since C_j(0)=0
        
        return {
            'production_rate': production_rate,
            'decay_rate': self.A_matrix[daughter_idx, daughter_idx],  # λ_j (negative)
            'net_initial_rate': net_rate
        }
    
    def run_simulation(self, chain):
        """Run LDM simulation for specific test chain"""
        print(f"\n🧪 Running simulation for {chain['name']}: {chain['parent']} → {chain['daughter']}")
        
        # Create test case configuration
        config_file = self.create_test_case_config(chain)
        print(f"📝 Created config: {config_file}")
        
        # Calculate theoretical growth rate
        theory = self.calculate_theoretical_growth_rate(chain)
        if theory:
            print(f"📊 Theoretical initial dC_j/dt = {theory['net_initial_rate']:.2e} s⁻¹")
            print(f"   Production: {theory['production_rate']:.2e} s⁻¹")
            print(f"   Decay λ_j: {theory['decay_rate']:.2e} s⁻¹")
        
        # Update main.cu to use this config
        self.update_main_config_file(config_file)
        
        # Compile and run
        try:
            # Clean and compile
            result = subprocess.run(['make', 'clean'], cwd=self.base_dir, 
                                  capture_output=True, text=True, check=True)
            result = subprocess.run(['make', '-j4'], cwd=self.base_dir, 
                                  capture_output=True, text=True, check=True)
            
            # Run simulation with timeout
            result = subprocess.run(['timeout', '120', './ldm'], cwd=self.base_dir,
                                  capture_output=True, text=True, check=True)
            
            print(f"✅ Simulation completed for {chain['name']}")
            
            # Move output file to experiment directory
            output_file = self.base_dir / "all_particles_nuclide_ratios.csv"
            if output_file.exists():
                chain_output = self.exp_dir / f"{chain['name']}_results.csv"
                output_file.rename(chain_output)
                return chain_output
            else:
                print(f"❌ No output file found for {chain['name']}")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Simulation failed for {chain['name']}: {e}")
            if e.stdout: print(f"STDOUT: {e.stdout}")
            if e.stderr: print(f"STDERR: {e.stderr}")
            return None
    
    def update_main_config_file(self, config_file):
        """Update main.cu to use specific config file"""
        main_file = self.base_dir / "main.cu"
        
        with open(main_file, 'r') as f:
            content = f.read()
        
        # Replace the config file line
        old_line = 'std::string nuclide_config_file = "./input/nuclides_daughter_stress_test.txt";'
        new_line = f'std::string nuclide_config_file = "{config_file}";'
        content = content.replace(old_line, new_line)
        
        with open(main_file, 'w') as f:
            f.write(content)
    
    def analyze_results(self, chain, result_file):
        """Analyze simulation results for daughter growth"""
        if not result_file or not result_file.exists():
            return {"success": False, "reason": "No results file"}
        
        try:
            df = pd.read_csv(result_file)
            
            # Get parent and daughter columns
            parent_col = f"total_Q_{chain['parent_idx']}"
            daughter_col = f"total_Q_{chain['daughter_idx']}"
            
            if parent_col not in df.columns or daughter_col not in df.columns:
                return {"success": False, "reason": f"Missing columns: {parent_col}, {daughter_col}"}
            
            # Get time series data
            times = df['time(s)'].values
            parent_conc = df[parent_col].values
            daughter_conc = df[daughter_col].values
            
            # Check for daughter growth in first hour
            first_hour_mask = times <= 3600
            if np.sum(first_hour_mask) < 2:
                return {"success": False, "reason": "Insufficient data points in first hour"}
            
            daughter_first_hour = daughter_conc[first_hour_mask]
            times_first_hour = times[first_hour_mask]
            
            # Check for positive growth
            initial_daughter = daughter_first_hour[0]
            max_daughter_1h = np.max(daughter_first_hour)
            
            # Success criteria
            growth_rate = (max_daughter_1h - initial_daughter) / (times_first_hour[-1] - times_first_hour[0]) if len(times_first_hour) > 1 else 0
            relative_growth = (max_daughter_1h / max(initial_daughter, 1e-12) - 1) * 100
            
            success = growth_rate > 1e-12 and max_daughter_1h > initial_daughter
            
            return {
                "success": success,
                "initial_daughter": initial_daughter,
                "max_daughter_1h": max_daughter_1h,
                "growth_rate": growth_rate,
                "relative_growth": relative_growth,
                "parent_initial": parent_conc[0] if len(parent_conc) > 0 else 0,
                "parent_final": parent_conc[-1] if len(parent_conc) > 0 else 0
            }
            
        except Exception as e:
            return {"success": False, "reason": f"Analysis error: {e}"}
    
    def create_visualization(self, chain, result_file, analysis):
        """Create visualization for test chain results"""
        if not result_file or not result_file.exists():
            return
        
        try:
            df = pd.read_csv(result_file)
            
            parent_col = f"total_Q_{chain['parent_idx']}"
            daughter_col = f"total_Q_{chain['daughter_idx']}"
            
            times_hours = df['time(s)'].values / 3600.0  # Convert to hours
            parent_conc = df[parent_col].values
            daughter_conc = df[daughter_col].values
            
            # Create two plots: full timeline and zoom to first 3 hours
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Full timeline (linear scale)
            ax1.plot(times_hours, parent_conc, 'b-', label=f'{chain["parent"]} (parent)', linewidth=2)
            ax1.plot(times_hours, daughter_conc, 'r-', label=f'{chain["daughter"]} (daughter)', linewidth=2)
            ax1.set_xlabel('Time (hours)')
            ax1.set_ylabel('Concentration')
            ax1.set_title(f'{chain["name"]}: Full Timeline')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Zoom to first 3 hours
            mask_3h = times_hours <= 3
            ax2.plot(times_hours[mask_3h], parent_conc[mask_3h], 'b-', 
                    label=f'{chain["parent"]} (parent)', linewidth=2)
            ax2.plot(times_hours[mask_3h], daughter_conc[mask_3h], 'r-', 
                    label=f'{chain["daughter"]} (daughter)', linewidth=2)
            ax2.set_xlabel('Time (hours)')
            ax2.set_ylabel('Concentration')
            ax2.set_title(f'{chain["name"]}: First 3 Hours (Daughter Growth Focus)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add analysis results as text
            if analysis:
                status = "✅ SUCCESS" if analysis["success"] else "❌ FAILED"
                growth_text = f"Growth Rate: {analysis.get('growth_rate', 0):.2e} s⁻¹"
                ax2.text(0.05, 0.95, f"{status}\n{growth_text}", 
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.exp_dir / f"{chain['name']}_plot.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Created visualization: {plot_file}")
            
        except Exception as e:
            print(f"❌ Failed to create visualization: {e}")
    
    def run_all_experiments(self):
        """Run all test chain experiments"""
        results = []
        
        print("🚀 Starting CRAM Daughter Nuclide Growth Experiments")
        print("=" * 60)
        
        for chain in self.test_chains:
            # Run simulation
            result_file = self.run_simulation(chain)
            
            # Analyze results
            analysis = self.analyze_results(chain, result_file)
            
            # Create visualization
            self.create_visualization(chain, result_file, analysis)
            
            # Store results
            result_summary = {
                "chain_name": chain['name'],
                "parent": chain['parent'],
                "daughter": chain['daughter'],
                **analysis
            }
            results.append(result_summary)
            
            # Print summary
            if analysis["success"]:
                print(f"✅ {chain['name']}: DAUGHTER GROWTH CONFIRMED!")
                print(f"   Growth rate: {analysis['growth_rate']:.2e} s⁻¹")
            else:
                print(f"❌ {chain['name']}: {analysis.get('reason', 'Growth not detected')}")
        
        # Generate final report
        self.generate_report(results)
        
        return results
    
    def generate_report(self, results):
        """Generate comprehensive experiment report"""
        report_file = self.exp_dir / "experiment_report.md"
        
        successful_chains = [r for r in results if r.get("success", False)]
        
        with open(report_file, 'w') as f:
            f.write("# CRAM Daughter Nuclide Growth Experiment Report\n\n")
            f.write(f"**Experiment Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Summary**: {len(successful_chains)}/{len(results)} chains showed daughter growth\n\n")
            
            f.write("## Test Chains\n\n")
            for result in results:
                status = "✅ SUCCESS" if result.get("success", False) else "❌ FAILED"
                f.write(f"### {result['chain_name']}: {result['parent']} → {result['daughter']} {status}\n\n")
                
                if result.get("success", False):
                    f.write(f"- **Growth Rate**: {result['growth_rate']:.2e} s⁻¹\n")
                    f.write(f"- **Initial Daughter**: {result['initial_daughter']:.2e}\n")
                    f.write(f"- **Max Daughter (1h)**: {result['max_daughter_1h']:.2e}\n")
                else:
                    f.write(f"- **Failure Reason**: {result.get('reason', 'Unknown')}\n")
                
                f.write(f"\n![{result['chain_name']} Plot]({result['chain_name']}_plot.png)\n\n")
            
            f.write("## Conclusions\n\n")
            if len(successful_chains) >= 3:
                f.write("🎉 **EXPERIMENT SUCCESS**: Multiple daughter nuclide chains show clear growth!\n\n")
                f.write("The CRAM implementation is working correctly and can model nuclear decay chains.\n")
            else:
                f.write("⚠️ **EXPERIMENT INCOMPLETE**: Need to investigate why growth is not detected.\n\n")
                f.write("Possible issues:\n")
                f.write("- CRAM matrix computation\n")
                f.write("- Kernel execution\n")
                f.write("- Output data processing\n")
        
        print(f"📄 Generated report: {report_file}")

def main():
    exp = DaughterGrowthExperiment()
    results = exp.run_all_experiments()
    
    successful = sum(1 for r in results if r.get("success", False))
    print("\n" + "=" * 60)
    print(f"🏁 EXPERIMENT COMPLETE: {successful}/{len(results)} chains successful")
    
    if successful >= 3:
        print("🎉 SUCCESS: CRAM daughter nuclide growth confirmed!")
    else:
        print("⚠️ INCOMPLETE: Further investigation needed")

if __name__ == "__main__":
    main()