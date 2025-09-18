#!/usr/bin/env python3
"""
Coupled LDM-EKI simulation workflow
Orchestrates the integrated simulation process
"""

import subprocess
import time
import os
import sys
import signal
from pathlib import Path

# Add EKI source to path
sys.path.append(str(Path(__file__).parent.parent.parent / "eki" / "src"))

from integration.communication.socket_interface import LDMEKIInterface, EKIStatusReporter

class CoupledSimulation:
    def __init__(self, config_file=None):
        self.config_file = config_file or "integration/configs/ldm_eki_config.json"
        self.ldm_process = None
        self.interface = LDMEKIInterface()
        self.status_reporter = EKIStatusReporter()
        
    def start_ldm_server(self):
        """Start LDM server in background"""
        ldm_path = Path(__file__).parent.parent.parent / "ldm"
        
        print("Starting LDM server...")
        self.ldm_process = subprocess.Popen(
            ["./ldm"],
            cwd=ldm_path,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
        )
        
        # Wait for server to start
        time.sleep(5)
        
        # Check if process is still running
        if self.ldm_process.poll() is not None:
            raise RuntimeError("LDM server failed to start")
            
        print("LDM server started successfully")
        
    def run_eki_client(self):
        """Run EKI client"""
        eki_path = Path(__file__).parent.parent.parent / "eki"
        
        print("Starting EKI client...")
        result = subprocess.run([
            "python", "src/RunEstimator.py", 
            "config/input_config", 
            "config/input_data"
        ], cwd=eki_path, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"EKI client failed: {result.stderr}")
            return False
        
        print("EKI client completed successfully")
        return True
        
    def cleanup(self):
        """Clean up processes and connections"""
        print("Cleaning up...")
        
        if self.ldm_process:
            self.ldm_process.terminate()
            try:
                self.ldm_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.ldm_process.kill()
                self.ldm_process.wait()
            print("LDM server stopped")
            
        self.interface.close()
        
    def run(self):
        """Run complete coupled simulation"""
        try:
            # Start LDM server
            self.start_ldm_server()
            
            # Run EKI client
            success = self.run_eki_client()
            
            if success:
                print("Coupled simulation completed successfully")
                self.status_reporter.report_completion({"status": "success"})
            else:
                print("Coupled simulation failed")
                self.status_reporter.report_completion({"status": "failed"})
                
        except KeyboardInterrupt:
            print("Simulation interrupted by user")
        except Exception as e:
            print(f"Simulation failed with error: {e}")
            self.status_reporter.report_completion({"status": "error", "error": str(e)})
        finally:
            self.cleanup()

if __name__ == "__main__":
    simulation = CoupledSimulation()
    simulation.run()