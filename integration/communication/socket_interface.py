#!/usr/bin/env python3
"""
Socket interface for LDM-EKI communication
Handles data exchange between LDM server and EKI client
"""

import socket
import struct
import numpy as np
import json
import time

class LDMEKIInterface:
    def __init__(self, host='127.0.0.1', port=8080):
        self.host = host
        self.port = port
        self.socket = None
        
    def connect_to_ldm(self):
        """Connect to LDM server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            print(f"Connected to LDM server at {self.host}:{self.port}")
            return True
        except ConnectionRefusedError:
            print(f"Failed to connect to LDM server at {self.host}:{self.port}")
            return False
    
    def send_source_parameters(self, source_params):
        """Send source parameters to LDM"""
        if not self.socket:
            raise RuntimeError("Not connected to LDM server")
            
        # Convert numpy array to bytes
        data = source_params.tobytes()
        size = len(data)
        
        # Send size first, then data
        self.socket.send(struct.pack('I', size))
        self.socket.send(data)
        
    def receive_gamma_dose(self, nreceptor, time_intervals):
        """Receive gamma dose matrix from LDM"""
        if not self.socket:
            raise RuntimeError("Not connected to LDM server")
            
        # Calculate expected data size
        expected_size = nreceptor * time_intervals * 8  # 8 bytes per double
        
        # Receive size header
        size_data = self.socket.recv(4)
        if len(size_data) != 4:
            raise RuntimeError("Failed to receive size header")
            
        size = struct.unpack('I', size_data)[0]
        
        # Receive actual data
        data = b''
        while len(data) < size:
            chunk = self.socket.recv(size - len(data))
            if not chunk:
                raise RuntimeError("Connection closed unexpectedly")
            data += chunk
        
        # Convert back to numpy array
        gamma_dose = np.frombuffer(data, dtype=np.float64)
        return gamma_dose.reshape(nreceptor, time_intervals)
    
    def close(self):
        """Close connection"""
        if self.socket:
            self.socket.close()
            self.socket = None
            print("Connection to LDM server closed")

class EKIStatusReporter:
    """Reports EKI status and progress to monitoring systems"""
    
    def __init__(self, log_file="integration_logs/eki_status.log"):
        self.log_file = log_file
        
    def report_iteration(self, iteration, convergence_metrics):
        """Report EKI iteration progress"""
        status = {
            'timestamp': time.time(),
            'iteration': iteration,
            'convergence': convergence_metrics,
            'status': 'running'
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(status) + '\n')
    
    def report_completion(self, final_results):
        """Report EKI completion"""
        status = {
            'timestamp': time.time(),
            'status': 'completed',
            'results': final_results
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(status) + '\n')