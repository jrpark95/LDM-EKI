#!/usr/bin/env python3
import struct
import sys

def verify_single_observations(filepath):
    print(f"Verifying single observations: {filepath}")
    with open(filepath, 'rb') as f:
        nreceptors = struct.unpack('i', f.read(4))[0]
        T = struct.unpack('i', f.read(4))[0]
        print(f"  nreceptors: {nreceptors}")
        print(f"  T: {T}")
        
        Y = []
        for i in range(nreceptors * T):
            val = struct.unpack('f', f.read(4))[0]
            Y.append(val)
        
        sigma_rel = struct.unpack('f', f.read(4))[0]
        MDA = struct.unpack('f', f.read(4))[0]
        
        print(f"  Y shape: {nreceptors} x {T}")
        print(f"  Y sample: {Y[:5]}")
        print(f"  sigma_rel: {sigma_rel}")
        print(f"  MDA: {MDA}")

def verify_ensemble_observations(filepath):
    print(f"Verifying ensemble observations: {filepath}")
    with open(filepath, 'rb') as f:
        Nens = struct.unpack('i', f.read(4))[0]
        nreceptors = struct.unpack('i', f.read(4))[0]
        T = struct.unpack('i', f.read(4))[0]
        print(f"  Nens: {Nens}")
        print(f"  nreceptors: {nreceptors}")
        print(f"  T: {T}")
        
        Y_ens = []
        for i in range(Nens * nreceptors * T):
            val = struct.unpack('f', f.read(4))[0]
            Y_ens.append(val)
        
        print(f"  Y_ens shape: {Nens} x {nreceptors} x {T}")
        print(f"  Y_ens sample: {Y_ens[:5]}")

def verify_emission_data(filepath):
    print(f"Verifying emission data: {filepath}")
    with open(filepath, 'rb') as f:
        Nens = struct.unpack('i', f.read(4))[0]
        T = struct.unpack('i', f.read(4))[0]
        print(f"  Nens: {Nens}")
        print(f"  T: {T}")
        
        flat = []
        for i in range(Nens * T):
            val = struct.unpack('f', f.read(4))[0]
            flat.append(val)
        
        print(f"  emission flat shape: {Nens} x {T}")
        print(f"  emission sample: {flat[:5]}")

def verify_state_data(filepath):
    print(f"Verifying state data: {filepath}")
    with open(filepath, 'rb') as f:
        Nens = struct.unpack('i', f.read(4))[0]
        state_dim = struct.unpack('i', f.read(4))[0]
        print(f"  Nens: {Nens}")
        print(f"  state_dim: {state_dim}")
        
        X = []
        for i in range(Nens * state_dim):
            val = struct.unpack('f', f.read(4))[0]
            X.append(val)
        
        print(f"  X shape: {Nens} x {state_dim}")
        print(f"  X sample: {X[:5]}")

if __name__ == "__main__":
    base = "/home/jrpark/LDM-EKI/logs"
    
    print("=== Binary File Format Verification ===\n")
    
    try:
        verify_single_observations(f"{base}/ldm_logs/observations_single_iter000.bin")
        print()
        
        verify_ensemble_observations(f"{base}/eki_logs/observations_ens_iter000.bin")
        print()
        
        verify_emission_data(f"{base}/eki_logs/emission_iter000.bin")
        print()
        
        verify_state_data(f"{base}/eki_logs/states_iter000.bin")
        print()
        
        print("=== All binary files verified successfully! ===")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)