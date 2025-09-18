import functools
import json
import os
from collections import defaultdict
import time

# Global tracking dictionary
FUNCTION_USAGE = defaultdict(lambda: {'calls': 0, 'total_time': 0, 'last_called': None})
TRACKING_FILE = '/home/jrpark/EKI-LDM5-dev/function_usage_log.json'

def track_function(func):
    """Decorator to track function usage and execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = f"{func.__module__}.{func.__name__}"
        
        # Update tracking info
        FUNCTION_USAGE[func_name]['calls'] += 1
        FUNCTION_USAGE[func_name]['last_called'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Record execution time
            execution_time = time.time() - start_time
            FUNCTION_USAGE[func_name]['total_time'] += execution_time
            
            # Save to file immediately (for crash safety)
            save_tracking_data()
    
    return wrapper

def save_tracking_data():
    """Save tracking data to JSON file"""
    try:
        with open(TRACKING_FILE, 'w') as f:
            json.dump(dict(FUNCTION_USAGE), f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save tracking data: {e}")

def load_tracking_data():
    """Load existing tracking data"""
    global FUNCTION_USAGE
    try:
        if os.path.exists(TRACKING_FILE):
            with open(TRACKING_FILE, 'r') as f:
                data = json.load(f)
                for func_name, stats in data.items():
                    FUNCTION_USAGE[func_name].update(stats)
    except Exception as e:
        print(f"Warning: Could not load tracking data: {e}")

def get_unused_functions():
    """Return list of functions that were never called"""
    return [func for func, stats in FUNCTION_USAGE.items() if stats['calls'] == 0]

def print_usage_report():
    """Print comprehensive usage report"""
    print("\n=== FUNCTION USAGE REPORT ===")
    
    # Sort by call count
    sorted_funcs = sorted(FUNCTION_USAGE.items(), key=lambda x: x[1]['calls'], reverse=True)
    
    used_funcs = [(f, s) for f, s in sorted_funcs if s['calls'] > 0]
    unused_funcs = [(f, s) for f, s in sorted_funcs if s['calls'] == 0]
    
    print(f"\nUSED FUNCTIONS ({len(used_funcs)}):")
    for func_name, stats in used_funcs:
        print(f"  {func_name}: {stats['calls']} calls, {stats['total_time']:.3f}s total")
    
    print(f"\nUNUSED FUNCTIONS ({len(unused_funcs)}):")
    for func_name, stats in unused_funcs:
        print(f"  {func_name}: NEVER CALLED")
    
    print(f"\nSUMMARY:")
    print(f"  Total functions tracked: {len(FUNCTION_USAGE)}")
    print(f"  Used functions: {len(used_funcs)}")
    print(f"  Unused functions: {len(unused_funcs)}")
    print(f"  Usage rate: {len(used_funcs)/len(FUNCTION_USAGE)*100:.1f}%")

# Load existing data on import
load_tracking_data()