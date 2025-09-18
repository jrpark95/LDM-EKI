#!/usr/bin/env python3
"""
Script to automatically add function tracking to all Python files
"""
import os
import re
import ast
import astor

class FunctionTracker(ast.NodeTransformer):
    def __init__(self):
        self.functions_found = []
        
    def visit_FunctionDef(self, node):
        # Skip functions that already have @track_function decorator
        if not any(isinstance(d, ast.Name) and d.id == 'track_function' for d in node.decorator_list):
            # Add @track_function decorator
            track_decorator = ast.Name(id='track_function', ctx=ast.Load())
            node.decorator_list.insert(0, track_decorator)
            
        # Track function name
        self.functions_found.append(node.name)
        
        # Continue visiting child nodes
        self.generic_visit(node)
        return node

def add_tracking_to_file(filepath):
    """Add tracking decorators to all functions in a Python file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        
        # Transform the AST
        transformer = FunctionTracker()
        new_tree = transformer.visit(tree)
        
        if transformer.functions_found:
            # Check if import already exists
            has_import = 'from function_tracker import track_function' in content
            
            # Generate new code
            new_content = astor.to_source(new_tree)
            
            # Add import if not present
            if not has_import:
                import_line = "from function_tracker import track_function\n"
                new_content = import_line + new_content
            
            # Write back to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"✓ {filepath}: Added tracking to {len(transformer.functions_found)} functions")
            return transformer.functions_found
        else:
            print(f"- {filepath}: No functions found")
            return []
            
    except Exception as e:
        print(f"✗ {filepath}: Error - {e}")
        return []

def main():
    """Main function to process all Python files"""
    eki_dir = '/home/jrpark/EKI-LDM5-dev/eki-20241030'
    
    # Find all Python files
    python_files = []
    for filename in os.listdir(eki_dir):
        if filename.endswith('.py') and filename not in ['function_tracker.py', 'add_tracking.py']:
            python_files.append(os.path.join(eki_dir, filename))
    
    total_functions = 0
    
    print("Adding function tracking to Python files...")
    print("=" * 50)
    
    for filepath in python_files:
        functions = add_tracking_to_file(filepath)
        total_functions += len(functions)
    
    print("=" * 50)
    print(f"Total functions tracked: {total_functions}")
    print("Tracking decorators added successfully!")
    print()
    print("To view usage report after running your code:")
    print("  from function_tracker import print_usage_report")
    print("  print_usage_report()")

if __name__ == '__main__':
    main()