import json
import os
from datetime import datetime

class ToolInventory:
    def __init__(self, report_file="master_report.json"):
        self.report_file = report_file
        self.tool_locations = {}
        self.load()
    
    def load(self):
        """Load tool locations from JSON file"""
        if os.path.exists(self.report_file):
            try:
                with open(self.report_file, 'r') as f:
                    self.tool_locations = json.load(f)
                print(f"✓ Loaded inventory from {self.report_file}")
            except:
                print(f"⚠️  Could not load {self.report_file}, starting fresh")
                self.tool_locations = {}
        else:
            print(f"📄 No {self.report_file} found, starting fresh")
            self.tool_locations = {}
    
    def save(self):
        """Save tool locations to JSON file"""
        with open(self.report_file, 'w') as f:
            json.dump(self.tool_locations, f, indent=2)
        print(f"💾 Inventory saved to {self.report_file}")
    
    def update_point(self, point, tools_list):
        """Update tools at a specific point"""
        self.tool_locations[point] = {
            'tools': tools_list,
            'last_updated': datetime.now().isoformat()
        }
        self.save()
    
    def get_all_tools(self):
        """Get dictionary of {tool_name: point_location}"""
        all_tools = {}
        for point, data in self.tool_locations.items():
            for tool in data.get('tools', []):
                all_tools[tool] = point
        return all_tools
    
    def get_tool_location(self, tool_name):
        """Find which point a tool is at, or None if not found"""
        for point, data in self.tool_locations.items():
            if tool_name in data.get('tools', []):
                return point
        return None
    
    def remove_tool(self, tool_name):
        """Remove a tool after it's been grabbed"""
        point = self.get_tool_location(tool_name)
        if point and point in self.tool_locations:
            if tool_name in self.tool_locations[point]['tools']:
                self.tool_locations[point]['tools'].remove(tool_name)
                self.save()
                print(f"🗑️  Removed {tool_name} from inventory")
                return True
        return False
    
    def print_inventory(self):
        """Display all available tools"""
        print("\n" + "="*50)
        print("🛠️  AVAILABLE TOOLS INVENTORY")
        print("="*50)
        
        if not self.tool_locations:
            print("No tools detected. Run --scan first")
            return
        
        tool_count = 0
        for point, data in sorted(self.tool_locations.items()):
            tools = data.get('tools', [])
            if tools:
                print(f"\n📍 Point {point}:")
                for tool in tools:
                    print(f"   • {tool}")
                    tool_count += 1
        
        print(f"\n📊 Total: {tool_count} tools available")
        print("="*50)
    
    def is_tool_available(self, tool_name):
        """Check if a tool exists in inventory"""
        return self.get_tool_location(tool_name) is not None