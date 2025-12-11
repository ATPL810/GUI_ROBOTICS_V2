import tkinter as tk
from tkinter import ttk

class ToolPromptDialog:
    def __init__(self, parent, tool_mapping):
        self.parent = parent
        self.tool_mapping = tool_mapping
        self.selected_tool = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("🤖 Fetch Tool")
        self.dialog.geometry("400x450")  # Slightly taller
        self.dialog.configure(bg="#1e1e2e")
        self.dialog.resizable(False, False)
        
        # Center the dialog
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        x = parent.winfo_rootx() + (parent.winfo_width() // 2) - (400 // 2)
        y = parent.winfo_rooty() + (parent.winfo_height() // 2) - (450 // 2)
        self.dialog.geometry(f"+{x}+{y}")
        
        self.init_ui()
    
    def init_ui(self):
        # Title
        title_label = tk.Label(self.dialog, text="Select or Type Tool Name:", 
                              font=("Arial", 14, "bold"), fg="#89b4fa", bg="#1e1e2e")
        title_label.pack(pady=10)
        
        # Available tools label
        list_label = tk.Label(self.dialog, text="Available Tools:", 
                             font=("Arial", 11), fg="#cdd6f4", bg="#1e1e2e")
        list_label.pack(anchor="w", padx=20)
        
        # Tools list
        list_frame = tk.Frame(self.dialog, bg="#313244")
        list_frame.pack(padx=20, pady=5, fill="both", expand=True)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tools_list = tk.Listbox(list_frame, bg="#313244", fg="#cdd6f4", 
                                    font=("Arial", 11), yscrollcommand=scrollbar.set,
                                    selectbackground="#585b70", selectforeground="#cdd6f4",
                                    borderwidth=2, relief="solid", highlightthickness=0,
                                    height=8)
        self.tools_list.pack(side=tk.LEFT, fill="both", expand=True)
        
        scrollbar.config(command=self.tools_list.yview)
        
        for tool_name in sorted(self.tool_mapping.keys()):
            locations = self.tool_mapping[tool_name]
            available = sum(1 for loc in locations if not loc.get("fetched", False))
            total = len(locations)
            
            if available > 0:
                if total > 1:
                    self.tools_list.insert(tk.END, f"{tool_name.upper()} ({available}/{total} available)")
                else:
                    self.tools_list.insert(tk.END, f"{tool_name.upper()}")
        
        self.tools_list.bind("<<ListboxSelect>>", self.on_list_select)
        self.tools_list.bind("<Double-Button-1>", self.on_double_click)
        
        # Type label
        type_label = tk.Label(self.dialog, text="Or type tool name:", 
                             font=("Arial", 11), fg="#cdd6f4", bg="#1e1e2e")
        type_label.pack(anchor="w", padx=20, pady=(10, 0))
        
        # Input field
        self.tool_input = tk.Entry(self.dialog, font=("Arial", 11), 
                                  bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
                                  borderwidth=2, relief="solid")
        self.tool_input.pack(padx=20, pady=5, fill="x")
        self.tool_input.bind("<KeyRelease>", self.on_input_change)
        self.tool_input.bind("<Return>", lambda e: self.on_ok())  # Enter key support
        
        # Buttons frame
        button_frame = tk.Frame(self.dialog, bg="#1e1e2e")
        button_frame.pack(pady=20)
        
        self.ok_button = tk.Button(button_frame, text="✅ Fetch", 
                                  font=("Arial", 11, "bold"),
                                  bg="#585b70", fg="#cdd6f4",
                                  activebackground="#89b4fa", activeforeground="#1e1e2e",
                                  borderwidth=2, relief="solid",
                                  width=10, state="disabled",
                                  command=self.on_ok)
        self.ok_button.pack(side=tk.LEFT, padx=5)
        
        self.cancel_button = tk.Button(button_frame, text="❌ Cancel", 
                                      font=("Arial", 11, "bold"),
                                      bg="#585b70", fg="#cdd6f4",
                                      activebackground="#89b4fa", activeforeground="#1e1e2e",
                                      borderwidth=2, relief="solid",
                                      width=10, command=self.on_cancel)
        self.cancel_button.pack(side=tk.LEFT, padx=5)
    
    def on_list_select(self, event):
        """Handle list selection"""
        selection = self.tools_list.curselection()
        if selection:
            tool_with_info = self.tools_list.get(selection[0])
            # Extract just the tool name (remove inventory info)
            tool = tool_with_info.split()[0].lower()
            self.tool_input.delete(0, tk.END)
            self.tool_input.insert(0, tool)
            self.ok_button.config(state="normal")
    
    def on_double_click(self, event):
        """Handle double click"""
        self.on_ok()
    
    def on_input_change(self, event):
        """Handle input change"""
        if self.tool_input.get().strip():
            self.ok_button.config(state="normal")
        else:
            self.ok_button.config(state="disabled")
    
    def on_ok(self):
        """OK button handler"""
        tool_name = self.tool_input.get().strip().lower()
        if tool_name:
            self.selected_tool = tool_name
            self.dialog.destroy()
    
    def on_cancel(self):
        """Cancel button handler"""
        self.selected_tool = None
        self.dialog.destroy()
    
    def show(self):
        """Show dialog and wait for result"""
        self.parent.wait_window(self.dialog)
        return self.selected_tool