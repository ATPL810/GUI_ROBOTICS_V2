import tkinter as tk

class ToolPromptDialog:
    def __init__(self, parent, tool_mapping):
        self.parent = parent
        self.tool_mapping = tool_mapping
        self.selected_tool = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Fetch Tool")
        self.dialog.geometry("400x500")
        self.dialog.configure(bg="#1e1e2e")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.init_ui()
    
    def init_ui(self):
        # Title
        title_label = tk.Label(
            self.dialog,
            text="Select or Type Tool Name:",
            font=("Arial", 16, "bold"),
            fg="#89b4fa",
            bg="#1e1e2e"
        )
        title_label.pack(pady=10)
        
        # Available tools list
        list_label = tk.Label(
            self.dialog,
            text="Available Tools:",
            font=("Arial", 12),
            fg="#cdd6f4",
            bg="#1e1e2e"
        )
        list_label.pack(anchor="w", padx=20)
        
        # Listbox for tools
        self.tools_listbox = tk.Listbox(
            self.dialog,
            bg="#313244",
            fg="#cdd6f4",
            selectbackground="#89b4fa",
            selectforeground="#1e1e2e",
            font=("Arial", 12),
            height=8
        )
        self.tools_listbox.pack(fill=tk.BOTH, expand=True, padx=20, pady=(5, 10))
        
        # Populate list with counts
        for tool_name in sorted(self.tool_mapping.keys()):
            locations = self.tool_mapping[tool_name]
            total_count = len(locations)
            available_count = sum(1 for loc in locations if not loc.get("fetched", False))
            
            if total_count > 1:
                display_text = f"{tool_name.upper()} ({available_count}/{total_count} available)"
            else:
                display_text = f"{tool_name.upper()} ({available_count}/{total_count} available)"
            
            self.tools_listbox.insert(tk.END, display_text)
            
            # Gray out if none available
            if available_count == 0:
                self.tools_listbox.itemconfig(tk.END, {'fg': '#585b70'})
        
        self.tools_listbox.bind('<<ListboxSelect>>', self.on_list_select)
        self.tools_listbox.bind('<Double-Button-1>', self.on_double_click)
        # Or type manually
        type_label = tk.Label(
            self.dialog,
            text="Or type tool name:",
            font=("Arial", 12),
            fg="#cdd6f4",
            bg="#1e1e2e"
        )
        type_label.pack(anchor="w", padx=20)
        
        self.tool_input = tk.Entry(
            self.dialog,
            bg="#313244",
            fg="#cdd6f4",
            insertbackground="#cdd6f4",
            font=("Arial", 12),
            width=30
        )
        self.tool_input.pack(padx=20, pady=(5, 20), fill=tk.X)
        self.tool_input.bind('<KeyRelease>', self.check_input)
        
        # Buttons
        button_frame = tk.Frame(self.dialog, bg="#1e1e2e")
        button_frame.pack(pady=(0, 20))
        
        self.ok_button = tk.Button(
            button_frame,
            text="Fetch",
            command=self.on_ok,
            bg="#585b70",
            fg="#cdd6f4",
            font=("Arial", 12, "bold"),
            width=12,
            state=tk.DISABLED
        )
        self.ok_button.pack(side=tk.LEFT, padx=10)
        
        self.cancel_button = tk.Button(
            button_frame,
            text="Cancel",
            command=self.on_cancel,
            bg="#585b70",
            fg="#cdd6f4",
            font=("Arial", 12, "bold"),
            width=12
        )
        self.cancel_button.pack(side=tk.LEFT, padx=10)
        
    def on_list_select(self, event):
        selection = self.tools_listbox.curselection()
        if selection:
            # Extract tool name from display text (removing count info)
            display_text = self.tools_listbox.get(selection[0])
            
            # Extract just the tool name (before the first space)
            # Example: "HAMMER (2/3 available)" -> "HAMMER"
            tool_name_parts = display_text.split(' ')
            if tool_name_parts:
                tool_name = tool_name_parts[0].lower()
                self.tool_input.delete(0, tk.END)
                self.tool_input.insert(0, tool_name)
                
                # Check if tool has available instances
                if tool_name in self.tool_mapping:
                    locations = self.tool_mapping[tool_name]
                    available_count = sum(1 for loc in locations if not loc.get("fetched", False))
                    if available_count > 0:
                        self.ok_button.config(state=tk.NORMAL)
                        self.tool_input.config(fg="#cdd6f4")
                    else:
                        self.ok_button.config(state=tk.DISABLED)
                        self.tool_input.config(fg="#f38ba8")  # Red text if none available
                else:
                    self.ok_button.config(state=tk.NORMAL)
                    self.tool_input.config(fg="#cdd6f4")
        
    def on_double_click(self, event):
        selection = self.tools_listbox.curselection()
        if selection:
            display_text = self.tools_listbox.get(selection[0])
            tool_name_parts = display_text.split(' ')
            if tool_name_parts:
                tool_name = tool_name_parts[0].lower()
                
                # Check availability before allowing double-click fetch
                if tool_name in self.tool_mapping:
                    locations = self.tool_mapping[tool_name]
                    available_count = sum(1 for loc in locations if not loc.get("fetched", False))
                    if available_count > 0:
                        self.on_ok()
    
    def check_input(self, event):
        tool_name = self.tool_input.get().strip().lower()
        if tool_name:
            # Check if tool exists and has available instances
            if tool_name in self.tool_mapping:
                locations = self.tool_mapping[tool_name]
                available_count = sum(1 for loc in locations if not loc.get("fetched", False))
                if available_count > 0:
                    self.ok_button.config(state=tk.NORMAL)
                    self.tool_input.config(fg="#cdd6f4")
                else:
                    self.ok_button.config(state=tk.DISABLED)
                    self.tool_input.config(fg="#f38ba8")
            else:
                self.ok_button.config(state=tk.NORMAL)
                self.tool_input.config(fg="#cdd6f4")
        else:
            self.ok_button.config(state=tk.DISABLED)
    
    def on_ok(self):
        tool_name = self.tool_input.get().strip().lower()
        if tool_name:
            self.selected_tool = tool_name
            self.dialog.destroy()
    
    def on_cancel(self):
        self.selected_tool = None
        self.dialog.destroy()
    
    def show(self):
        self.dialog.wait_window()
        return self.selected_tool