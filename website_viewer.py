import webview
import tkinter as tk
from tkinter import ttk
import threading
import json
import os
import urllib.parse
import requests

class WebsiteViewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Website Control Panel")
        self.root.geometry("400x220")
        
        # Theme colors
        self.themes = {
            'dark': {
                'bg': '#2E2E2E',
                'fg': '#FFFFFF',
                'button_bg': '#404040',
                'button_fg': '#FFFFFF',
                'entry_bg': '#404040',
                'entry_fg': '#FFFFFF'
            },
            'light': {
                'bg': '#F0F0F0',
                'fg': '#000000',
                'button_bg': '#E0E0E0',
                'button_fg': '#000000',
                'entry_bg': '#FFFFFF',
                'entry_fg': '#000000'
            }
        }
        
        # Load saved theme preference
        self.current_theme = self.load_theme_preference()
        
        # Make window stay on top and set minimum size
        self.root.attributes('-topmost', True)
        self.root.minsize(400, 220)
        
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        
        # URL Entry
        self.url_label = ttk.Label(self.main_frame, text="URL:")
        self.url_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.url_entry = ttk.Entry(self.main_frame)
        self.url_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.url_entry.insert(0, "http://localhost:3000")
        
        # Window size controls
        self.size_frame = ttk.LabelFrame(self.main_frame, text="Window Size", padding="5")
        self.size_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        self.width_label = ttk.Label(self.size_frame, text="Width:")
        self.width_label.grid(row=0, column=0, padx=5, pady=5)
        self.width_entry = ttk.Entry(self.size_frame, width=8)
        self.width_entry.insert(0, "1024")
        self.width_entry.grid(row=0, column=1, padx=5, pady=5)
        
        self.height_label = ttk.Label(self.size_frame, text="Height:")
        self.height_label.grid(row=0, column=2, padx=5, pady=5)
        self.height_entry = ttk.Entry(self.size_frame, width=8)
        self.height_entry.insert(0, "768")
        self.height_entry.grid(row=0, column=3, padx=5, pady=5)
        
        # Buttons frame
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=2, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        self.button_frame.columnconfigure(0, weight=1)
        self.button_frame.columnconfigure(1, weight=1)
        
        # Control buttons with icons
        self.open_button = ttk.Button(self.button_frame, text="🌐 Open Website", command=self.open_website)
        self.open_button.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        self.close_button = ttk.Button(self.button_frame, text="🚫 Close Website", command=self.close_website)
        self.close_button.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Theme toggle
        self.theme_button = ttk.Button(self.main_frame, text="🌓 Toggle Theme", command=self.toggle_theme)
        self.theme_button.grid(row=3, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E))
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.grid(row=4, column=0, columnspan=3, pady=(5,0), sticky=(tk.W, tk.E))
        
        self.window = None
        
        # Apply initial theme
        self.apply_theme(self.current_theme)

    def validate_url(self, url):
        try:
            # Parse the URL
            parsed = urllib.parse.urlparse(url)
            
            # Check if scheme is present, if not add https://
            if not parsed.scheme:
                url = 'https://' + url
                parsed = urllib.parse.urlparse(url)
            
            # Try to connect to the URL
            response = requests.head(url, timeout=5)
            return url, True, "URL is valid"
        except requests.RequestException as e:
            return url, False, f"Error accessing URL: {str(e)}"
        except Exception as e:
            return url, False, f"Invalid URL format: {str(e)}"
        
    def load_theme_preference(self):
        try:
            if os.path.exists('theme_preference.json'):
                with open('theme_preference.json', 'r') as f:
                    return json.load(f)['theme']
        except:
            pass
        return 'light'
    
    def save_theme_preference(self):
        with open('theme_preference.json', 'w') as f:
            json.dump({'theme': self.current_theme}, f)
    
    def toggle_theme(self):
        self.current_theme = 'light' if self.current_theme == 'dark' else 'dark'
        self.apply_theme(self.current_theme)
        self.save_theme_preference()
    
    def apply_theme(self, theme_name):
        theme = self.themes[theme_name]
        style = ttk.Style()
        
        # Configure TTK styles
        style.configure('TFrame', background=theme['bg'])
        style.configure('TLabel', background=theme['bg'], foreground=theme['fg'])
        style.configure('TButton', background=theme['button_bg'], foreground=theme['button_fg'])
        style.configure('TEntry', fieldbackground=theme['entry_bg'], foreground=theme['entry_fg'])
        style.configure('TLabelframe', background=theme['bg'], foreground=theme['fg'])
        style.configure('TLabelframe.Label', background=theme['bg'], foreground=theme['fg'])
        
        # Apply colors to the main window
        self.root.configure(bg=theme['bg'])
        self.main_frame.configure(style='TFrame')
        
        # Update status bar appearance
        self.status_bar.configure(background=theme['bg'], foreground=theme['fg'])
    
    def open_website(self):
        if self.window is None:
            try:
                url = self.url_entry.get()
                width = int(self.width_entry.get())
                height = int(self.height_entry.get())
                
                self.status_var.set("Validating URL...")
                url, is_valid, message = self.validate_url(url)
                
                if not is_valid:
                    self.status_var.set(message)
                    return
                
                self.status_var.set(f"Opening {url}...")
                self.url_entry.delete(0, tk.END)
                self.url_entry.insert(0, url)
                
                def open_webview():
                    self.window = webview.create_window('Website Viewer', url, width=width, height=height)
                    webview.start()
                
                threading.Thread(target=open_webview, daemon=True).start()
                self.status_var.set("Website opened successfully")
            except ValueError:
                self.status_var.set("Error: Invalid width or height values")
            except Exception as e:
                self.status_var.set(f"Error: {str(e)}")
    
    def close_website(self):
        if self.window:
            self.window.destroy()
            self.window = None
            self.status_var.set("Website closed")
        else:
            self.status_var.set("No website window is open")
    
    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    viewer = WebsiteViewer()
    viewer.run()
