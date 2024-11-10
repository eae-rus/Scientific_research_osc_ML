import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from search_oscillograms import SearchOscillograms
import json

class OscillogramApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Oscillogram Copy Utility")

        # Path selection
        self.source_dir = tk.StringVar()
        self.dest_dir = tk.StringVar()
        self.hash_table_path = tk.StringVar()

        tk.Label(root, text="Source Directory:").grid(row=0, column=0, sticky="e")
        tk.Entry(root, textvariable=self.source_dir, width=50).grid(row=0, column=1)
        tk.Button(root, text="Browse...", command=self.select_source_dir).grid(row=0, column=2)

        tk.Label(root, text="Destination Directory:").grid(row=1, column=0, sticky="e")
        tk.Entry(root, textvariable=self.dest_dir, width=50).grid(row=1, column=1)
        tk.Button(root, text="Browse...", command=self.select_dest_dir).grid(row=1, column=2)

        tk.Label(root, text="Hash Table Files (json):").grid(row=2, column=0, sticky="e")
        tk.Entry(root, textvariable=self.hash_table_path, width=50).grid(row=2, column=1)
        tk.Button(root, text="Browse...", command=self.select_hash_table_path).grid(row=2, column=2)

        # Checkbox options for file types
        self.use_comtrade = tk.BooleanVar(value=True)
        self.use_new_comtrade = tk.BooleanVar(value=True)
        self.use_brs = tk.BooleanVar(value=True)
        self.use_neva = tk.BooleanVar(value=True)
        self.use_ekra = tk.BooleanVar(value=True)
        self.use_parma = tk.BooleanVar(value=False)
        self.use_black_box = tk.BooleanVar(value=True)
        self.use_res_3 = tk.BooleanVar(value=True)
        self.use_osc = tk.BooleanVar(value=True)
        self.preserve_dir_structure = tk.BooleanVar(value=True)
        self.is_write_names = True # TODO: tk.BooleanVar(value=False)


        tk.Checkbutton(root, text="Use Comtrade", variable=self.use_comtrade).grid(row=3, column=0, sticky="w")
        tk.Checkbutton(root, text="Use New Comtrade", variable=self.use_new_comtrade).grid(row=3, column=1, sticky="w")
        tk.Checkbutton(root, text="Use Bresler Files", variable=self.use_brs).grid(row=4, column=0, sticky="w")
        tk.Checkbutton(root, text="Use NEVA Files", variable=self.use_neva).grid(row=4, column=1, sticky="w")
        tk.Checkbutton(root, text="Use EKRA Files", variable=self.use_ekra).grid(row=5, column=0, sticky="w")
        tk.Checkbutton(root, text="Use Parma Files", variable=self.use_parma).grid(row=5, column=1, sticky="w")
        tk.Checkbutton(root, text="Use Black Box", variable=self.use_black_box).grid(row=6, column=0, sticky="w")
        tk.Checkbutton(root, text="Use RES_3 Files", variable=self.use_res_3).grid(row=6, column=1, sticky="w")
        tk.Checkbutton(root, text="Use OSC Files", variable=self.use_osc).grid(row=7, column=0, sticky="w")
        
        tk.Checkbutton(root, text="Preserve dir structure", variable=self.preserve_dir_structure).grid(row=3, column=2, sticky="w")
        # TODO: tk.Checkbutton(root, text="Is write files names", variable=self.is_write_names).grid(row=4, column=2, sticky="w")

        # Run button
        tk.Button(root, text="Run Copy Process", command=self.run_copy_process).grid(row=8, column=1, pady=10)
        
        # Cancel button to stop the process
        self.stop_processing = False
        tk.Button(root, text="Cancel", command=self.cancel_copy_process).grid(row=8, column=2, pady=10)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.grid(row=9, column=0, columnspan=3, padx=10, pady=10)
        
        self.progress_label = tk.Label(root, text="0/0")
        self.progress_label.grid(row=10, column=0, columnspan=3, pady=(0, 10))

        # Text box for logging process output
        self.log_text = tk.Text(root, width=80, height=20)
        self.log_text.grid(row=11, column=0, columnspan=3, padx=10, pady=10)

    def log_message(self, message):
        """Log message to the text box."""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)  # Auto-scroll to the end
        self.root.update()  # Update the UI

    def select_source_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.source_dir.set(directory)

    def select_dest_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.dest_dir.set(directory)

    def select_hash_table_path(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            self.hash_table_path.set(file_path)
            
    def cancel_copy_process(self):
        self.stop_processing = True 

    def run_copy_process(self):
        source = self.source_dir.get()
        dest = self.dest_dir.get()

        # Validation for directory selection
        if not source or not dest:
            messagebox.showerror("Error", "Please select both source and destination directories.")
            return

        copied_hashes = {}
        hash_table_path = self.hash_table_path.get()
        try:
            with open(hash_table_path, 'r') as file:
                copied_hashes = json.load(file)
        except:
            self.log_message("Не удалось прочитать hash_table из JSON файла")

        # Call the method with selected options
        search = SearchOscillograms()

        # Initialize the progress bar with the total file count
        total_files = sum([len(files) for r, d, files in os.walk(source)])
        self.progress_bar["maximum"] = total_files
        self.progress_bar["value"] = 0

        # Wrap `copy_new_oscillograms` to capture intermediate logs
        def progress_callback(message, step=1):
            self.log_message(message)
            self.progress_bar["value"] += step
            self.progress_label.config(text=f"{int(self.progress_bar['value'])}/{total_files}")
            self.root.update_idletasks()  # Refresh the progress bar and label

        copied_files = search.copy_new_oscillograms(
            source_dir=source,
            dest_dir=dest,
            copied_hashes=copied_hashes,
            use_comtrade=self.use_comtrade.get(),
            use_new_comtrade=self.use_new_comtrade.get(),
            use_brs=self.use_brs.get(),
            use_neva=self.use_neva.get(),
            use_ekra=self.use_ekra.get(),
            use_parma=self.use_parma.get(),
            use_black_box=self.use_black_box.get(),
            use_res_3=self.use_res_3.get(),
            use_osc=self.use_osc.get(),
            preserve_dir_structure=self.preserve_dir_structure.get(),
            is_write_names=self.is_write_names, # TODO: self.is_write_names.get()
            progress_callback=progress_callback,
            stop_processing_fn=lambda: self.stop_processing
        )

        # Show result
        messagebox.showinfo("Process Complete", f"Total files copied: {copied_files}")
        self.stop_processing = False  # Reset cancel flag
        self.progress_bar["value"] = 0  # Reset progress bar

# Tkinter main loop
if __name__ == "__main__":
    root = tk.Tk()
    app = OscillogramApp(root)
    root.mainloop()