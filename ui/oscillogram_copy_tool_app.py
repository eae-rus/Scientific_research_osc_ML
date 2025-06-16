import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from filesystem.finder import OscillogramFinder, TYPE_OSC # MODIFIED IMPORT
import json
from typing import Dict # For type hint

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
        self.is_write_names_var = tk.BooleanVar(value=False) # ADDED tk.BooleanVar


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
        tk.Checkbutton(root, text="Log Copied File Names", variable=self.is_write_names_var).grid(row=4, column=2, sticky="w") # ADDED Checkbutton

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

        # Instantiate OscillogramFinder
        finder = OscillogramFinder(is_print_message=False) # GUI handles messages

        # Prepare file_type_flags dictionary
        file_type_flags: Dict[TYPE_OSC, bool] = {
            TYPE_OSC.COMTRADE_CFG_DAT: self.use_comtrade.get(),
            TYPE_OSC.COMTRADE_CFF: self.use_new_comtrade.get(), # Assuming "New Comtrade" means CFF
            TYPE_OSC.BRESLER: self.use_brs.get(),
            TYPE_OSC.EVA: self.use_neva.get(), # Assuming "NEVA" means EVA
            TYPE_OSC.EKRA: self.use_ekra.get(),
            # Assuming self.use_parma covers both PARMA and PARMA_TO if they were distinct in enum
            # For simplicity, if PARMA in enum is one type, map it. If two, map both to self.use_parma.get()
            # This depends on the exact members of TYPE_OSC. Assuming specific PARMA types exist in TYPE_OSC:
            # TYPE_OSC.PARMA: self.use_parma.get(),
            # TYPE_OSC.PARMA_TO: self.use_parma.get(), # Or separate checkboxes if needed
            # For now, let's assume a generic PARMA if defined, or skip if not in TYPE_OSC
            # The original SearchOscillograms had a single use_parma flag.
            # We need to map GUI checkboxes to all relevant TYPE_OSC members.
            # This example maps based on common assumptions.
            # Add specific TYPE_OSC members as they are defined in your finder.py
        }
        # Add other types based on your TYPE_OSC enum and GUI checkboxes
        # Example: if TYPE_OSC.PARMA exists:
        if hasattr(TYPE_OSC, 'PARMA'):
             file_type_flags[TYPE_OSC.PARMA] = self.use_parma.get()
        if hasattr(TYPE_OSC, 'PARMA_TO'): # If there's a separate PARMA_TO
             file_type_flags[TYPE_OSC.PARMA_TO] = self.use_parma.get() # Mapped to same GUI flag
        if hasattr(TYPE_OSC, 'BLACK_BOX'): # Assuming this maps to a type like KRUG or NIIM if they are black boxes
             file_type_flags[TYPE_OSC.BLACK_BOX] = self.use_black_box.get() # Or map to specific types
        if hasattr(TYPE_OSC, 'RES_3'): # Example mapping
             file_type_flags[TYPE_OSC.RES_3] = self.use_res_3.get()
        if hasattr(TYPE_OSC, 'OSC'): # Generic .osc, map if TYPE_OSC has it
             file_type_flags[TYPE_OSC.OSC] = self.use_osc.get()

        # For archive processing, these are implicitly handled by OscillogramFinder if corresponding archive types are enabled.
        # The file_type_flags above control what's extracted *from* them or copied directly.
        # Add archive types to flags if you want explicit control over processing archives themselves:
        if hasattr(TYPE_OSC, 'ARCH_7Z'): file_type_flags[TYPE_OSC.ARCH_7Z] = True # Always try to process archives
        if hasattr(TYPE_OSC, 'ARCH_ZIP'): file_type_flags[TYPE_OSC.ARCH_ZIP] = True
        if hasattr(TYPE_OSC, 'ARCH_RAR'): file_type_flags[TYPE_OSC.ARCH_RAR] = True


        # Initialize the progress bar
        total_files = 0
        try:
            for r, d, files in os.walk(source): # Recalculate total files carefully
                total_files += len(files)
        except Exception as e:
            self.log_message(f"Error counting files: {e}")
            messagebox.showerror("Error", f"Could not access source directory to count files: {e}")
            return

        self.progress_bar["maximum"] = total_files if total_files > 0 else 1 # Avoid zero maximum
        self.progress_bar["value"] = 0
        self.progress_label.config(text=f"0/{total_files}")
        self.root.update_idletasks()

        def gui_progress_callback(message, step=1): # Renamed for clarity
            if self.stop_processing: # Check stop flag early
                raise InterruptedError("Processing cancelled by user.")
            self.log_message(message) # Log message from finder
            # Finder's pbar update is internal. GUI progress bar is based on overall file count.
            # We need a way for finder to report overall progress step for GUI.
            # For now, let's assume finder's internal tqdm handles detailed progress,
            # and GUI progress bar might be updated less frequently or after finder completes.
            # Or, finder's progress_callback could be designed to provide overall progress.
            # The original `search.copy_new_oscillograms` called its progress_callback per file.
            # Let's assume finder.copy_new_oscillograms does the same.
            current_val = self.progress_bar["value"] + step
            self.progress_bar["value"] = current_val if current_val <= total_files else total_files
            self.progress_label.config(text=f"{int(self.progress_bar['value'])}/{total_files}")
            self.root.update_idletasks()

        try:
            copied_files = finder.copy_new_oscillograms(
                source_dir=source,
                dest_dir=dest,
                copied_hashes_input=copied_hashes, # Pass as input
                preserve_dir_structure=self.preserve_dir_structure.get(),
                use_hashes=True, # Assuming GUI always wants to use hash logic
                file_type_flags=file_type_flags,
                # max_archive_depth can be taken from config or a new GUI element
                progress_callback=gui_progress_callback,
                stop_processing_fn=lambda: self.stop_processing,
                is_write_names_fn=self.is_write_names_var.get() # Pass boolean value
            )
            messagebox.showinfo("Process Complete", f"Total new files/pairs copied: {copied_files}")
        except InterruptedError:
            messagebox.showinfo("Cancelled", "Processing was cancelled by the user.")
            self.log_message("Processing cancelled by user.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            self.log_message(f"An error occurred: {e}")

        self.stop_processing = False  # Reset cancel flag
        self.progress_bar["value"] = 0  # Reset progress bar

# Tkinter main loop
if __name__ == "__main__":
    root = tk.Tk()
    app = OscillogramApp(root)
    root.mainloop()