import pandas as pd

# Load your original CSV file
# Replace "your_file.csv" with the path to your actual file
df = pd.read_csv("your_file.csv")

# Define the essential features to keep
selected_columns = [
    # identifiers
    "subject_id", "start_timestamp", "end_timestamp",

    # Time-domain
    "Magnitude_mean", "Magnitude_std_dev", "Magnitude_rms", "Magnitude_energy",
    "PC1_mean", "PC1_std_dev", "PC1_rms", "PC1_energy",

    # Tremor rhythm indicators
    "Magnitude_peaks_rt", "Magnitude_zero_cross_rt", "Magnitude_ssc_rt",
    "PC1_peaks_rt", "PC1_zero_cross_rt", "PC1_ssc_rt",

    # Frequency-domain
    "Magnitude_fft_dom_freq", "Magnitude_fft_tot_power", "Magnitude_fft_energy", "Magnitude_fft_entropy",
    "PC1_fft_dom_freq", "PC1_fft_tot_power", "PC1_fft_energy", "PC1_fft_entropy",

    # Complexity
    "Magnitude_sampen", "Magnitude_dfa", "PC1_sampen", "PC1_dfa",

    # Tremor labels/features
    "Rest_tremor", "Postural_tremor", "Kinetic_tremor"
]

# Keep only the available columns (some files may miss a few)
available_cols = [c for c in selected_columns if c in df.columns]
df_selected = df[available_cols]

# Save to new Excel file
df_selected.to_excel("parkinson_selected_features.xlsx", index=False)

print("✅ New Excel file saved as 'parkinson_selected_features.xlsx' with", len(available_cols), "columns.")
