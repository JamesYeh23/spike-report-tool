import spikeinterface.full as si
import probeinterface as pi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import shutil
from pathlib import Path
import os

def generate_report(results_folder, binary_path, probe_path, output_folder, 
                    sampling_freq=40000, num_channels=64, dtype='int16'):
    
    results_folder = Path(results_folder)
    binary_path = Path(binary_path)
    # We ignore probe_path to prevent mapping bugs, using a dummy linear probe instead.
    output_folder = Path(output_folder)
    
    # 1. Setup
    folder_name = results_folder.name
    parent_name = results_folder.parent.name
    cache_folder = output_folder / "cache" / f"{parent_name}_{folder_name}"
    
    print(f"--- SpikeReport ---")
    print(f"Processing: {results_folder}")

    # 2. Load Recording & Filter
    print("Loading recording...")
    recording = si.read_binary(
        binary_path,
        sampling_frequency=sampling_freq,
        dtype=dtype,
        num_channels=num_channels
    )
    # Bandpass filter is crucial for SNR and Waveform shape
    print("Applying bandpass filter (300-6000Hz)...")
    recording = si.bandpass_filter(recording, freq_min=300, freq_max=6000)

    # Use Dummy Linear Probe to guarantee correct channel mapping
    print("Setting linear probe...")
    linear_probe = pi.generate_linear_probe(num_elec=num_channels)
    linear_probe.set_device_channel_indices(np.arange(num_channels))
    recording = recording.set_probe(linear_probe)

    # 3. Load Sorting
    print("Loading sorting data...")
    sorting = si.read_phy(results_folder)
    tsv_path = results_folder / "cluster_info.tsv"
    
    unit_best_ch = {}
    sorting_good = sorting

    if tsv_path.exists():
        try:
            df = pd.read_csv(tsv_path, sep='\t')
            # Detect label column
            if 'group' in df.columns: label_col = 'group'
            elif 'KSLabel' in df.columns: label_col = 'KSLabel'
            else: label_col = None

            if label_col:
                # Filter 'good' units
                good_df = df[df[label_col] == 'good']
                
                # Create map: unit_id -> best_channel (from Kilosort)
                for _, row in good_df.iterrows():
                    unit_best_ch[str(row['cluster_id'])] = int(row['ch'])
                
                # Select units
                good_ids = good_df['cluster_id'].values.astype(str)
                
                # Handle potential int/str mismatch in sorting object
                if len(sorting.unit_ids) > 0 and isinstance(sorting.unit_ids[0], (int, np.integer)):
                    valid_ids = [int(u) for u in good_ids if int(u) in sorting.unit_ids]
                else:
                    valid_ids = [u for u in good_ids if u in sorting.unit_ids]

                if len(valid_ids) > 0:
                    sorting_good = sorting.select_units(valid_ids)
                    print(f"Selected {len(valid_ids)} manual 'good' units.")
        except Exception as e:
            print(f"Error reading TSV: {e}. Using all units.")
    else:
        print("No cluster_info.tsv found.")

    # 4. Create Analyzer
    if cache_folder.exists():
        try: shutil.rmtree(cache_folder)
        except: pass

    print("Running SortingAnalyzer...")
    analyzer = si.create_sorting_analyzer(
        sorting_good, recording, format="binary_folder",
        folder=cache_folder, overwrite=True, sparse=False
    )
    
    analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
    analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0)
    analyzer.compute("templates")
    analyzer.compute("noise_levels")
    
    # --- FIX: Re-enable PCA for Isolation Distance ---
    print("Computing PCA (needed for Isolation Distance)...")
    analyzer.compute("principal_components", n_components=5, mode='by_channel_local')

    # 5. Compute Metrics
    print("Calculating metrics (SNR, ISI, Isolation Distance)...")
    # Added 'isolation_distance' back to the list
    metrics = analyzer.compute("quality_metrics", 
                               metric_names=['snr', 'isi_violation', 'isolation_distance'],
                               skip_pc_metrics=False).get_data()
    
    metrics.index.name = 'Unit_ID'
    metrics = metrics.reset_index()
    metrics['Unit_ID'] = metrics['Unit_ID'].astype(str)

    if 'isi_violations_ratio' in metrics.columns:
        metrics = metrics.rename(columns={'isi_violations_ratio': 'isi_violation'})

    # 6. Merge Data
    channel_data = []
    templates_ext = analyzer.get_extension("templates")
    
    for uid in sorting_good.unit_ids:
        str_uid = str(uid)
        # Use channel from Kilosort file if available, else calculate
        if str_uid in unit_best_ch:
            best_ch = unit_best_ch[str_uid]
        else:
            template = templates_ext.get_unit_template(uid)
            best_ch = np.argmax(np.ptp(template, axis=0))
            
        channel_data.append({'Unit_ID': str_uid, 'Best_Channel': best_ch})

    df_ch = pd.DataFrame(channel_data)
    final_df = pd.merge(metrics, df_ch, on='Unit_ID')
    
    final_df['Display_Ch'] = final_df['Best_Channel'] 
    
    cols = ['Unit_ID', 'Display_Ch', 'snr', 'isi_violation', 'isolation_distance']
    final_cols = [c for c in cols if c in final_df.columns]
    final_df = final_df[final_cols]

    # 7. Generate PDF
    pdf_filename = output_folder / f"Report_{parent_name}_{folder_name}.pdf"
    print(f"Generating PDF: {pdf_filename}")

    with PdfPages(pdf_filename) as pdf:
        # Table Page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.set_title(f"Quality Report: {folder_name}", fontsize=16, fontweight='bold')
        table_data = [final_df.columns.values.tolist()] + final_df.round(3).values.tolist()
        table = ax.table(cellText=table_data, colLabels=None, loc='center', cellLoc='center')
        table.scale(1, 1.5)
        pdf.savefig(fig); plt.close()

        # Waveform Pages
        units = sorting_good.unit_ids
        per_page = 6
        pages = int(np.ceil(len(units) / per_page))

        for page in range(pages):
            fig, axes = plt.subplots(3, 2, figsize=(8.5, 11))
            axes = axes.flatten()
            batch = units[page*per_page : (page+1)*per_page]

            for i, uid in enumerate(batch):
                ax = axes[i]
                row = final_df[final_df['Unit_ID'] == str(uid)].iloc[0]
                ch_idx = int(row['Display_Ch']) 
                
                tmpl = templates_ext.get_unit_template(uid)
                # Handle cases where ch_idx might be out of bounds (unlikely with sparse=False)
                if ch_idx < tmpl.shape[1]:
                    wave = tmpl[:, ch_idx]
                else:
                    wave = tmpl[:, 0]

                # Prepare title with safe fallback for missing metrics
                iso = row.get('isolation_distance', np.nan)
                title_str = (f"Unit {uid} (Ch {ch_idx})\n"
                             f"SNR:{row['snr']:.1f}  ISI:{row['isi_violation']:.3f}  IsoDist:{iso:.1f}")
                
                ax.plot(wave, 'k', lw=1.5)
                ax.set_title(title_str, fontsize=10)
                ax.axis('off')
                ax.axhline(0, color='r', ls='--', lw=0.5, alpha=0.5)

            for j in range(len(batch), len(axes)): axes[j].axis('off')
            plt.tight_layout()
            pdf.savefig(fig); plt.close()

    print("Done!")