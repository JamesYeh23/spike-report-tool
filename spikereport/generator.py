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
    """
    Generates a PDF quality report for Kilosort results.
    """
    
    results_folder = Path(results_folder)
    binary_path = Path(binary_path)
    probe_path = Path(probe_path)
    output_folder = Path(output_folder)
    
    # 1. Setup Cache Path (Unique per run)
    folder_name = results_folder.name
    parent_name = results_folder.parent.name
    cache_folder = output_folder / "cache" / f"{parent_name}_{folder_name}"
    
    print(f"--- SpikeReport ---")
    print(f"Processing: {results_folder}")
    print(f"Cache: {cache_folder}")

    # 2. Load Recording & Probe
    print("Loading recording...")
    recording = si.read_binary(
        binary_path,
        sampling_frequency=sampling_freq,
        dtype=dtype,
        num_channels=num_channels
    )

    probe_group = pi.read_prb(probe_path)
    probe = probe_group.probes[0]  
    recording = recording.set_probe(probe)
    recording.annotate(is_filtered=True) 

    # 3. Load Sorting & Filter 'Good' Units
    print("Loading sorting data...")
    sorting = si.read_phy(results_folder)
    tsv_path = results_folder / "cluster_group.tsv"

    if tsv_path.exists():
        df_groups = pd.read_csv(tsv_path, sep='\t')
        good_ids = df_groups[df_groups['group'] == 'good']['cluster_id'].astype(str).values
        
        # Handle int/str mismatch
        if len(sorting.unit_ids) > 0 and isinstance(sorting.unit_ids[0], (int, np.integer)):
             good_ids = df_groups[df_groups['group'] == 'good']['cluster_id'].values
                
        valid_ids = [uid for uid in good_ids if uid in sorting.unit_ids]
        if len(valid_ids) > 0:
            sorting_good = sorting.select_units(valid_ids)
            print(f"Selected {len(valid_ids)} manual 'good' units.")
        else:
            print("Warning: TSV IDs don't match. Using all units.")
            sorting_good = sorting
    else:
        print("No cluster_group.tsv found. Using all units.")
        sorting_good = sorting

    # 4. Create Analyzer
    if cache_folder.exists():
        try: shutil.rmtree(cache_folder)
        except: pass

    print("Running SortingAnalyzer (this may take a moment)...")
    analyzer = si.create_sorting_analyzer(
        sorting_good, recording, format="binary_folder",
        folder=cache_folder, overwrite=True, sparse=True
    )
    
    analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
    analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0)
    analyzer.compute("templates")
    analyzer.compute("noise_levels")
    analyzer.compute("principal_components", n_components=5, mode='by_channel_local')

    # 5. Compute Metrics
    print("Calculating quality metrics...")
    metrics = analyzer.compute("quality_metrics", metric_names=['snr', 'isi_violation', 'isolation_distance']).get_data()
    if 'isi_violations_ratio' in metrics.columns:
        metrics = metrics.rename(columns={'isi_violations_ratio': 'isi_violation'})

    # 6. Find Best Channels
    templates_ext = analyzer.get_extension("templates")
    ext_channels = {}
    for unit_id in sorting_good.unit_ids:
        template = templates_ext.get_unit_template(unit_id)
        peak_to_peak = np.ptp(template, axis=0)
        ext_channels[unit_id] = np.argmax(peak_to_peak)

    # 7. Merge Data
    channel_info = [{'Unit_ID': u, 'Best_Channel': c + 1} for u, c in ext_channels.items()]
    df_channels = pd.DataFrame(channel_info)
    
    metrics = metrics.reset_index()
    metrics['Unit_ID'] = metrics['Unit_ID'].astype(str)
    df_channels['Unit_ID'] = df_channels['Unit_ID'].astype(str) # For merging
    
    final_df = pd.merge(metrics, df_channels, on='Unit_ID')
    cols = ['Unit_ID', 'Best_Channel', 'snr', 'isi_violation', 'isolation_distance']
    final_df = final_df[[c for c in cols if c in final_df.columns]]

    # 8. Generate PDF
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
                tmpl = templates_ext.get_unit_template(uid)
                ch_idx = ext_channels[uid]
                wave = tmpl[:, ch_idx]
                
                # Get metrics
                row = final_df[final_df['Unit_ID'] == str(uid)].iloc[0]
                
                ax.plot(wave, 'k', lw=1.5)
                ax.set_title(f"Unit {uid} (Ch {row['Best_Channel']})\nSNR:{row['snr']:.1f} ISI:{row['isi_violation']:.2f}", fontsize=10)
                ax.axis('off')
                ax.axhline(0, color='r', ls='--', lw=0.5, alpha=0.5)

            for j in range(len(batch), len(axes)): axes[j].axis('off')
            plt.tight_layout()
            pdf.savefig(fig); plt.close()

    print("Done!")