import logging
import os
import time
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

from my_pipeline.processing.constants import (BAND_NAMES, STANDARD_EEG_CHANNELS)

logger = logging.getLogger(__name__)

def create_graphs(aggregate_results, property_name):
    # aggregate_results[seizure/non_seizure][channel_pair][band_pair][property_name] = value
    
    nodes = [(ch, band) for ch in range(len(STANDARD_EEG_CHANNELS)) for band in BAND_NAMES]
    seizure_graph = nx.Graph()
    non_seizure_graph = nx.Graph()
    for node in nodes:
        seizure_graph.add_node(node)
        non_seizure_graph.add_node(node)
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i >= j:
                continue
            ch1, band1 = node1
            ch2, band2 = node2
            if ch1 == ch2:
                continue
            channel_pair = (ch1, ch2)
            band_pair = (band1, band2)
            
            # Robustness: check both orders
            if channel_pair not in aggregate_results['seizure']:
                channel_pair = (ch2, ch1)
            if band_pair not in aggregate_results['seizure'][channel_pair]:
                band_pair = (band2, band1)
            
            seizure_mean = aggregate_results['seizure'][channel_pair][band_pair][property_name]
            non_seizure_mean = aggregate_results['non_seizure'][channel_pair][band_pair][property_name]

            seizure_graph.add_edge(node1, node2, weight=seizure_mean)
            non_seizure_graph.add_edge(node1, node2, weight=non_seizure_mean)
    logging.info(f"Seizure graph: {seizure_graph.number_of_nodes()} nodes, {seizure_graph.number_of_edges()} edges")
    logging.info(f"Non-seizure graph: {non_seizure_graph.number_of_nodes()} nodes, {non_seizure_graph.number_of_edges()} edges")
    return seizure_graph, non_seizure_graph

def apply_quantile_threshold(G, Q, weight_attr='weight'):
    if Q == 0:
        return G.copy()
    weights = [d[weight_attr] for u, v, d in G.edges(data=True) if weight_attr in d]
    if not weights:
        return G.copy()
    threshold_value = np.quantile(weights, Q)
    G_thresh = G.copy()
    edges_to_remove = [
        (u, v) for u, v, data in G_thresh.edges(data=True)
        if data.get(weight_attr, 0) < threshold_value
    ]
    G_thresh.remove_edges_from(edges_to_remove)
    return G_thresh

def invert_weights_as_distance_eq9(G, weight_attr='weight', dist_attr='distance'):
    G_dist = G.copy()
    weights = [
        d[weight_attr] for u, v, d in G.edges(data=True)
        if weight_attr in d and np.isfinite(d[weight_attr])
    ]
    if not weights:
        return G_dist
    max_W = np.max(weights)
    min_W = np.min(weights)
    edges_to_remove = []
    for u, v, data in G_dist.edges(data=True):
        w_ij = data.get(weight_attr, 0)
        dist = (max_W + min_W) - w_ij
        if not np.isfinite(dist) or dist <= 0:
            data[dist_attr] = float('inf')
            edges_to_remove.append((u, v))
        else:
            data[dist_attr] = dist
    G_dist.remove_edges_from(edges_to_remove)
    return G_dist

def get_edges_of_type_c(G, c):
    band_a, band_b = c
    edges_c = []
    for u, v in G.edges():
        band_u = G.nodes[u].get('band', u[1])
        band_v = G.nodes[v].get('band', v[1])
        if (band_u == band_a and band_v == band_b) or (band_u == band_b and band_v == band_a):
            edges_c.append((u, v))
    return edges_c

def get_all_coupling_types(G):
    types_set = set()
    for u, v in G.edges():
        band_u = G.nodes[u].get('band', u[1])
        band_v = G.nodes[v].get('band', v[1])
        c = tuple(sorted((band_u, band_v)))
        types_set.add(c)
    return list(types_set)

def safe_log_transform(data):
    data_array = np.array(data)
    if np.any(data_array < 0):
        logging.warning(f"Negative data detected {data_array}, log transform will produce NaN.")
    return np.log1p(data_array)

def calculate_global_efficiency(G, dist_attr='distance'):
    total_efficiency_sum = 0.0
    N = G.number_of_nodes()
    if N < 2: return 0.0
    all_paths_iter = nx.all_pairs_dijkstra_path_length(G, weight=dist_attr)
    for source, targets in all_paths_iter:
        for target, d_ij in targets.items():
            if source != target:
                total_efficiency_sum += 1.0 / d_ij
    return total_efficiency_sum / (N * (N - 1))

def calculate_local_efficiency(G, dist_attr='distance'):
    total_local_efficiency = 0.0
    N = G.number_of_nodes()
    if N == 0: return 0.0
    for i in G.nodes():
        neighbors_i = list(G.neighbors(i))
        G_i = G.subgraph(neighbors_i)
        if G_i.number_of_nodes() > 1:
            E_G_i = calculate_global_efficiency(G_i, dist_attr=dist_attr)
            total_local_efficiency += E_G_i
    return total_local_efficiency / N

def calculate_coupling_specific_betweenness(G_dist, c, dist_attr='distance'):
    edge_betweenness = nx.edge_betweenness_centrality(G_dist, weight=dist_attr, normalized=False)
    edges_c = get_edges_of_type_c(G_dist, c)
    E_c = len(edges_c)
    if E_c == 0: return 0.0
    total_cbw = 0.0
    for u, v in edges_c:
        if (u, v) in edge_betweenness:
            total_cbw += edge_betweenness[(u, v)]
        elif (v, u) in edge_betweenness:
            total_cbw += edge_betweenness[(v, u)]
    return total_cbw / E_c

def calculate_vulnerability(G_dist, c, dist_attr='distance'):
    E_G_full = calculate_global_efficiency(G_dist, dist_attr=dist_attr)
    E_L_full = calculate_local_efficiency(G_dist, dist_attr=dist_attr)
    V_G = 0.0
    V_L = 0.0
    edges_to_remove = get_edges_of_type_c(G_dist, c)
    G_c = G_dist.copy()
    G_c.remove_edges_from(edges_to_remove)
    E_G_perturbed = calculate_global_efficiency(G_c, dist_attr=dist_attr)
    E_L_perturbed = calculate_local_efficiency(G_c, dist_attr=dist_attr)
    if E_G_full > 0:
        V_G = 1.0 - (E_G_perturbed / E_G_full)
    if E_L_full > 0:
        V_L = 1.0 - (E_L_perturbed / E_L_full)
    return V_G, V_L

def run_multi_threshold_analysis(seizure_graph, non_seizure_graph, Q_values):
    all_c_types = get_all_coupling_types(seizure_graph)
    logging.info(f"Found {len(all_c_types)} coupling types.")
    all_results_list = []
    logging.info(f"Starting multi-threshold analysis for {len(Q_values)} thresholds.")
    total_start_time = time.time()
    for Q in Q_values:
        loop_start_time = time.time()
        logging.info(f"Processing Threshold Q = {Q:.2f}")
        G_s_thresh = apply_quantile_threshold(seizure_graph, Q, weight_attr='weight')
        G_n_thresh = apply_quantile_threshold(non_seizure_graph, Q, weight_attr='weight')
        logging.info(f"G_s edges: {G_s_thresh.number_of_edges()}, G_n edges: {G_n_thresh.number_of_edges()}")
        G_s_dist = invert_weights_as_distance_eq9(G_s_thresh, weight_attr='weight')
        G_n_dist = invert_weights_as_distance_eq9(G_n_thresh, weight_attr='weight')
        for c in all_c_types:
            c_name = f"{c[0]}-{c[1]}"
            V_G_s, V_L_s = calculate_vulnerability(G_s_dist, c)
            CBW_s = calculate_coupling_specific_betweenness(G_s_dist, c)
            all_results_list.append({
                'Q': Q,
                'Graph': 'Seizure',
                'CouplingType': c_name,
                'VG_log': safe_log_transform(V_G_s),
                'VL': V_L_s,
                'CBW_log': safe_log_transform(CBW_s)
            })
            V_G_n, V_L_n = calculate_vulnerability(G_n_dist, c)
            CBW_n = calculate_coupling_specific_betweenness(G_n_dist, c)
            all_results_list.append({
                'Q': Q,
                'Graph': 'Non-Seizure',
                'CouplingType': c_name,
                'VG_log': safe_log_transform(V_G_n),
                'VL': V_L_n,
                'CBW_log': safe_log_transform(CBW_n)
            })
        loop_end_time = time.time()
        logging.info(f"Threshold Q={Q:.2f} completed in {(loop_end_time - loop_start_time):.2f} seconds.")
    total_end_time = time.time()
    logging.info(f"Multi-threshold analysis complete! Total execution time: {(total_end_time - total_start_time) / 60:.2f} minutes.")
    df_results = pd.DataFrame(all_results_list)
    return df_results

def plot_metrics_across_thresholds(df_results, metric_name='VG_log', save_file_path=None):
    logging.info(f"Plotting {metric_name} across all thresholds...")
    g = sns.relplot(
        data=df_results,
        x='Q',
        y=metric_name,
        hue='Graph',
        col='CouplingType',
        kind='line',
        col_wrap=5,
        palette={'Seizure': 'red', 'Non-Seizure': 'blue'},
        height=4,
        aspect=1.2
    )
    g.fig.suptitle(f'Metric "{metric_name}" vs. Threshold (Q)', y=1.03, fontsize=16)
    g.set_axis_labels('Quantile Threshold (Q)', metric_name)
    g.set_titles("{col_name}")
    plt.tight_layout()
    if save_file_path:
        output_file_path = os.path.join(save_file_path, f'plot_{metric_name}_across_thresholds.png')
        plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Plot saved to {output_file_path}")
    else:
        plt.show()

def plot_metrics_at_single_threshold(df_results, Q_value, metric_name='VG_log', save_file_path=None):
    logging.info(f"Plotting {metric_name} at single threshold Q={Q_value}...")
    df_filtered = df_results[df_results['Q'] == Q_value]
    if df_filtered.empty:
        logging.warning(f"No data found for Q = {Q_value}. Did you run that threshold?")
        return
    plt.figure(figsize=(18, 7))
    ax = sns.barplot(
        data=df_filtered,
        x='CouplingType',
        y=metric_name,
        hue='Graph',
        palette={'Seizure': 'red', 'Non-Seizure': 'blue'}
    )
    ax.set_title(f'Metric "{metric_name}" Comparison at Q = {Q_value}', fontsize=16)
    ax.set_ylabel(metric_name)
    ax.set_xlabel('Coupling Type')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.legend(title='Graph')
    plt.tight_layout()
    if save_file_path:
        output_file_path = os.path.join(save_file_path, f'plot_{metric_name}_at_Q_{Q_value}.png')
        plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Plot saved to {output_file_path}")
    else:
        plt.show()

def run(aggregate_results, seg_metadatas, visualization_save_path, property_names):
    logger.info("Running network analysis...")
    
    # Assuming all segments belong to the same patient for naming
    patient_name = seg_metadatas[0].patient
    
    for property_name in property_names:
        seizure_graph, non_seizure_graph = create_graphs(aggregate_results, property_name)
    
        Q_values = np.arange(0.0, 1.0, 0.05)[::-1]
        logging.info(f"Defined Q values for analysis: {Q_values}")
        all_results_df = run_multi_threshold_analysis(seizure_graph, non_seizure_graph, Q_values)
        
        # save results
        results_file_path = os.path.join(visualization_save_path, patient_name, f'network_analysis_results_{property_name}.csv')
        os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
        all_results_df.to_csv(results_file_path, index=False)
        logging.info(f"Network analysis results for {patient_name} saved to {results_file_path}")
        
        visualization_save_path = os.path.join(visualization_save_path, patient_name)
        os.makedirs(visualization_save_path, exist_ok=True)
        plot_metrics_across_thresholds(all_results_df, metric_name='VG_log', save_file_path=visualization_save_path)
        plot_metrics_across_thresholds(all_results_df, metric_name='VL', save_file_path=visualization_save_path)
        return