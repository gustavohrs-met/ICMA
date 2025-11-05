import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats 
import matplotlib.pyplot as plt 
import time
from PIL import Image
# --- IMPORT PLOTLY ---
import plotly.express as px 
from plotly.subplots import make_subplots
import plotly.graph_objects as go # Required for advanced manipulation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns # Required for heatmap (matplotlib/seaborn)

### --- NEW IMPORTS (for PLS-DA) --- ###
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
# ----------------------------------------------

# --- ⚠️ MANDATORY FIX FOR LARGE GRAPHS ---
Image.MAX_IMAGE_PIXELS = 200000000 

# --- SESSION STATE VARIABLES ---
if 'data_uploaded' not in st.session_state:
    st.session_state['data_uploaded'] = False
if 'data_processed' not in st.session_state:
    st.session_state['data_processed'] = False
if 'groups_loaded' not in st.session_state:
    st.session_state['groups_loaded'] = False # For Mapping control

# --- PAGE CONFIGURATION (REQUEST 1) ---
st.set_page_config(
    page_title="In vitro NMR-Metabolomics Analysis", # Shortened for tab
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# --- MAIN TITLE AND ANALYSIS SELECTION (REQUEST 1) ---
# ----------------------------------------------------
st.title("🔬 In vitro Cellular NMR-Metabolomics Analysis Platform") # Translated
st.markdown("Welcome! This platform is designed to analyze cellular metabolomics (NMR) data.") 

analysis_type = st.radio(
    "Select the analysis type:", 
    ["Exometabolome", "Endometabolome"],
    horizontal=True,
    captions=["Analysis of the culture medium", "Analysis of the cell extract (Coming soon!)"] 
)
st.markdown("---")

# --- ANALYSIS SELECTION LOGIC ---

if analysis_type == "Endometabolome":
    st.info("Endometabolome analysis is under development and will be added soon!") 
    st.stop()

# --- ALL EXOMETABOLOME CODE GOES INSIDE HERE ---
if analysis_type == "Exometabolome":

    # ----------------------------------------------------
    # --- GLOBAL ANALYSIS FUNCTIONS ---
    # ----------------------------------------------------

    def classify_metabolite(level):
        """Classifies the metabolite profile based on the % Pure Medium Level."""
        if level < 95:
            return 'Consumed'
        elif level > 105:
            return 'Excreted'
        else:
            return 'Unaltered'

    def validate_and_clean_data(analytes_df):
        """Validates and cleans data AFTER normalization and transformation."""
        
        st.subheader("🔍 Post-processing Data Validation and Quality") 
        
        missing_summary = analytes_df.isnull().sum()
        total_missing = missing_summary.sum()
        
        if total_missing > 0:
            st.warning(f"Found {total_missing} 'NaN' (Not a Number) values post-processing.") 
            st.info("This can occur if an entire column was zero before Log or Scaling transformation. These values will be filled with 0.") 
            analytes_df = analytes_df.fillna(0)
            
        inf_summary = (analytes_df == np.inf).sum() + (analytes_df == -np.inf).sum()
        total_inf = inf_summary.sum()
        if total_inf > 0:
            st.warning(f"Found {total_inf} 'Infinite' values post-processing (e.g., Log(0)).") 
            st.info("Infinite values will be replaced with 0.") 
            analytes_df = analytes_df.replace([np.inf, -np.inf], 0)

        initial_columns = len(analytes_df.columns)
        analytes_df = analytes_df.loc[:, (analytes_df != 0).any(axis=0)]
        final_columns = len(analytes_df.columns)
        
        if initial_columns != final_columns:
            st.info(f"Removed {initial_columns - final_columns} metabolites with all zero or constant values.") 
        
        return analytes_df

    def calculate_descriptive_stats(analytes_df, metadata_df):
        """Calculates Mean and Standard Deviation per Group."""
        
        data_with_group = analytes_df.join(metadata_df)
        mean_df = data_with_group.groupby('Grupo').mean(numeric_only=True).T.rename(columns=lambda x: f'{x} (Mean)')
        sd_df = data_with_group.groupby('Grupo').std(numeric_only=True).T.rename(columns=lambda x: f'{x} (SD)')
        groups = sorted(metadata_df['Grupo'].unique())
        final_cols = []
        for group in groups:
            if f'{group} (Mean)' in mean_df.columns:
                final_cols.extend([f'{group} (Mean)', f'{group} (SD)'])
        stats_df = pd.concat([mean_df, sd_df], axis=1)
        stats_df = stats_df.reindex(columns=final_cols)
        return stats_df

    def style_comparison_table(df):
        """Applies conditional formatting (background color and highlight) and CENTER aligns text in the comparison table."""
        
        is_multiindex = isinstance(df.columns, pd.MultiIndex)
        bg_color_var_pct = '#E6D8E6'
        bg_color_var_fc = '#FFFACD'
        bg_color_p_value = '#ffebeb'
        bg_color_hedges_g = '#F0FFF0'
        
        bg_map_display = {
            'Variation %': bg_color_var_pct,      
            'Uncertainty (%)': bg_color_var_pct,  
            'Fold Change': bg_color_var_fc,         
            'Uncertainty (FC)': bg_color_var_fc,           
            'P-value': bg_color_p_value,
            "Effect Size (Hedges' g)": bg_color_hedges_g, 
        }

        def apply_style_data(row):
            styles = [''] * len(row)
            for i, col_name in enumerate(row.index):
                if is_multiindex:
                    metric_display_name = col_name[1] 
                else:
                    metric_display_name = col_name
                
                styles[i] = f'background-color: {bg_map_display.get(metric_display_name, "")};'
                
                if metric_display_name == 'P-value': 
                    p_value = row[col_name]
                    if pd.notna(p_value) and p_value < 0.05:
                        styles[i] += 'color: red; font-weight: bold;'
                
                elif metric_display_name == "Effect Size (Hedges' g)": 
                    hedges_g_value = row[col_name]
                    if pd.notna(hedges_g_value) and ((hedges_g_value > 0.8) or (hedges_g_value < -0.8)):
                        styles[i] += 'color: green; font-weight: bold;'
            return styles

        styled_df = df.style.apply(apply_style_data, axis=1)
        table_styles = [
            {'selector': 'td', 'props': [('text-align', 'center')]},
            {'selector': 'tbody th', 'props': [('text-align', 'left'), ('font-weight', 'normal')]},
        ]
        
        if is_multiindex:
            table_styles.append({'selector': 'th.level0', 'props': [
                ('font-size', '1.1em'), 
                ('font-weight', 'bold'), 
                ('background-color', '#dddddd')
            ]})
            styles_level1 = []
            for col_name in df.columns:
                metric_display_name = col_name[1] 
                styles_level1.append({'selector': f'th.level1.col{df.columns.get_loc(col_name)}', 
                                      'props': [('background-color', bg_map_display.get(metric_display_name, '#ffffff'))]})
            styled_df = styled_df.set_table_styles(styles_level1, overwrite=False, axis=0)
            
        styled_df = styled_df.set_table_styles(table_styles, overwrite=False)
        return styled_df

    def _calculate_comparison_table_core(analytes_df, metadata_df, control_group, group, stats_df, return_multiindex=True):
        """Core function to calculate all comparison metrics between two groups."""
        
        control_mean_col = f'{control_group} (Mean)' 
        treatment_mean_col = f'{group} (Mean)' 
        data_with_group = analytes_df.join(metadata_df)
        treatment_data_raw = data_with_group[data_with_group['Grupo'] == group].drop(columns=['Grupo'])
        n_treat = len(treatment_data_raw)
        control_data_raw = data_with_group[data_with_group['Grupo'] == control_group].drop(columns=['Grupo'])
        n_control = len(control_data_raw)
        
        p_values = []
        hedges_g_values = []
        uncertainties_pct = []
        uncertainties_fc = []

        control_means_for_ratio = stats_df[control_mean_col].replace(0, np.nan) 
        percent_variation = ((stats_df[treatment_mean_col] - stats_df[control_mean_col]) / control_means_for_ratio) * 100
        percent_variation = percent_variation.fillna(0) 
        fold_change_values = (stats_df[treatment_mean_col] / control_means_for_ratio)
        fold_change_values = fold_change_values.fillna(1.0) 
        
        for metabolite in stats_df.index:
            p_value = np.nan
            hedges_g = np.nan
            ratio_uncertainty = np.nan 
            uncertainty_pct = np.nan   
            
            mean_treat = stats_df.loc[metabolite, treatment_mean_col]
            mean_control = stats_df.loc[metabolite, control_mean_col]
            treat_vals = treatment_data_raw[metabolite].values
            control_vals = control_data_raw[metabolite].values

            std_treat = np.std(treat_vals, ddof=1) if n_treat > 1 else np.nan
            std_control = np.std(control_vals, ddof=1) if n_control > 1 else np.nan
            se_treat = std_treat / np.sqrt(n_treat) if n_treat > 0 and pd.notna(std_treat) else np.nan
            se_control = std_control / np.sqrt(n_control) if n_control > 0 and pd.notna(std_control) else np.nan

            if pd.notna(se_treat) and pd.notna(se_control) and mean_control != 0 and mean_treat != 0:
                try:
                    term_treat = (se_treat / mean_treat)**2
                    term_control = (se_control / mean_control)**2
                    ratio_uncertainty = abs(mean_treat / mean_control) * np.sqrt(term_treat + term_control)
                    uncertainty_pct = ratio_uncertainty * 100 
                except Exception:
                    ratio_uncertainty = np.nan
                    uncertainty_pct = np.nan
            
            uncertainties_fc.append(ratio_uncertainty)
            uncertainties_pct.append(uncertainty_pct)
            
            if n_treat > 1 and n_control > 1:
                try:
                    t_stat, p_value = stats.ttest_ind(treat_vals, control_vals, equal_var=False, nan_policy='omit')
                except Exception:
                    p_value = np.nan

                df_hedges = n_treat + n_control - 2
                if df_hedges > 0 and pd.notna(std_treat) and pd.notna(std_control):
                    sp = np.sqrt(((n_treat - 1) * std_treat**2 + (n_control - 1) * std_control**2) / df_hedges)
                    J = 1 - (3 / (4 * df_hedges - 1)) if (4 * df_hedges - 1) > 0 else 1 
                    if sp > 0:
                        cohens_d = (mean_treat - mean_control) / sp
                        hedges_g = cohens_d * J
                    else:
                        hedges_g = 0.0
                else:
                    hedges_g = np.nan
            p_values.append(p_value)
            hedges_g_values.append(hedges_g)
        
        metrics_internal_names = [
            'Variation %', 'Uncertainty (Error Prop.)', 'Fold Change', 'Uncertainty (FC)', 'P-value', "Effect Size (Hedges' g)"
        ]
        comparison_results = pd.DataFrame({
            'Variation %': percent_variation.round(2),
            'Uncertainty (Error Prop.)': np.array(uncertainties_pct).round(2), 
            'Fold Change': fold_change_values.round(2),
            'Uncertainty (FC)': np.array(uncertainties_fc).round(2),
            'P-value': np.array(p_values).round(4),
            "Effect Size (Hedges' g)": np.array(hedges_g_values).round(3) 
        }, index=stats_df.index)
        comparison_results = comparison_results.reindex(columns=metrics_internal_names)

        metrics_display_names = [
            'Variation %', 'Uncertainty (%)', 'Fold Change', 'Uncertainty (FC)', 'P-value', "Effect Size (Hedges' g)",
        ]
        
        if return_multiindex:
            comparison_results.columns = pd.MultiIndex.from_product([
                [f'{group} vs {control_group}'], metrics_display_names 
            ])
        else:
            comparison_results.columns = metrics_display_names
        return comparison_results

    def plot_zscore_heatmap(analytes_df):
        """Plots a Z-Score heatmap (Metabolites x Samples)"""
        st.subheader("Z-Score Heatmap (Metabolites x Samples)") 
        st.info("This heatmap applies an *additional* Z-Score (Auto-scaling) for visualization purposes only, centering the mean of each metabolite at zero.") 
        try:
            analytes_heatmap = analytes_df.loc[:, analytes_df.var() > 0].copy()
            if not analytes_heatmap.empty:
                scaler_hm = StandardScaler()
                z_score_scaled = scaler_hm.fit_transform(analytes_heatmap)
                df_zscore = pd.DataFrame(z_score_scaled, index=analytes_heatmap.index, columns=analytes_heatmap.columns)
                df_zscore_T = df_zscore.T.copy()
                fig_heatmap = px.imshow(
                    df_zscore_T, color_continuous_scale='RdBu_r', zmin=-3, zmax=3,
                    aspect="auto", title="Z-Score Heatmap (Metabolites x Samples)" 
                )
                fig_heatmap.update_layout(
                    yaxis_title="Metabolites", xaxis_title="Samples", height=800, 
                    margin=dict(l=50, r=50, t=50, b=50),
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                 st.warning("Could not generate heatmap (no variance in data).") 
        except Exception as e:
            st.error(f"Error generating Z-Score Heatmap: {e}") 

    def data_quality_report(analytes_df, metadata_df):
        """Generates a data quality report (PCA, Histogram, CV) Post-processing."""
        
        st.markdown("---")
        st.header("📊 Data Quality Report (Post-processing)") 
        
        quality_tabs = st.tabs(["📋 Metrics and Distribution", "📊 PCA (Overview)", "🔥 Data Variation (Heatmap)"])
        
        with quality_tabs[0]:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", len(analytes_df))
                st.metric("Total Metabolites", len(analytes_df.columns))
            with col2:
                missing_percent = (analytes_df.isnull().sum().sum() / (analytes_df.shape[0] * analytes_df.shape[1]) * 100)
                st.metric("Missing Values", f"{missing_percent:.2f}%")
                zero_percent = ((analytes_df == 0).sum().sum() / (analytes_df.shape[0] * analytes_df.shape[1]) * 100)
                st.metric("Zero Values", f"{zero_percent:.2f}%")
            with col3:
                cv_values = analytes_df.std() / analytes_df.mean()
                high_cv = (cv_values > 0.5).sum()
                st.metric("Metabolites with CV > 50%", high_cv)
            
            st.subheader("Value Distribution (Post-processing)") 
            fig = px.histogram(
                x=analytes_df.values.flatten(), nbins=50,
                title="Value Distribution (All Data)", 
                labels={'x': 'Post-processed Value', 'y': 'Frequency'} 
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with quality_tabs[1]:
            st.subheader("PCA Analysis - All Groups (Post-processing)") 
            pca_all_fig, pca_all_df = perform_pca_analysis(analytes_df, metadata_df)
            if pca_all_fig:
                st.plotly_chart(pca_all_fig, use_container_width=False) 
                if pca_all_df is not None:
                    st.subheader("Explained Variance per Component") 
                    scaler = StandardScaler()
                    analytes_pca = analytes_df.loc[:, analytes_df.var() > 0]
                    if not analytes_pca.empty:
                        scaled_data = scaler.fit_transform(analytes_pca)
                        pca_full = PCA()
                        pca_full.fit(scaled_data)
                        explained_var = pca_full.explained_variance_ratio_
                        cumsum_var = np.cumsum(explained_var)
                        var_df = pd.DataFrame({
                            'Component': [f'PC{i+1}' for i in range(len(explained_var))], 
                            'Explained Variance': explained_var, 
                            'Cumulative Variance': cumsum_var 
                        })
                        st.dataframe(var_df.head(10))
                    else:
                        st.warning("Could not calculate explained variance (no variance in data).") 
            else:
                st.warning("Could not perform PCA analysis with all groups.") 

        with quality_tabs[2]:
            plot_zscore_heatmap(analytes_df)

    def perform_pca_analysis(analytes_df, metadata_df):
        """
        Performs PCA analysis. Fixed size 600x600 and proportional axes (1:1).
        INCLUDES HOVER_NAME.
        """
        try:
            analytes_filtered = analytes_df.loc[:, analytes_df.var() > 0]
            if len(analytes_filtered.columns) < 2:
                st.error("Insufficient metabolites with variance for PCA.") 
                return None, None
                
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(analytes_filtered)
            
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(scaled_data)
            
            pca_df = pd.DataFrame(
                data=principal_components, columns=['PC1', 'PC2'], index=analytes_df.index
            )
            pca_df['Grupo'] = metadata_df['Grupo']
            explained_var = pca.explained_variance_ratio_
            
            fig = px.scatter(
                pca_df, 
                x='PC1', 
                y='PC2', 
                color='Grupo',
                title="Scores Plot (Hover over points to see sample name)", 
                hover_name=pca_df.index, 
                size_max=10
            )
            fig.update_layout(
                xaxis_title=f"Component 1 ({explained_var[0]:.1%})", 
                yaxis_title=f"Component 2 ({explained_var[1]:.1%})", 
                width=600,  
                height=600, 
                font=dict(size=12),
                plot_bgcolor='white', paper_bgcolor='white',
                showlegend=True,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.05),
                yaxis_scaleanchor="x",
                yaxis_scaleratio=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
            fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            return fig, pca_df
            
        except Exception as e:
            st.error(f"Error performing PCA analysis: {e}") 
            return None, None

    def _get_confidence_ellipse(x_data, y_data, confidence_level=0.95):
        """Calculates points for a confidence ellipse."""
        if len(x_data) < 3:
            return None, None
        try:
            mean_x, mean_y = np.mean(x_data), np.mean(y_data)
            cov_matrix = np.cov(x_data, y_data)
            chi2_val = stats.chi2.ppf(confidence_level, 2)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            lambda1, lambda2 = eigenvalues
            width = 2 * np.sqrt(chi2_val * lambda1)
            height = 2 * np.sqrt(chi2_val * lambda2)
            angle = np.degrees(np.arctan2(*eigenvectors[:,0][::-1]))
            t = np.linspace(0, 2 * np.pi, 100)
            ellipse_x_r = (width / 2) * np.cos(t)
            ellipse_y_r = (height / 2) * np.sin(t)
            R = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                          [np.sin(np.radians(angle)),  np.cos(np.radians(angle))]])
            x_path, y_path = R.dot([ellipse_x_r, ellipse_y_r])
            x_path += mean_x
            y_path += mean_y
            return x_path, y_path
        except Exception as e:
            st.warning(f"Could not calculate confidence ellipse: {e}") 
            return None, None

    def perform_pca_with_ellipses(analytes_df, metadata_df, pc_x, pc_y, n_components_to_calc=10):
        """
        Performs PCA with ellipses. Fixed size 600x600 and proportional axes (1:1).
        INCLUDES HOVER_NAME.
        """
        try:
            analytes_filtered = analytes_df.loc[:, analytes_df.var() > 0]
            max_comps = min(n_components_to_calc, len(analytes_filtered.index), len(analytes_filtered.columns))
            if max_comps < 2:
                st.error("Insufficient metabolites or samples with variance for PCA.") 
                return None, None, None
                
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(analytes_filtered)
            
            pca = PCA(n_components=max_comps)
            principal_components = pca.fit_transform(scaled_data)
            pc_columns = [f'PC{i+1}' for i in range(max_comps)]
            
            pca_df = pd.DataFrame(data=principal_components, columns=pc_columns, index=analytes_df.index)
            pca_df['Grupo'] = metadata_df['Grupo']
            
            explained_var_ratio = pca.explained_variance_ratio_
            explained_var_dict = {col: var for col, var in zip(pc_columns, explained_var_ratio)}
            
            fig = go.Figure()
            colors_map = px.colors.qualitative.Plotly
            groups = sorted(pca_df['Grupo'].unique())
            
            for i, group in enumerate(groups):
                group_data = pca_df[pca_df['Grupo'] == group]
                color = colors_map[i % len(colors_map)]
                fig.add_trace(go.Scatter(
                    x=group_data[pc_x], y=group_data[pc_y], mode='markers', name=group,
                    marker=dict(color=color, size=10), 
                    hovertext=group_data.index 
                ))
                x_ellipse, y_ellipse = _get_confidence_ellipse(group_data[pc_x], group_data[pc_y])
                if x_ellipse is not None and y_ellipse is not None:
                    fig.add_trace(go.Scatter(
                        x=x_ellipse, y=y_ellipse, mode='lines', name=f'{group} (95% CI)',
                        line=dict(color=color, dash='dot', width=2), showlegend=False,
                        hoverinfo='none'
                    ))

            fig.update_layout(
                title=f"PCA Scores Plot ({pc_y} vs {pc_x}) <br>(Hover over points to see sample name)", 
                xaxis_title=f"{pc_x} ({explained_var_dict.get(pc_x, 0):.1%})",
                yaxis_title=f"{pc_y} ({explained_var_dict.get(pc_y, 0):.1%})",
                width=600,  
                height=600, 
                font=dict(size=12),
                plot_bgcolor='white', paper_bgcolor='white',
                showlegend=True,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.05),
                yaxis_scaleanchor="x",
                yaxis_scaleratio=1,
                xaxis=dict(showline=True, mirror=True, linewidth=2, linecolor='black',
                           showgrid=True, gridwidth=1, gridcolor='lightgray'),
                yaxis=dict(showline=True, mirror=True, linewidth=2, linecolor='black',
                           showgrid=True, gridwidth=1, gridcolor='lightgray')
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
            fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
            return fig, pca_df, explained_var_dict
            
        except Exception as e:
            st.error(f"Error performing PCA analysis with ellipses: {e}") 
            return None, None, None


    def perform_plsda_analysis(analytes_df, metadata_df, n_components=5):
        """Performs PLS-DA analysis and calculates performance metrics (R2 and Q2)."""
        try:
            X = analytes_df.loc[:, analytes_df.var() > 0]
            X_index = X.index
            X_columns = X.columns
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            Y_dummies = pd.get_dummies(metadata_df['Grupo'])
            Y_groups = metadata_df['Grupo']
            
            max_comps = min(n_components, len(X_index)-1, len(X_columns))
            if max_comps < 1:
                st.error("Insufficient data for PLS-DA (n_components < 1).") 
                return None, None, None, None, None, None
            
            plsda = PLSRegression(n_components=max_comps, scale=False)
            plsda.fit(X_scaled, Y_dummies)
            
            scores = plsda.x_scores_
            loadings = plsda.x_loadings_
            comp_names = [f'Component {i+1}' for i in range(max_comps)] 
            
            scores_df = pd.DataFrame(scores, columns=comp_names, index=X_index)
            scores_df['Grupo'] = Y_groups
            loadings_df = pd.DataFrame(loadings, columns=comp_names, index=X_columns)
            
            r2x = np.sum(np.var(scores, axis=0)) / np.sum(np.var(X_scaled, axis=0))
            r2y = r2_score(Y_dummies, plsda.predict(X_scaled))
            
            cv_folds = min(5, len(X_index)-1)
            if cv_folds < 2:
                 st.warning("Cannot calculate Q2 (too few samples for cross-validation).") 
                 q2 = np.nan
            else:
                 q2 = r2_score(Y_dummies, cross_val_predict(plsda, X_scaled, Y_dummies, cv=cv_folds))
            
            explained_var_x = np.var(scores, axis=0) / np.sum(np.var(X_scaled, axis=0))
            explained_var_dict = {comp: var for comp, var in zip(comp_names, explained_var_x)}

            return scores_df, loadings_df, explained_var_dict, r2y, q2, r2x

        except Exception as e:
            st.error(f"Error calculating PLS-DA: {e}") 
            return None, None, None, None, None, None

    def plot_plsda_scores(scores_df, comp_x, comp_y, explained_var_dict):
        """
        Plots the PLS-DA Scores Plot with ellipses. Size 600x600 and axes 1:1.
        INCLUDES HOVER_NAME.
        """
        fig = go.Figure()
        colors_map = px.colors.qualitative.Plotly
        groups = sorted(scores_df['Grupo'].unique())

        for i, group in enumerate(groups):
            group_data = scores_df[scores_df['Grupo'] == group]
            color = colors_map[i % len(colors_map)]
            fig.add_trace(go.Scatter(
                x=group_data[comp_x], y=group_data[comp_y], mode='markers', name=group,
                marker=dict(color=color, size=10), 
                hovertext=group_data.index 
            ))
            x_ellipse, y_ellipse = _get_confidence_ellipse(group_data[comp_x], group_data[comp_y])
            if x_ellipse is not None and y_ellipse is not None:
                fig.add_trace(go.Scatter(
                    x=x_ellipse, y=y_ellipse, mode='lines', name=f'{group} (95% CI)',
                    line=dict(color=color, dash='dot', width=2), showlegend=False,
                    hoverinfo='none'
                ))

        fig.update_layout(
            title=f"PLS-DA Scores Plot ({comp_y} vs {comp_x}) <br>(Hover over points to see sample name)", 
            xaxis_title=f"{comp_x} ({explained_var_dict.get(comp_x, 0):.1%})",
            yaxis_title=f"{comp_y} ({explained_var_dict.get(comp_y, 0):.1%})",
            width=600,  
            height=600, 
            font=dict(size=12),
            plot_bgcolor='white', paper_bgcolor='white',
            showlegend=True,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.05),
            yaxis_scaleanchor="x",
            yaxis_scaleratio=1,
            xaxis=dict(showline=True, mirror=True, linewidth=2, linecolor='black',
                       showgrid=True, gridwidth=1, gridcolor='lightgray'),
            yaxis=dict(showline=True, mirror=True, linewidth=2, linecolor='black',
                       showgrid=True, gridwidth=1, gridcolor='lightgray')
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
        return fig

    def plot_plsda_loadings(loadings_df, comp_to_plot, top_n=20):
        """Plots the PLS-DA Loadings for a component."""
        try:
            loadings_comp = loadings_df[comp_to_plot].sort_values(ascending=False)
            top_pos = loadings_comp.head(top_n)
            top_neg = loadings_comp.tail(top_n)
            loadings_to_plot = pd.concat([top_pos, top_neg]).sort_values(ascending=False)
            
            fig = px.bar(
                loadings_to_plot, x=loadings_to_plot.index, y=loadings_to_plot.values,
                title=f"PLS-DA Loading Plot ({comp_to_plot})", 
                labels={'x': 'Metabolite', 'y': 'Contribution (Loading)'}, 
                color=loadings_to_plot.values, color_continuous_scale='RdBu_r' 
            )
            fig.update_layout(xaxis_tickangle=45, showlegend=False, coloraxis_showscale=False)
            return fig
        except Exception as e:
            st.error(f"Error plotting loadings: {e}") 
            return go.Figure()

    @st.cache_data
    def get_metabolite_profile_classification(_analytes_df, _metadata_df):
        """Calculates the Consumed/Excreted profile (MCC vs MP)"""
        try:
            required_groups = ["Pure Medium (MP)", "Cell Control (MCC)"] 
            if not all(g in _metadata_df['Grupo'].unique() for g in required_groups):
                st.error("Cache Error: MP or MCC groups not found for classification.") 
                return pd.DataFrame(columns=['% MP Level', 'Profile']) 
                
            meta_filtered = _metadata_df[_metadata_df['Grupo'].isin(required_groups)]
            analytes_filtered = _analytes_df.loc[meta_filtered.index]
            stats_df = calculate_descriptive_stats(analytes_filtered, meta_filtered)
            
            comparison_df = _calculate_comparison_table_core(
                analytes_filtered, meta_filtered, "Pure Medium (MP)", "Cell Control (MCC)", 
                stats_df, return_multiindex=False
            )
            
            comparison_df['% MP Level'] = comparison_df['Fold Change'] * 100 
            comparison_df['Profile'] = comparison_df['% MP Level'].apply(classify_metabolite) 
            comparison_df = comparison_df.sort_values(by='% MP Level', ascending=True)
            
            return comparison_df[['% MP Level', 'Profile']]
            
        except Exception as e:
            st.error(f"Error calculating metabolite order: {e}") 
            return pd.DataFrame(columns=['% MP Level', 'Profile'])


    def _prepare_heatmap_data(analytes_data, metadata_data, reference_group, comparison_groups, 
                              metric_name, metabolite_order):
        """Helper function to calculate heatmap data."""
        try:
            stats = calculate_descriptive_stats(analytes_data, metadata_data)
            heatmap_metric_df = pd.DataFrame(index=analytes_data.columns)
            heatmap_pvalue_df = pd.DataFrame(index=analytes_data.columns)
            
            for group in comparison_groups:
                comp_df = _calculate_comparison_table_core(
                    analytes_data, metadata_data, reference_group, group, stats, return_multiindex=False
                )
                heatmap_pvalue_df[f"{group} vs {reference_group}"] = comp_df["P-value"]
                
                if metric_name == "Variation %": 
                    heatmap_metric_df[f"{group} vs {reference_group}"] = comp_df["Variation %"]
                elif metric_name == "Log2 Fold Change": 
                    epsilon = 1e-9
                    log2_fc = np.log2(comp_df['Fold Change'].replace(0, epsilon).replace(np.nan, 1))
                    heatmap_metric_df[f"{group} vs {reference_group}"] = log2_fc
            
            metric_data = heatmap_metric_df.T.dropna(how='all', axis=1)
            pvalue_data = heatmap_pvalue_df.T.dropna(how='all', axis=1)
            
            ordered_cols = [col for col in metabolite_order if col in metric_data.columns]
            remaining_cols = [col for col in metric_data.columns if col not in ordered_cols]
            final_ordered_cols = ordered_cols + remaining_cols
            
            metric_data = metric_data[final_ordered_cols]
            pvalue_data = pvalue_data.reindex(columns=final_ordered_cols) 
            return metric_data, pvalue_data
            
        except Exception as e:
            st.error(f"Error preparing heatmap data: {e}") 
            return pd.DataFrame(), pd.DataFrame()


    def plot_comparison_heatmap(metric_df, pvalue_df, metric_name):
        """Plots the heatmap with asterisks."""
        try:
            all_values = metric_df.stack().dropna().replace([np.inf, -np.inf], np.nan).dropna()
            if not all_values.empty:
                q_low = all_values.quantile(0.05)
                q_high = all_values.quantile(0.95)
                if metric_name == "Log2 Fold Change":
                    vmax = max(abs(q_low), abs(q_high), 1.0); vmax = min(vmax, 5) 
                else:
                    vmax = max(abs(q_low), abs(q_high), 50.0); vmax = min(vmax, 500)
            else:
                vmax = 1.0 if metric_name == "Log2 Fold Change" else 100.0
            zmin_calc = -vmax
            zmax_calc = vmax
        except Exception:
            zmin_calc = -2.0 if metric_name == "Log2 Fold Change" else -100.0
            zmax_calc = 2.0 if metric_name == "Log2 Fold Change" else 100.0

        if metric_name == "Variation %": 
            title = "Variation % Heatmap (Comparison vs Reference)" 
            color_label = "Variation %" 
        else:
            title = "Log2 Fold Change Heatmap (Comparison vs Reference)" 
            color_label = "Log2 (Fold Change)" 

        color_scale = 'RdBu_r' 
        text_df = pvalue_df.applymap(lambda p: '*' if pd.notna(p) and p < 0.05 else '')
        
        fig = px.imshow(
            metric_df, aspect="auto", color_continuous_scale=color_scale, 
            zmin=zmin_calc, zmax=zmax_calc, title=title,
            labels=dict(color=color_label, x="Metabolite", y="Comparison") 
        )
        fig.update_traces(
            text=text_df.values, texttemplate="%{text}", textfont=dict(size=16, color='black'),
            hovertemplate= (
                "<b>Metabolite</b>: %{x}<br>" + "<b>Comparison</b>: %{y}<br>" + 
                f"<b>{color_label}</b>: %{{z:.2f}}<br>" + "<b>P-value</b>: %{customdata:.4f}<extra></extra>" 
            ),
            customdata=pvalue_df.values
        )
        fig.update_layout(
            height=max(350, len(metric_df.index) * 40 + 200), 
            xaxis_tickangle=90,
            coloraxis_colorbar=dict(
                title=color_label, thicknessmode="pixels", thickness=25, 
                lenmode="pixels", len=max(200, len(metric_df.index) * 40 + 100), 
                tickfont=dict(size=12), title_font=dict(size=14)
            ),
            margin=dict(t=50, r=100, b=50, l=50)
        )
        return fig

    def generate_interpretation_table(base_profile_df, treatment_comparison_df):
        """Generates the interpretation table."""
        results = []
        for metabolite in base_profile_df.index:
            if metabolite not in treatment_comparison_df.index:
                continue
            base_profile = base_profile_df.loc[metabolite, 'Profile']
            fc_treat = treatment_comparison_df.loc[metabolite, 'Fold Change']
            p_val_treat = treatment_comparison_df.loc[metabolite, 'P-value']
            interpretation = "No Significant Change"
            
            if pd.notna(p_val_treat) and p_val_treat < 0.05:
                fc_threshold_low = 0.95 
                fc_threshold_high = 1.05
                if base_profile == "Consumed":
                    if fc_treat > fc_threshold_high: interpretation = "Less Consumed"
                    elif fc_treat < fc_threshold_low: interpretation = "More Consumed"
                elif base_profile == "Excreted":
                    if fc_treat > fc_threshold_high: interpretation = "More Excreted"
                    elif fc_treat < fc_threshold_low: interpretation = "Less Excreted"
                elif base_profile == "Unaltered":
                    if fc_treat > fc_threshold_high: interpretation = "Induced Excretion"
                    elif fc_treat < fc_threshold_low: interpretation = "Induced Consumption"
                        
            results.append({
                'Profile (Base)': base_profile, 'Metabolite': metabolite, 
                'Fold Change (vs MCC)': fc_treat, 'P-value (vs MCC)': p_val_treat, 
                'Interpretation': interpretation 
            })
        return pd.DataFrame(results)

    def style_interpretation_table(df):
        """Applies colors to the interpretation table."""
        color_map = {
            "More Consumed": 'background-color: #ffadad',
            "Induced Consumption": 'background-color: #ffd6d6',
            "Less Consumed": 'background-color: #d6ffd6',
            "More Excreted": 'background-color: #adffad',
            "Induced Excretion": 'background-color: #d6f5ff',
            "Less Excreted": 'background-color: #fff5d6',
            "No Significant Change": 'background-color: #f0f0f0'
        }
        def apply_color(val):
            return color_map.get(val, '')

        styled_df = df.style.applymap(apply_color, subset=['Interpretation']) 
        styled_df = styled_df.format({
            'Fold Change (vs MCC)': '{:.2f}', 'P-value (vs MCC)': '{:.4f}' 
        })
        styled_df = styled_df.applymap(
            lambda p: 'color: red; font-weight: bold;' if pd.notna(p) and p < 0.05 else '', 
            subset=['P-value (vs MCC)'] 
        )
        return styled_df


    # ----------------------------------------------------
    # --- SIDEBAR (REORGANIZED AND TRANSLATED) (REQUEST 2) ---
    # ----------------------------------------------------

    with st.sidebar:
        st.title("Navigation (Exometabolome)") 
        menu_option = st.radio(
            "Select Step:", 
            [
                "1. 📤 Upload and Map", 
                "2. 🔍 Review (Raw Data)",
                "3. 🔧 Pre-processing",
                "4. 📊 Analysis: Cellular Profile", 
                "5. 🔬 Analysis: Treatments", 
                "6. 📈 Advanced Analyses" # <-- Changed
            ] 
        )

    # ----------------------------------------------------
    # --- SECTION 1: UPLOAD AND MAPPING ---
    # ----------------------------------------------------

    if menu_option == "1. 📤 Upload and Map": 
        st.header("1. 📤 Upload and Group Mapping") 
        
        st.subheader("1.1 Main Data File (Required)") 
        st.markdown("""
        **Expected Format (CSV, TXT or Excel):**
        - **Column 1:** Sample Name (unique ID)
        - **Column 2:** Group (e.g., "Control", "Medium", "Treat_A")
        - **Column 3 onwards:** Metabolite Concentration
        - The **first row** must be the header.
        """) 
        
        uploader_main = st.file_uploader(
            "Upload data file (CSV, TXT, XLSX)", 
            type=["csv", "txt", "xlsx"],
            key="uploader_main"
        )
        
        st.subheader("1.2 Normalization Factor File (Optional)") 
        st.markdown("""
        Used to normalize data by cell count, dry weight, total protein, etc.
        The **Sample Name** (Column 1) must exactly match the Main Data File.
        """) 
        
        uploader_factor = st.file_uploader(
            "Upload normalization factor file (CSV, TXT, XLSX)", 
            type=["csv", "txt", "xlsx"],
            key="uploader_factor"
        )
        
        st.markdown("---")
        st.subheader("⚙️ File Reading Settings (CSV/TXT)") 
        st.info(f"""
        ⚠️ **Check your Settings BEFORE uploading!**
        - For `.txt` or `.tsv` files, the separator is usually **Tab (TAB)** and the decimal is **Period (.)**.
        - For `.csv` (International), the separator is usually **Comma (,)** and the decimal is **Period (.)**.
        - For `.csv` (Brazil/Europe), the separator is usually **Semicolon (;) ** and the decimal is **Comma (,)**.
        """) 
        
        sep_options = {
            'Comma (,)': ',', 
            'Semicolon (;)': ';', 
            'Tab (TAB)': '\t'
        } 
        selected_sep_name = st.radio(
            "Column Separator:", 
            options=list(sep_options.keys()), 
            index=0, 
            horizontal=True,
            key='sep_radio'
        )
        sep_char = sep_options[selected_sep_name]
        
        decimal_char = st.radio(
            "Decimal Separator:", 
            options=['Period (.)', 'Comma (,)'], 
            index=0,
            horizontal=True,
            key='decimal_radio'
        )
        
        selected_encoding = st.selectbox("Encoding", options=['utf-8', 'latin1', 'cp1252'], index=1)
        st.markdown("---")
        
        process_button = st.button("▶️ 1. LOAD AND VERIFY DATA") 

        if process_button:
            if uploader_main is not None:
                with st.spinner('Reading and processing files...'): 
                    try:
                        file_ext = uploader_main.name.split('.')[-1].lower()
                        if file_ext == 'xlsx':
                            df_main = pd.read_excel(uploader_main)
                        else: 
                            df_main = pd.read_csv(
                                uploader_main, 
                                sep=sep_char, 
                                encoding=selected_encoding, 
                                decimal=',' if decimal_char == 'Comma (,)' else '.' 
                            )
                        
                        df_main = df_main.dropna(how='all')
                        
                        if len(df_main.columns) < 3:
                            st.error(f"Error: The main file must have at least 3 columns (Sample, Group, Metabolite1). Found: {len(df_main.columns)}") 
                            st.stop()

                        sample_col = df_main.columns[0]
                        group_col = df_main.columns[1]
                        df_main = df_main.set_index(sample_col)
                        
                        metadata_df_raw = df_main[[group_col]].rename(columns={group_col: 'Grupo'})
                        analytes_df_raw_text = df_main.drop(columns=[group_col])
                        
                        metadata_df = metadata_df_raw.dropna(subset=['Grupo'])
                        analytes_df_text_clean = analytes_df_raw_text.loc[metadata_df.index]
                        
                        analytes_df_numeric = analytes_df_text_clean.apply(pd.to_numeric, errors='coerce')
                        num_nan_before = analytes_df_numeric.isnull().sum().sum()
                        analytes_df_filled = analytes_df_numeric.fillna(0)
                        
                        df_factor = None
                        if uploader_factor is not None:
                            file_ext_f = uploader_factor.name.split('.')[-1].lower()
                            if file_ext_f == 'xlsx':
                                df_factor = pd.read_excel(uploader_factor)
                            else:
                                df_factor = pd.read_csv(
                                    uploader_factor, sep=sep_char, encoding=selected_encoding,
                                    decimal=',' if decimal_char == 'Comma (,)' else '.' 
                                )
                            
                            if len(df_factor.columns) < 2:
                                st.error(f"Error: The factor file must have 2 columns (Sample, Factor). Found: {len(df_factor.columns)}") 
                                df_factor = None
                            else:
                                factor_sample_col = df_factor.columns[0]
                                df_factor = df_factor.set_index(factor_sample_col)
                                if not all(idx in df_factor.index for idx in analytes_df_filled.index):
                                    st.error("Error: The 'Sample Names' in the factor file do not 100% match the 'Sample Names' in the main file. Factor normalization has been disabled.") 
                                    df_factor = None
                        
                        st.session_state['temp_analytes'] = analytes_df_filled
                        st.session_state['temp_metadata'] = metadata_df
                        st.session_state['temp_preview'] = analytes_df_text_clean 
                        st.session_state['temp_sample_col'] = sample_col
                        st.session_state['temp_factor'] = df_factor
                        st.session_state['temp_nan_count'] = num_nan_before
                        
                        st.session_state['groups_loaded'] = True
                        st.session_state['data_uploaded'] = False 
                        st.session_state['data_processed'] = False
                        
                        st.success("✅ File(s) Read! Proceed to Group Mapping below.") 
                        
                    except Exception as e:
                        st.error(f"Error reading the file: {e}") 
                        st.error("Please check if the **Column Separator** and **Decimal Separator** are correct for your file.") 
                        st.session_state['groups_loaded'] = False

        if st.session_state.get('groups_loaded', False):
            
            st.markdown("---")
            st.subheader("📋 Loaded Data Summary (Original)") 
            
            temp_metadata = st.session_state['temp_metadata']
            total_samples = len(temp_metadata)
            total_groups = temp_metadata['Grupo'].nunique()
            col_sum1, col_sum2 = st.columns(2)
            col_sum1.metric("Total Samples", total_samples) 
            col_sum2.metric("Total Groups", total_groups) 
            
            st.markdown("#### Sample Count per Group (Original)") 
            group_counts = temp_metadata['Grupo'].value_counts().rename_axis('Group').reset_index(name='# of Samples') 
            st.dataframe(group_counts, use_container_width=True)
            
            num_nan_before = st.session_state['temp_nan_count']
            if num_nan_before > 0:
                st.warning(f"⚠️ Found **{num_nan_before}** missing (NaN) or non-numeric values *in the metabolite data*. They will be filled with zero (0) before normalization.") 
            
            st.subheader("Loaded Data Table Preview (Original)") 
            preview_df = temp_metadata.join(st.session_state['temp_preview']) 
            preview_df = preview_df.reset_index().rename(columns={'index': st.session_state['temp_sample_col']})
            st.dataframe(preview_df, height=300, use_container_width=True)
            
            st.markdown("---")
            
            st.subheader("1.3 Group Mapping (Required)") 
            st.warning("The platform needs to know which of YOUR groups are the main controls. All other analyses depend on this.") 
            
            all_groups = temp_metadata['Grupo'].unique().tolist()
            
            col_map1, col_map2 = st.columns(2)
            with col_map1:
                mp_group_select = st.selectbox(
                    "Select which of YOUR groups is the 'Pure Medium' (MP):", 
                    options=all_groups, 
                    index=0
                )
            with col_map2:
                default_mcc_index = min(1, len(all_groups) - 1)
                mcc_group_select = st.selectbox(
                    "Select which of YOUR groups is the 'Cell Control' (MCC):", 
                    options=all_groups, 
                    index=default_mcc_index
                )
                
            confirm_button = st.button("▶️ 2. CONFIRM GROUPS AND SAVE DATA", type="primary") 
            
            if confirm_button:
                if mp_group_select == mcc_group_select:
                    st.error("Error: 'Pure Medium' and 'Cell Control' cannot be the same group. Please select different groups.") 
                else:
                    with st.spinner("Mapping groups and saving data..."): 
                        mapping = {
                            mp_group_select: "Pure Medium (MP)", 
                            mcc_group_select: "Cell Control (MCC)" 
                        }
                        metadata_to_rename = st.session_state['temp_metadata'].copy()
                        metadata_renamed = metadata_to_rename
                        metadata_renamed['Grupo'] = metadata_to_rename['Grupo'].map(mapping).fillna(metadata_to_rename['Grupo'])
                        
                        st.session_state['data_analytes_raw'] = st.session_state['temp_analytes']
                        st.session_state['data_metadata_raw'] = metadata_renamed
                        st.session_state['data_sample_col_name'] = st.session_state['temp_sample_col']
                        if st.session_state['temp_factor'] is not None:
                             st.session_state['data_factor'] = st.session_state['temp_factor']
                        
                        del st.session_state['temp_analytes']
                        del st.session_state['temp_metadata']
                        del st.session_state['temp_preview']
                        del st.session_state['temp_sample_col']
                        del st.session_state['temp_factor']
                        
                        st.session_state['data_uploaded'] = True
                        st.session_state['data_processed'] = False
                        st.session_state['groups_loaded'] = False
                        
                        st.success("Mapping saved successfully! Your data is ready.") 
                        st.info("You can proceed to **Section 2. 🔍 Review (Raw Data)** in the sidebar.") 
                        time.sleep(2)
                        st.rerun()


    # ----------------------------------------------------
    # --- SECTION 2: REVIEW (Raw Data) ---
    # ----------------------------------------------------
    elif menu_option == "2. 🔍 Review (Raw Data)": 
        
        st.header("2. 🔍 Review (Raw Data)") 
        
        if not st.session_state.get('data_uploaded', False):
            st.warning("💡 **Please upload and map your data in 'Section 1. 📤 Upload and Map' first.**") 
            st.stop()
            
        analytes_raw = st.session_state['data_analytes_raw']
        metadata_raw = st.session_state['data_metadata_raw']

        st.info("This is a review of your raw data quality, **before** any normalization. The control groups already reflect the mapping from Section 1.") 
        
        st.subheader("📋 Raw Data Summary (Post-Mapping)") 
        
        total_samples = len(metadata_raw)
        total_groups = metadata_raw['Grupo'].nunique()
        total_metabolites = analytes_raw.shape[1] 
        
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        col_sum1.metric("Total Samples", total_samples) 
        col_sum2.metric("Total Groups", total_groups) 
        col_sum3.metric("Total Metabolites", total_metabolites) 
        
        st.markdown("#### Sample Count per Group (Mapped)") 
        group_counts = metadata_raw['Grupo'].value_counts().rename_axis('Group').reset_index(name='# of Samples') 
        st.dataframe(group_counts, use_container_width=True)
        
        st.markdown("---") 

        quality_tabs_raw = st.tabs(["📑 Raw Data Table", "📊 PCA (Overview)", "🔥 Data Variation (Heatmap)"])
        
        with quality_tabs_raw[0]:
            st.subheader("Raw Data Table (Post-Mapping)") 
            st.markdown("This is the raw data (with NaNs filled as 0) that will be used in Section 3 (Pre-processing).") 
            
            full_raw_df = metadata_raw.join(analytes_raw)
            full_raw_df = full_raw_df.reset_index().rename(columns={'index': st.session_state['data_sample_col_name']})
            st.dataframe(full_raw_df, height=500, use_container_width=True)

        with quality_tabs_raw[1]:
            st.subheader("PCA Analysis - All Groups (Raw Data)") 
            pca_all_fig, pca_all_df = perform_pca_analysis(analytes_raw, metadata_raw)
            if pca_all_fig:
                st.plotly_chart(pca_all_fig, use_container_width=False)
            else:
                st.warning("Could not perform PCA analysis with raw data.") 

        with quality_tabs_raw[2]:
            plot_zscore_heatmap(analytes_raw)

    # ----------------------------------------------------
    # --- SECTION 3: PRE-PROCESSING ---
    # ----------------------------------------------------
    elif menu_option == "3. 🔧 Pre-processing": 
        
        st.header("3. 🔧 Normalization and Pre-processing") 
        
        if not st.session_state.get('data_uploaded', False):
            st.warning("💡 **Please upload and map your data in 'Section 1. 📤 Upload and Map' first.**") 
            st.stop()
            
        analytes_raw = st.session_state['data_analytes_raw']
        metadata_raw = st.session_state['data_metadata_raw']
        factor_df = st.session_state.get('data_factor', None)

        st.markdown("Select the normalization (per-sample) and transformation/scaling (per-metabolite) methods.") 
        st.info("**Tip:** To proceed with the raw data (unchanged) in the following analyses, just select **'None'** for both options and click 'APPLY'.") 
        
        col_norm, col_scale = st.columns(2)
        
        with col_norm:
            st.subheader("1. Normalization (per-Sample)") 
            
            norm_options = ["None"]
            if factor_df is not None:
                norm_options.insert(1, "By External Factor (e.g., Cell Count)")
            norm_options.extend(["By Sum (Total Sum)", "By Median"])
            
            norm_choice = st.selectbox(
                "Select normalization method:", 
                options=norm_options, index=0, key='norm_select',
                help="Normalization adjusts for differences between samples (e.g., different cell amounts)." 
            )
            
            if norm_choice == "By External Factor (e.g., Cell Count)" and factor_df is None:
                st.error("You selected 'By External Factor', but no factor file was uploaded in Section 1.") 
                st.stop()
                
        with col_scale:
            st.subheader("2. Transformation and Scaling (per-Metabolite)") 
            scale_options = [
                "None",
                "Log10 Transformation (Log(x+1))",
                "Auto-scaling (Z-score)",
                "Pareto Scaling"
            ]
            scale_choice = st.selectbox(
                "Select scaling/transformation method:", 
                options=scale_options, index=0, key='scale_select',
                help="Scaling adjusts data so that high and low concentration metabolites have comparable weights." 
            )

        st.markdown("---")
        
        if st.button("▶️ APPLY AND PROCESS DATA", type="primary"): 
            with st.spinner("Applying normalization and scaling..."): 
                
                analytes_processed = analytes_raw.copy()
                
                if norm_choice == "By External Factor (e.g., Cell Count)":
                    st.write("Applying External Factor Normalization...") 
                    factor_col_name = factor_df.columns[0]
                    factor_aligned = factor_df.reindex(analytes_processed.index)
                    analytes_processed = analytes_processed.div(factor_aligned[factor_col_name], axis=0)
                
                elif norm_choice == "By Sum (Total Sum)":
                    st.write("Applying Total Sum Normalization...") 
                    sample_sums = analytes_processed.sum(axis=1)
                    analytes_processed = analytes_processed.div(sample_sums, axis=0)
                elif norm_choice == "By Median":
                    st.write("Applying Median Normalization...") 
                    sample_medians = analytes_processed.median(axis=1)
                    sample_medians[sample_medians == 0] = 1 
                    analytes_processed = analytes_processed.div(sample_medians, axis=0)
                else:
                    st.write("No per-sample normalization applied.") 

                if scale_choice == "Log10 Transformation (Log(x+1))":
                    st.write("Applying Log10 Transformation...") 
                    analytes_processed = analytes_processed.apply(lambda x: np.log10(x + 1))
                elif scale_choice == "Auto-scaling (Z-score)":
                    st.write("Applying Auto-scaling (Z-score)...") 
                    scaler = StandardScaler()
                    analytes_processed_scaled = scaler.fit_transform(analytes_processed)
                    analytes_processed = pd.DataFrame(analytes_processed_scaled, 
                                                      index=analytes_processed.index, 
                                                      columns=analytes_processed.columns)
                elif scale_choice == "Pareto Scaling":
                    st.write("Applying Pareto Scaling...") 
                    analytes_mean = analytes_processed.mean()
                    analytes_std_sqrt = np.sqrt(analytes_processed.std())
                    analytes_std_sqrt[analytes_std_sqrt == 0] = 1
                    analytes_processed = (analytes_processed - analytes_mean) / analytes_std_sqrt
                else:
                     st.write("No per-metabolite transformation/scaling applied.") 

                st.write("Validating post-processed data...") 
                analytes_final = validate_and_clean_data(analytes_processed)
                metadata_final = metadata_raw.copy()
                
                st.session_state['data_analytes'] = analytes_final
                st.session_state['data_metadata'] = metadata_final
                st.session_state['data_processed'] = True
                
                st.success("✅ Data processed and ready for analysis!") 
                
                st.subheader("Post-processing Data Preview") 
                st.dataframe(analytes_final.head(10), use_container_width=True)
                data_quality_report(analytes_final, metadata_final)

        if st.session_state.get('data_processed', False) and not st.button:
            st.info("Data has already been processed. To re-process, change the options and click the button above.") 
            
            st.subheader("Post-processing Data Preview") 
            st.dataframe(st.session_state['data_analytes'].head(10), use_container_width=True)
            data_quality_report(st.session_state['data_analytes'], st.session_state['data_metadata'])

    # ----------------------------------------------------
    # --- SECTION 4: CELLULAR PROFILE ---
    # ----------------------------------------------------
    elif menu_option == "4. 📊 Analysis: Cellular Profile": 
        
        if not st.session_state.get('data_processed', False):
            st.warning("💡 **Please upload, map, and process your data in Sections 1, 2, and 3 first.**") 
            st.stop()
            
        metadata_df = st.session_state['data_metadata']
        analytes_df = st.session_state['data_analytes']
        
        st.header("4. 📊 Cellular Consumption and Excretion Profile (MP vs MCC)") 
        st.info("This analysis uses the POST-processed (normalized/scaled) data from Section 3.") 
        
        required_groups_2 = ["Pure Medium (MP)", "Cell Control (MCC)"] 
        if not all(g in metadata_df['Grupo'].unique() for g in required_groups_2):
            st.error(f"Error: The groups '{required_groups_2[0]}' and '{required_groups_2[1]}' were not found. Go back to Section 1 and map your groups.") 
            st.stop()
        
        analysis_options = [
            "4.1 Descriptive Statistics (MP vs MCC)",
            "4.2 Comparison and Statistical Tests (MP vs MCC)",
            "4.3 Consumption/Excretion Plot"
        ]
        selected_analysis = st.radio(
            "Select Analysis Step (MP vs MCC):", 
            options=analysis_options, index=0, horizontal=True
        )
        filtered_metadata = metadata_df[metadata_df['Grupo'].isin(required_groups_2)].copy()
        filtered_analytes = analytes_df.loc[filtered_metadata.index].copy()
        
        if selected_analysis == "4.1 Descriptive Statistics (MP vs MCC)":
            st.subheader("4.1 Descriptive Statistics (MP vs MCC)") 
            stats_df = calculate_descriptive_stats(filtered_analytes, filtered_metadata)
            st.markdown("##### Mean and Standard Deviation for Pure Medium and Cell Control") 
            st.dataframe(stats_df, use_container_width=True)
            st.download_button(
                label="Download Descriptive Statistics (MP vs MCC)", 
                data=stats_df.to_csv().encode('utf-8'),
                file_name='descriptive_statistics_mp_vs_mcc.csv', mime='text/csv',
            )

        elif selected_analysis == "4.2 Comparison and Statistical Tests (MP vs MCC)":
            st.subheader("4.2 Comparison and Statistical Tests (MP vs MCC)") 
            control_group = "Pure Medium (MP)" 
            comparison_group = "Cell Control (MCC)" 
            
            if st.button(f"Calculate Comparison: {comparison_group} vs {control_group}", key='calc_mp_vs_mcc_btn'): 
                stats_df = calculate_descriptive_stats(filtered_analytes, filtered_metadata)
                comparison_df = _calculate_comparison_table_core(
                    filtered_analytes, filtered_metadata, control_group, comparison_group, stats_df,
                    return_multiindex=False 
                )
                comparison_df = comparison_df.rename_axis(index='Metabolite') 
                st.markdown("#### Basic Profile Comparison Table (Consumption/Excretion)") 
                st.write(f"Comparison: **{comparison_group}** vs **{control_group}** (Reference)") 
                st.write("Light Purple Block: **Variation %** and **Uncertainty (%)**.") 
                st.write("Light Yellow Block: **Fold Change** and **Uncertainty (FC)**.") 
                st.write("Light Pink Column: **P-value**.") 
                st.write("Light Mint Column: **Effect Size (Hedges' g)**.") 
                st.markdown("---")
                st.write("Values in **red** are statistically significant (P-value < 0.05).") 
                st.write("Values in **green** indicate a large Effect Size (|Hedges' g| > 0.8).") 
                styled_df = style_comparison_table(comparison_df)
                st.dataframe(styled_df, use_container_width=True)
                st.download_button(
                    label="Download Consumption/Excretion Profile (CSV)", 
                    data=comparison_df.to_csv().encode('utf-8'),
                    file_name='consumption_excretion_profile.csv', mime='text/csv',
                )

        elif selected_analysis == "4.3 Consumption/Excretion Plot":
            st.subheader("4.3 Metabolite Level (% Pure Medium) and Change Magnitude (Log2 Fold Change) Plot") 
            control_group = "Pure Medium (MP)" 
            comparison_group = "Cell Control (MCC)" 

            stats_df = calculate_descriptive_stats(filtered_analytes, filtered_metadata)
            comparison_df = _calculate_comparison_table_core(
                filtered_analytes, filtered_metadata, control_group, comparison_group, stats_df,
                return_multiindex=False 
            )
            comparison_df = comparison_df.rename_axis(index='Metabolite').reset_index() 
            
            plot_df = comparison_df[['Metabolite', 'Fold Change', 'Uncertainty (FC)']].copy() 
            plot_df['% MP Level'] = plot_df['Fold Change'] * 100 
            plot_df['Error Upper'] = plot_df['Uncertainty (FC)'] * 100 
            plot_df['Error Lower'] = plot_df['Uncertainty (FC)'] * 100 
            epsilon = 1e-9 
            plot_df['Log2 FC'] = np.log2(plot_df['Fold Change'].replace(0, epsilon))
            safe_fc_minus_inc = (plot_df['Fold Change'] - plot_df['Uncertainty (FC)']).clip(lower=epsilon)
            plot_df['Log2 Error Upper'] = np.log2(plot_df['Fold Change'] + plot_df['Uncertainty (FC)']) - plot_df['Log2 FC'] 
            plot_df['Log2 Error Lower'] = plot_df['Log2 FC'] - np.log2(safe_fc_minus_inc) 
            plot_df['Significance'] = np.where(
                comparison_df['P-value'] < 0.05, 'Significant (P < 0.05)', 'Not Significant (P ≥ 0.05)' 
            )
            plot_df['Profile'] = plot_df['% MP Level'].apply(classify_metabolite) 
            plot_df = plot_df.sort_values(by='% MP Level', ascending=True).reset_index(drop=True) 
            metabolite_order = plot_df['Metabolite'].tolist()

            st.markdown("##### All Metabolites Consumption/Excretion Visualization") 
            st.write(f"Comparison: **{comparison_group}** vs **{control_group}**") 
            st.warning("**Warning:** The interpretation of 'Consumed/Excreted' (Plot 1) and 'Log2 FC' (Plot 2) can be affected by the normalization/scaling method chosen in Section 3. For example, Z-scored data should not be interpreted as % of MP.") 
            st.markdown("---")
            
            BACKGROUND_COLORS = {
                'Consumed': 'rgba(240, 240, 240, 0.7)', 'Excreted': 'rgba(255, 192, 203, 0.8)',
                'Unaltered': 'rgba(240, 240, 240, 0.7)'
            }
            BAR_COLOR = 'gray' 
            
            st.subheader("1. % Pure Medium Level (Focus on Near-100% Variation)") 
            Y_MAX_LIMIT_PCT = 200 
            truncated_count = (plot_df['% MP Level'] > Y_MAX_LIMIT_PCT).sum()
            if truncated_count > 0:
                 st.info(f"⚠️ {truncated_count} metabolites (>{Y_MAX_LIMIT_PCT}%) had their values visually truncated.") 

            try:
                fig_pct = go.Figure()
                category_indices = plot_df.groupby('Profile').apply(lambda x: (x.index.min(), x.index.max())).to_dict() 
                shapes_pct = []
                y_min_range_pct, y_max_range_pct = 0, Y_MAX_LIMIT_PCT
                for perfil, (start_index, end_index) in category_indices.items():
                    shapes_pct.append(
                        dict(type="rect", x0=start_index-0.5, y0=y_min_range_pct, x1=end_index+0.5, y1=y_max_range_pct,
                             fillcolor=BACKGROUND_COLORS[perfil], layer="below", line_width=0, xref='x', yref='y')
                    )
                plot_y_pct = plot_df['% MP Level'].clip(upper=Y_MAX_LIMIT_PCT)
                fig_pct.add_trace(go.Bar(
                    x=plot_df['Metabolite'], y=plot_y_pct, name='% MP Level', marker_color=BAR_COLOR, 
                    error_y=dict(type='data', array=plot_df['Error Upper'], arrayminus=plot_df['Error Lower'], 
                                 visible=True, color=BAR_COLOR)
                ))
                fig_pct.add_shape(type="line", xref="paper", yref="y", x0=0, y0=100, x1=1, y1=100,
                                  line=dict(color="black", width=2, dash="dot"), layer="above")
                fig_pct.update_layout(
                    shapes=shapes_pct,
                    yaxis={'type': 'linear', 'range': [y_min_range_pct, y_max_range_pct],
                           'title': {'text': f"Metabolite Level (% of {control_group})", 
                                     'font': {'color': '#4b5563', 'size': 14, 'weight': 'bold'}}},
                    xaxis={'categoryorder': 'array', 'categoryarray': metabolite_order, 'tickangle': 90}, 
                    showlegend=False, height=600, margin=dict(l=50, r=50, t=50, b=150), 
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                )
                for perfil, (start_index, end_index) in category_indices.items():
                     fig_pct.add_annotation(
                        text=perfil, x=(start_index + end_index) / 2, y=Y_MAX_LIMIT_PCT * 0.95, 
                        xref="x", yref="y", showarrow=False, font=dict(size=14, color="#666666"),
                     )
                st.plotly_chart(fig_pct, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error generating Plot 1 (% MP Level). Detail: {e}") 

            st.markdown("---") 
            st.subheader("2. Log2 Fold Change (Focus on Magnitude of Change) and Significance") 
            st.info("The 0-axis represents No Change (Fold Change = 1). **Bar color indicates statistical significance (P-value < 0.05).**") 
            
            try:
                fig_log2 = go.Figure()
                y_min_range_log2 = plot_df['Log2 FC'].min() * 1.1 
                y_max_range_log2 = plot_df['Log2 FC'].max() * 1.1
                max_abs = max(abs(y_min_range_log2), abs(y_max_range_log2))
                y_min_range_log2 = -max_abs * 1.05
                y_max_range_log2 = max_abs * 1.05
                shapes_log2 = []
                for perfil, (start_index, end_index) in category_indices.items():
                    shapes_log2.append(
                        dict(type="rect", x0=start_index-0.5, y0=y_min_range_log2, x1=end_index+0.5, y1=y_max_range_log2,
                             fillcolor=BACKGROUND_COLORS[perfil], layer="below", line_width=0, xref='x', yref='y')
                    )
                color_map_sig = {'Significant (P < 0.05)': 'darkred', 'Not Significant (P ≥ 0.05)': 'gray'} 
                
                for sig_level, color in color_map_sig.items():
                    subset = plot_df[plot_df['Significance'] == sig_level] 
                    if not subset.empty:
                        fig_log2.add_trace(go.Bar(
                            x=subset['Metabolite'], y=subset['Log2 FC'], name=sig_level, marker_color=color,
                            error_y=dict(type='data', array=subset['Log2 Error Upper'], arrayminus=subset['Log2 Error Lower'], 
                                         visible=True, color=color)
                        ))
                fig_log2.add_shape(type="line", xref="paper", yref="y", x0=0, y0=0, x1=1, y1=0,
                                  line=dict(color="black", width=2, dash="dot"), layer="above")
                fig_log2.update_layout(
                    shapes=shapes_log2,
                    yaxis={'type': 'linear', 'range': [y_min_range_log2, y_max_range_log2],
                           'title': {'text': "Magnitude of Change (Log₂ Fold Change)", 
                                     'font': {'color': '#4b5563', 'size': 14, 'weight': 'bold'}}},
                     xaxis={'categoryorder': 'array', 'categoryarray': metabolite_order, 'tickangle': 90}, 
                    showlegend=True, legend_title_text="T-Test Significance", height=600, 
                    margin=dict(l=50, r=50, t=50, b=150), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                )
                for perfil, (start_index, end_index) in category_indices.items():
                     fig_log2.add_annotation(
                        text=perfil, x=(start_index + end_index) / 2, y=y_max_range_log2 * 0.95, 
                        xref="x", yref="y", showarrow=False, font=dict(size=14, color="#666666"),
                     )
                st.plotly_chart(fig_log2, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error generating Plot 2 (Log2 FC). Detail: {e}") 

            st.markdown("---")
            st.subheader("Plot Data for Export") 
            plot_df_indexed = plot_df.set_index('Metabolite')
            stats_only_df = comparison_df[['Metabolite', 'P-value', "Effect Size (Hedges' g)"]].set_index('Metabolite')
            export_df = plot_df_indexed.join(stats_only_df).reset_index()
            export_df.columns = [
                'Metabolite', 'Fold Change', 'Uncertainty (FC)', '% MP Level', 'Uncertainty Upper (%)', 'Uncertainty Lower (%)', 
                'Log2 Fold Change', 'Log2 FC Error Upper', 'Log2 FC Error Lower',
                'Significance (P-value < 0.05)', 'Profile (Consumption/Excretion)', 'P-value (T-Test)', 'Effect Size (Hedges\' g)'
            ]
            st.dataframe(export_df, use_container_width=True)
            st.download_button(
                label="Download Complete Consumption/Excretion Data (CSV)", 
                data=export_df.to_csv(index=False).encode('utf-8'),
                file_name='consumption_excretion_plot_data_complete.csv', mime='text/csv',
            )

    # ----------------------------------------------------
    # --- SECTION 5: TREATMENTS ---
    # ----------------------------------------------------
    elif menu_option == "5. 🔬 Analysis: Treatments": 

        if not st.session_state.get('data_processed', False):
            st.warning("💡 **Please upload, map, and process your data in Sections 1, 2, and 3 first.**") 
            st.stop()
            
        metadata_df = st.session_state['data_metadata']
        analytes_df = st.session_state['data_analytes']
        
        st.header("5. 🔬 Treatment Comparison") 
        st.info("This analysis uses the POST-processed (normalized/scaled) data from Section 3.") 
        
        all_groups = metadata_df['Grupo'].unique().tolist()
        
        required_group_3 = "Cell Control (MCC)" 
        if required_group_3 not in all_groups:
            st.error(f"Error: The group '{required_group_3}' was not found. Go back to Section 1 and map your groups.") 
            st.stop()
            
        treatment_groups = [g for g in all_groups if g not in ["Pure Medium (MP)", "Cell Control (MCC)"]] 
        control_group_treatment = "Cell Control (MCC)" 
        
        if not treatment_groups:
            st.info("No 'Treatment' groups (other than MP and MCC) were found in the loaded data.") 
        else:
            selected_treatment_groups = st.multiselect(
                "Select the Treatment Groups to include in the analyses:", 
                options=treatment_groups,
                default=treatment_groups[0] if treatment_groups else []
            )
            
            if selected_treatment_groups:
                groups_to_compare = [control_group_treatment] + selected_treatment_groups
                filtered_metadata_3 = metadata_df[metadata_df['Grupo'].isin(groups_to_compare)].copy()
                filtered_analytes_3 = analytes_df.loc[filtered_metadata_3.index].copy()
                
                analysis_options_3 = [
                    "5.1 Descriptive Statistics (MCC + Treatments)",
                    "5.2 Comparison and Statistical Tests (vs MCC)",
                    "5.3 Multivariate Analyses", 
                    "5.4 Univariate Analysis (Heatmap)",
                    "5.5 Results Interpretation"
                ]
                selected_analysis_3 = st.radio(
                    "Select Treatment Analysis Step:", 
                    options=analysis_options_3, index=0, horizontal=True
                )
                stats_df_3 = calculate_descriptive_stats(filtered_analytes_3, filtered_metadata_3)

                if selected_analysis_3 == "5.1 Descriptive Statistics (MCC + Treatments)":
                    st.subheader("5.1 Descriptive Statistics") 
                    st.markdown(f"##### Mean and Standard Deviation for {control_group_treatment} and Selected Groups") 
                    st.dataframe(stats_df_3, use_container_width=True)
                    st.download_button(
                        label="Download Descriptive Statistics (Treatments)", 
                        data=stats_df_3.to_csv().encode('utf-8'),
                        file_name='descriptive_statistics_treatments.csv', mime='text/csv',
                    )

                elif selected_analysis_3 == "5.2 Comparison and Statistical Tests (vs MCC)":
                    st.subheader("5.2 Comparison and Statistical Tests (vs. Cell Control)") 
                    st.info("This section compares the selected treatments **only** against the Cell Control (MCC).") 
                    
                    if 'comparison_results_dict_3' not in st.session_state:
                        st.session_state['comparison_results_dict_3'] = {}
                    
                    if st.button("Calculate Selected Comparisons (vs MCC)", key='calc_tratamentos_btn'): 
                        comparison_results_dict = {}
                        with st.spinner('Calculating comparisons and statistical tests...'): 
                            for treatment_group in selected_treatment_groups:
                                st.markdown(f"### 🧪 Comparison: **{treatment_group}** vs **{control_group_treatment}**") 
                                comparison_df_single = _calculate_comparison_table_core(
                                    filtered_analytes_3, filtered_metadata_3, control_group_treatment,
                                    treatment_group, stats_df_3, return_multiindex=False
                                )
                                comparison_results_dict[treatment_group] = comparison_df_single.copy()
                                comparison_df_single = comparison_df_single.rename_axis(index="Metabolite") 
                                st.write("Values in **red** (P-value < 0.05) or **green** (|Hedges' g| > 0.8) indicate significance.") 
                                styled_df_single = style_comparison_table(comparison_df_single)
                                st.dataframe(styled_df_single, use_container_width=True)
                                st.download_button(
                                    label=f"Download Comparison ({treatment_group} vs MCC)", 
                                    data=comparison_df_single.to_csv().encode("utf-8"),
                                    file_name=f"comparison_{treatment_group.lower().replace(' ', '_')}_vs_mcc.csv",
                                    mime="text/csv",
                                )
                            st.session_state['comparison_results_dict_3'] = comparison_results_dict
                            st.success("Comparison Calculations Complete!") 

                    comparison_results_dict = st.session_state.get('comparison_results_dict_3', {})
                    if len(comparison_results_dict) >= 1 and selected_treatment_groups:
                        st.markdown("---")
                        st.subheader("🔍 Cross-Treatment Comparison by Metric (vs MCC)") 
                        metric_options = ['Variation %', 'Fold Change', 'P-value', "Effect Size (Hedges' g)"] 
                        selected_metric = st.selectbox(
                            "Select metric to compare:", options=metric_options, 
                            index=0, key='metric_selector_3' 
                        )
                        comparison_matrix = pd.DataFrame(index=analytes_df.columns) 
                        for treatment in selected_treatment_groups:
                            if treatment in comparison_results_dict:
                                df = comparison_results_dict[treatment]
                                if selected_metric in df.columns:
                                    comparison_matrix[f"{treatment} vs MCC"] = df[selected_metric]
                        comparison_matrix = comparison_matrix.dropna(how='all').rename_axis("Metabolite") 
                        st.markdown(f"#### Table: {selected_metric} across All Treatments (vs MCC)") 
                        format_str = "{:.4f}" if selected_metric == 'P-value' else "{:.2f}"
                        st.dataframe(comparison_matrix.style.format(format_str), use_container_width=True)

                        st.markdown("#### Graphical Visualization of Selected Metric") 
                        max_metabolites = len(comparison_matrix) 
                        top_n = st.slider(
                            "Number of metabolites to display:", min_value=5, 
                            max_value=max(5, min(50, max_metabolites)), 
                            value=max(5, min(15, max_metabolites)), key='metabolite_slider_3'
                        )
                        if selected_metric == 'P-value':
                            matrix_to_plot = comparison_matrix.sort_values(by=comparison_matrix.columns[0], ascending=True, na_position='last').head(top_n)
                        elif selected_metric in ['Fold Change', 'Effect Size (Hedges\' g)']:
                            mean_magnitude = comparison_matrix.apply(lambda x: abs(x).mean(), axis=1).sort_values(ascending=False)
                            matrix_to_plot = comparison_matrix.loc[mean_magnitude.index].head(top_n)
                        else:
                            mean_value = comparison_matrix.mean(axis=1).sort_values(ascending=False)
                            matrix_to_plot = comparison_matrix.loc[mean_value.index].head(top_n)

                        df_melt = (matrix_to_plot.reset_index().melt(id_vars="Metabolite", var_name="Comparison", value_name=selected_metric)) 
                        fig = px.bar(
                            df_melt, x="Metabolite", y=selected_metric, color="Comparison", 
                            barmode="group", title=f"{selected_metric} - Cross-Treatment Comparison (Top {top_n})" 
                        )
                        fig.update_layout(xaxis_tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                    
                elif selected_analysis_3 == "5.3 Multivariate Analyses":
                    st.subheader("5.3 Multivariate Analyses") 
                    st.info(f"These analyses use the POST-processed data from **{control_group_treatment}** and the selected treatments.") 
                    pca_tab, plsda_tab = st.tabs(["📊 PCA", "🔬 PLS-DA (Discriminant Model)"]) 
                    
                    with pca_tab:
                        st.markdown("#### Principal Component Analysis (PCA)") 
                        st.markdown("PCA is *unsupervised*. It shows the natural variation in your data.") 
                        n_samples_pca = len(filtered_analytes_3)
                        n_features_pca = len(filtered_analytes_3.columns)
                        n_comps_max_pca = min(10, n_samples_pca, n_features_pca) 
                        
                        if n_comps_max_pca < 2:
                            st.warning("Cannot perform PCA. At least 2 samples and 2 metabolites are required.") 
                        else:
                            pc_options_pca = [f'PC{i+1}' for i in range(n_comps_max_pca)]
                            st.markdown("##### Select Principal Components") 
                            col_pc1, col_pc2 = st.columns(2)
                            with col_pc1:
                                pc_x_selected = st.selectbox("X-Axis (PCA):", options=pc_options_pca, index=0, key='pc_x_selector_3') 
                            with col_pc2:
                                default_pc_y_index = 1 if n_comps_max_pca > 1 else 0
                                pc_y_selected = st.selectbox("Y-Axis (PCA):", options=pc_options_pca, index=default_pc_y_index, key='pc_y_selector_3') 
                            
                            if pc_x_selected == pc_y_selected:
                                st.error("X and Y axes must be different components.") 
                            else:
                                with st.spinner("Calculating PCA and confidence ellipses..."): 
                                    fig_pca_3, pca_data_3, explained_var_3 = perform_pca_with_ellipses(
                                        filtered_analytes_3, filtered_metadata_3, 
                                        pc_x_selected, pc_y_selected, n_comps_max_pca
                                    )
                                if fig_pca_3:
                                    st.plotly_chart(fig_pca_3, use_container_width=False) 
                                    st.markdown("##### Explained Variance (PCA)") 
                                    col_var1, col_var2 = st.columns(2)
                                    col_var1.metric(f"Explained Variance ({pc_x_selected})", f"{explained_var_3.get(pc_x_selected, 0):.2%}") 
                                    col_var2.metric(f"Explained Variance ({pc_y_selected})", f"{explained_var_3.get(pc_y_selected, 0):.2%}") 
                                    st.subheader("Explained Variance per Component (PCA)") 
                                    var_df_3 = pd.DataFrame({
                                        'Component': pc_options_pca, 
                                        'Explained Variance': [explained_var_3.get(pc, 0) for pc in pc_options_pca], 
                                    })
                                    var_df_3['Cumulative Variance'] = var_df_3['Explained Variance'].cumsum() 
                                    st.dataframe(var_df_3)
                                else:
                                    st.error("Could not generate PCA plot.") 

                    with plsda_tab:
                        st.markdown("#### Partial Least Squares Discriminant Analysis (PLS-DA)") 
                        st.markdown("PLS-DA is *supervised*. It optimizes the separation between the selected groups.") 
                        n_samples_plsda = len(filtered_analytes_3)
                        n_features_plsda = len(filtered_analytes_3.columns)
                        
                        if len(filtered_metadata_3['Grupo'].unique()) < 2:
                            st.warning("PLS-DA requires at least 2 groups for comparison.") 
                        else:
                            n_comps_max_plsda = min(10, n_samples_plsda - 1, n_features_plsda)
                            if n_comps_max_plsda < 1:
                                st.warning("Insufficient data to calculate PLS-DA model.") 
                            else:
                                n_comps_selected_plsda = st.slider(
                                    "Number of Components for the PLS-DA model:", 
                                    min_value=1, max_value=n_comps_max_plsda,
                                    value=min(2, n_comps_max_plsda), key='plsda_n_comps_slider'
                                )
                                comp_options_plsda = [f'Component {i+1}' for i in range(n_comps_selected_plsda)] 

                                with st.spinner("Calculating PLS-DA and Cross-Validation..."): 
                                    scores_df, loadings_df, exp_var_dict, r2y, q2, r2x = perform_plsda_analysis(
                                        filtered_analytes_3, filtered_metadata_3, n_comps_selected_plsda
                                    )
                                
                                if scores_df is not None:
                                    st.subheader("PLS-DA Model Performance") 
                                    st.markdown("- **$R^2X$:** How much of the *metabolite profile* the model captures.\n"
                                                "- **$R^2Y$:** How much of the *group separation* the model explains (Fit).\n"
                                                "- **$Q^2$:** How well the model can *predict* the group (Prediction).") 
                                    col_m1, col_m2, col_m3 = st.columns(3)
                                    col_m1.metric(f"R²X (Explained Variance)", f"{r2x:.3f}") 
                                    col_m2.metric(f"R²Y (Model Fit)", f"{r2y:.3f}") 
                                    col_m3.metric(f"Q² (Predictive Power)", f"{q2:.3f}") 
                                    if q2 < 0.05: st.warning(f"**Overfitting Alert:** The $Q^2$ ({q2:.3f}) is very low.") 

                                    st.markdown("---")
                                    st.markdown("##### PLS-DA: Scores Plot") 
                                    col_plsda_1, col_plsda_2 = st.columns(2)
                                    with col_plsda_1:
                                        comp_x_plsda = st.selectbox("X-Axis (PLS-DA):", options=comp_options_plsda, index=0, key='plsda_x_selector') 
                                    with col_plsda_2:
                                        comp_y_plsda = st.selectbox("Y-Axis (PLS-DA):", options=comp_options_plsda, index=min(1, len(comp_options_plsda)-1), key='plsda_y_selector') 
                                    
                                    if comp_x_plsda == comp_y_plsda and len(comp_options_plsda) > 1:
                                        st.error("Select different components for the X and Y axes.") 
                                    else:
                                        fig_plsda_scores = plot_plsda_scores(scores_df, comp_x_plsda, comp_y_plsda, exp_var_dict)
                                        st.plotly_chart(fig_plsda_scores, use_container_width=False) 

                                    st.markdown("---")
                                    st.markdown("##### PLS-DA: Loading Plot") 
                                    st.markdown("Shows which metabolites are most important for the separation.") 
                                    comp_load_plsda = st.selectbox(
                                        "Select Component to view Loadings:", 
                                        options=comp_options_plsda, index=0, key='plsda_loading_selector'
                                    )
                                    top_n_loadings = st.slider(
                                        "Number of metabolites to display (Top N):", 
                                        min_value=10, max_value=50, value=20, key='plsda_loading_slider'
                                    )
                                    fig_plsda_loadings = plot_plsda_loadings(loadings_df, comp_load_plsda, top_n=top_n_loadings)
                                    st.plotly_chart(fig_plsda_loadings, use_container_width=True)
                                else:
                                    st.error("Could not generate PLS-DA plots.") 
                
                elif selected_analysis_3 == "5.4 Univariate Analysis (Heatmap)":
                    st.subheader("5.4 Univariate Analysis (Heatmap)") 
                    st.markdown("Visualize the average changes of all treatments relative to a reference.") 
                    st.info("The columns (metabolites) are ordered by the consumption/excretion profile (calculated in Section 4).") 
                    metric_heatmap = st.selectbox(
                        "Select Metric for the Heatmap:", 
                        ["Log2 Fold Change", "Variation %"], key='heatmap_metric',
                        help="Log2(FC) is generally preferred for normalized/scaled data." 
                    )
                    try:
                        profile_df = get_metabolite_profile_classification(analytes_df, metadata_df)
                        metabolite_order_sorted = profile_df.index.tolist()
                    except Exception as e:
                        st.error(f"Could not calculate metabolite order: {e}") 
                        metabolite_order_sorted = sorted(analytes_df.columns.tolist()) 
                    
                    tab_mcc, tab_mp = st.tabs(["Reference: Cell Control (MCC)", "Reference: Pure Medium (MP)"])
                    
                    with tab_mcc:
                        st.markdown(f"Comparing **{', '.join(selected_treatment_groups)}** vs. **{control_group_treatment}**") 
                        with st.spinner("Calculating heatmap (Ref: MCC)..."): 
                            metric_data_mcc, pvalue_data_mcc = _prepare_heatmap_data(
                                filtered_analytes_3, filtered_metadata_3, control_group_treatment,
                                selected_treatment_groups, metric_heatmap, metabolite_order_sorted
                            )
                            if not metric_data_mcc.empty:
                                fig_heatmap_mcc = plot_comparison_heatmap(metric_data_mcc, pvalue_data_mcc, metric_heatmap)
                                st.plotly_chart(fig_heatmap_mcc, use_container_width=True)
                            else:
                                st.warning("Could not generate heatmap for MCC reference.") 

                    with tab_mp:
                        reference_group_mp = "Pure Medium (MP)" 
                        comparison_groups_mp = ["Cell Control (MCC)"] + selected_treatment_groups 
                        st.markdown(f"Comparing **{', '.join(comparison_groups_mp)}** vs. **{reference_group_mp}**") 
                        with st.spinner("Calculating heatmap (Ref: MP)..."): 
                            groups_for_mp_comparison = [reference_group_mp] + comparison_groups_mp
                            meta_mp = metadata_df[metadata_df['Grupo'].isin(groups_for_mp_comparison)]
                            analytes_mp = analytes_df.loc[meta_mp.index]
                            metric_data_mp, pvalue_data_mp = _prepare_heatmap_data(
                                analytes_mp, meta_mp, reference_group_mp,
                                comparison_groups_mp, metric_heatmap, metabolite_order_sorted
                            )
                            if not metric_data_mp.empty:
                                fig_heatmap_mp = plot_comparison_heatmap(metric_data_mp, pvalue_data_mp, metric_heatmap)
                                st.plotly_chart(fig_heatmap_mp, use_container_width=True)
                            else:
                                st.warning("Could not generate heatmap for MP reference.") 
                
                elif selected_analysis_3 == "5.5 Results Interpretation":
                    st.subheader("5.5 Results Interpretation (Treatment Effect)") 
                    st.markdown("Combines the **Base Profile** (MCC vs MP) with the **Treatment Effect** (Treatment vs MCC).") 
                    st.warning("Interpretation is based on POST-processed data. If Log or Z-score data was used, interpret the 'Fold Change' with caution.") 
                    
                    comparison_results_dict = st.session_state.get('comparison_results_dict_3', {})
                    if not comparison_results_dict:
                        st.error("⚠️ **No comparison results found!**") 
                        st.info("Please go to the **'5.2'** tab and click the **'Calculate Selected Comparisons (vs MCC)'** button first.") 
                    else:
                        try:
                            base_profile_df = get_metabolite_profile_classification(analytes_df, metadata_df)
                        except Exception as e:
                            st.error(f"Could not calculate base profile (MCC vs MP). Error: {e}") 
                            base_profile_df = pd.DataFrame() 
                        
                        if base_profile_df.empty:
                            st.error("Could not generate interpretation table (base profile not calculated).") 
                        else:
                            available_treatments = list(comparison_results_dict.keys())
                            selected_treatment_for_interp = st.selectbox(
                                "Select the Treatment to interpret:", 
                                options=available_treatments
                            )
                            if selected_treatment_for_interp:
                                treatment_comparison_df = comparison_results_dict[selected_treatment_for_interp]
                                st.markdown(f"#### Interpretation: **{selected_treatment_for_interp}** vs **{control_group_treatment}**") 
                                with st.spinner("Generating interpretation table..."): 
                                    interp_df = generate_interpretation_table(base_profile_df, treatment_comparison_df)
                                    styled_interp_df = style_interpretation_table(interp_df)
                                    st.dataframe(styled_interp_df, use_container_width=True)
                                    st.download_button(
                                        label=f"Download Interpretation ({selected_treatment_for_interp})", 
                                        data=interp_df.to_csv(index=False).encode('utf-8'),
                                        file_name=f'interpretation_{selected_treatment_for_interp.replace(" ", "_")}.csv',
                                        mime='text/csv',
                                    )

    # ----------------------------------------------------
    # --- SECTION 6: ADVANCED (REQUEST 2) ---
    # ----------------------------------------------------
    elif menu_option == "6. 📈 Advanced Analyses": # Changed to match sidebar

        st.header("6. 📈 Advanced Analyses") # Title
        
        # Content replaced with "Coming soon" message
        st.info("This section is under development and will be available soon!") # "Em breve"

    # ----------------------------------------------------
    # --- INITIAL MESSAGE ---
    # ----------------------------------------------------
    else: 
        if menu_option != "1. 📤 Upload and Map": 
            st.info("💡 **To begin, go to the '1. 📤 Upload and Map' section in the sidebar to load your files.**")