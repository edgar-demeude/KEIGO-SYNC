import pandas as pd
import os

def load_all_benchmarks(file_path):
    """
    Charge tous les onglets du Excel initial, formate avec 1 ligne par prompt.
    Ajoute la catégorie, l'ID Question (commun aux 6 variantes) et l'ID Prompt (unique par ligne).
    """
    all_sheets = pd.read_excel(file_path, sheet_name=None)
    processed_dfs = []

    static_cols = [
        'ID_Question', 'Categorie', 'Biais', 'Comments/Answer_Elements'
    ]
        
    prompt_cols = [
        #'FR_tu', 
        #'FR_vous',
        'JP_Tameguchi', 
        'JP_Teineigo', 
        'JP_Sonkeigo', 
        'EN_Base'
    ]

    sheet_names_to_process = list(all_sheets.keys())[:-1]

    for sheet_name in sheet_names_to_process:
        df = all_sheets[sheet_name].copy()

        df = df.dropna(subset=[c for c in prompt_cols if c in df.columns], how='all')
        
        # A. ID de question (commun aux 6 langues)
        df['ID_Question'] = [f"{sheet_name}_{i+1}" for i in range(len(df))]
        df['Categorie'] = sheet_name
        
        available_static = [c for c in static_cols if c in df.columns]
        
        # B. Transformation "Melt"
        df_long = df.melt(
            id_vars=available_static,
            value_vars=[c for c in prompt_cols if c in df.columns],
            var_name='Langue_Variante', 
            value_name='Prompt_Texte'
        )
        
        # C. Ajout d'un ID unique par prompt (Prompt ID)
        df_long['ID_Prompt'] = df_long['ID_Question'] + "_" + df_long['Langue_Variante']
        
        processed_dfs.append(df_long)

    final_df = pd.concat(processed_dfs, ignore_index=True)
    
    # Réorganiser les colonnes pour avoir les IDs au début (plus lisible)
    cols = ['ID_Prompt', 'ID_Question', 'Categorie', 'Langue_Variante', 'Prompt_Texte']
    other_cols = [c for c in final_df.columns if c not in cols]
    final_df = final_df[cols + other_cols]

    # Réorganiser les lignes, par Catégorie puis num de question avec le PB 10 vient avant 2
    final_df['temp_num'] = final_df['ID_Question'].str.split('_').str[-1].astype(int)
    final_df = final_df.sort_values(by=['Categorie', 'temp_num', 'Langue_Variante'])
    final_df = final_df.drop(columns=['temp_num'])

    return final_df