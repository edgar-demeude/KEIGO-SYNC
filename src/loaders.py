import pandas as pd
import os

def load_excel_all_sheets(file_path):
    """
    Charge un fichier Excel et renvoie un dictionnaire de DataFrames. Clés = noms des onglets, Valeurs = tables Pandas.
    """
    # Vérification si le fichier existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")

    try:
        # En passant sheet_name=None, Pandas lit TOUS les onglets
        dict_dfs = pd.read_excel(file_path, sheet_name=None)
        
        print(f"Succès : {len(dict_dfs)} onglet(s) chargé(s) depuis {file_path}")
        return dict_dfs
    
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier Excel : {e}")
        return {}
