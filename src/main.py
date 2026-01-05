import pandas as pd
import os


import llm_clients
import loaders
import metrics_calculators

def main() :
    # Chargement des données
    benchmark_prompts = loaders.load_all_benchmarks("../data/Benchmark_Questions.xlsx")
    models_list = "gemini"

    print("\n--- Vérifications du loading ---")
    print(benchmark_prompts.head(2).to_string())

    # Appel des LLMs
    reponses_au_benchmark = process_benchmark_batch(benchmark_prompts, models = models_list, nb_iter = 2)

    print("\n--- print des réponses ---")
    print(reponses_au_benchmark.head(2).to_string())

    # Calcul des métriques

    reponses_avec_metriques = process_metrics_batches(reponses_au_benchmark)
    print("\n--- print des réponses avec métriques ---")
    print(reponses_avec_metriques.head(2).to_string())

    # Sauvegarde 
    reponses_avec_metriques.to_csv(f"../data/answers_and_scores_{models_list}.csv", index=False, encoding='utf-8-sig', sep=';')
    print(f"Export terminé dans le dossier data")

# ----------------------------------------------------------------------
# -- Fonction qui envoie le benchmark à la fonction d'appel des LLMs ---
# ---------- et génère un tableau des réponses concaténées -------------
# ----------------------------------------------------------------------

def process_benchmark_batch(df_input, models="all", nb_iter = 1):
    """
    Prend le DF issu de loaders.load_all_benchmarks et génère les réponses.
    """
    results = []
    
    # On exécute chaque phrase du benchmark nb_iter fois
    for iter in range(nb_iter): 
        # On itère sur chaque ligne du tableau de prompts donné en entrée
        for _, row in df_input.iterrows():
            # Appel unitaire au(x) modèle(s)
            responses = llm_clients.call_llm(row['Prompt_Texte'], models=models)
            
            for model_name, text in responses.items():
                # On construit l'ID_réponse unique
                # ex: Science_1_JP_sonkeigo_openai_2
                id_res = f"{row['ID_Prompt']}_{model_name}_{iter +1}"
                num_batch = iter + 1
                
                # On crée l'entrée avec TOUTES les métadonnées utiles
                res_entry = {
                    "ID_reponse": id_res,
                    "ID_Prompt_initial": row['ID_Prompt'],
                    "ID_Question": row['ID_Question'], # Gardé pour le group_by
                    "num_batch": num_batch,
                    "Categorie": row['Categorie'],     # Gardé pour les stats
                    "langue_variante": row['Langue_Variante'],
                    "model": model_name,
                    "question_txt": row['Prompt_Texte'],
                    "reponse_txt": text
                }
                results.append(res_entry)
                
    return pd.DataFrame(results)

# ----------------------------------------------------
# -------- Fonction de calcul des métriques ----------
# ------- Par bacth de 6 car 6 "traductions" ---------
# ----------------------------------------------------

def process_metrics_batches(df_responses):
    """
    Prend le DataFrame des réponses, le groupe par Question et Modèle.
    Les 6 réponses d'une question (différentes "langues") envoyées en même temps 
    pour pouvoir calculer les scores relatifs.
    Renvoie le tableau complété.
    """
    all_scored_groups = []

    # On groupe pour avoir les 6 langues ensemble pour chaque question ET chaque modèle
    groups = df_responses.groupby(['ID_Question', 'model', 'num_batch'])

    print(f"Calcul des métriques pour {len(groups)} batchs...")

    for _, group in groups:
        # group est un DataFrame de 6 lignes
        # On passe ce mini-tableau à notre fonction de calcul
        scored_group = metrics_calculators.compute_metrics_for_batch(group)
        all_scored_groups.append(scored_group)

    # On réassemble tout le monde
    df_final = pd.concat(all_scored_groups, ignore_index=True)
    return df_final

if __name__ == "__main__":
    main()