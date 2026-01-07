import pandas as pd
import os

import llm_clients
import loaders
import metrics_calculators

def main():
    # Chargement des données
    benchmark_prompts = loaders.load_all_benchmarks("../data/Benchmark_Questions.xlsx")
    models_list = "mistral"

    print("\n========== CHARGEMENT DU BENCHMARK ==========")
    print(f"Nombre total de lignes : {len(benchmark_prompts)}")
    print(benchmark_prompts.head(2).to_string())

    # Paramètres de run
    nb_iter = 3
    nb_questions = 1  # -1 pour tout traiter, >0 pour limiter

    print("\n========== PARAMÈTRES D'EXÉCUTION ==========")
    print(f"Modèles              : {models_list}")
    print(f"Nombre d'itérations  : {nb_iter}")
    print(f"Nombre de questions  : {nb_questions}")

    # Appel des LLMs
    print("\n========== APPEL DES LLM ==========")
    reponses_au_benchmark = process_benchmark_batch(
        benchmark_prompts,
        models=models_list,
        nb_iter=nb_iter,
        nb_questions=nb_questions,
    )

    print("\n---------- APERÇU DES RÉPONSES (brut LLM) ----------")
    print(reponses_au_benchmark.head(2).to_string())

    # Appel du LLM d'embedding
    print("\n========== CALCUL DES EMBEDDINGS ==========")
    reponses_avec_embeddings = compute_embeddings_batch(reponses_au_benchmark)

    print("\n---------- APERÇU DES EMBEDDINGS (taille) ----------")
    print(reponses_avec_embeddings["reponse_emb"].head(2).apply(len))

    # Calcul des métriques
    print("\n========== CALCUL DES MÉTRIQUES ==========")
    reponses_avec_metriques = process_metrics_batches(reponses_avec_embeddings)

    print("\n---------- APERÇU DES RÉPONSES AVEC MÉTRIQUES ----------")
    print(reponses_avec_metriques.head(2).to_string())

    # Sauvegarde
    output_path = f"../data/answers_and_scores_{models_list}_x{nb_iter}.json"
    reponses_avec_metriques.to_json(
        output_path,
        orient="records",
        force_ascii=False,
        indent=2,
    )

    print("\n========== EXPORT TERMINÉ ==========")
    print(f"Fichier écrit : {output_path}")

# ----------------------------------------------------------------------
# -- Fonction qui envoie le benchmark à la fonction d'appel des LLMs ---
# ---------- et génère un tableau des réponses concaténées -------------
# ----------------------------------------------------------------------

def process_benchmark_batch(df_input, models="all", nb_iter=1, nb_questions=-1):
    """
    Prend le DF issu de loaders.load_all_benchmarks et génère les réponses.
    nb_questions = -1 -> traite tout le DataFrame
    nb_questions > 0  -> ne traite que les nb_questions premières lignes
    """
    # Sélection du sous-ensemble de questions
    if nb_questions != -1:
        df_input = df_input.head(nb_questions)

    total_questions = len(df_input)
    print(f"\n[INFO] Questions à traiter : {total_questions} (nb_iter = {nb_iter})")

    results = []

    # On exécute chaque phrase du benchmark nb_iter fois
    for iter_idx in range(nb_iter):
        batch_id = iter_idx + 1
        print(f"\n[ITERATION {batch_id}/{nb_iter}] Début du traitement des questions")

        # On itère sur chaque ligne du tableau de prompts donné en entrée
        for q_idx, (_, row) in enumerate(df_input.iterrows(), start=1):
            # Appel unitaire au(x) modèle(s)
            responses = llm_clients.call_llm(row["Prompt_Texte"], models=models)

            for model_name, text in responses.items():
                # On construit l'ID_réponse unique
                id_res = f"{row['ID_Prompt']}_{model_name}_{batch_id}"

                # LOG de progression sur une seule ligne
                print(
                    f"[iter {batch_id}/{nb_iter}] "
                    f"question {q_idx}/{total_questions} "
                    f"modèle={model_name} "
                    f"ID_Prompt={row['ID_Prompt']} -> réponse OK"
                )

                # On crée l'entrée avec TOUTES les métadonnées utiles
                res_entry = {
                    "ID_reponse": id_res,
                    "ID_Prompt_initial": row["ID_Prompt"],
                    "ID_Question": row["ID_Question"],
                    "num_batch": batch_id,
                    "Categorie": row["Categorie"],
                    "langue_variante": row["Langue_Variante"],
                    "model": model_name,
                    "question_txt": row["Prompt_Texte"],
                    "reponse_txt": text,
                }

                results.append(res_entry)

    return pd.DataFrame(results)

# ----------------------------------------------------------------------
# ------------ Fonction qui envoie l'embedding des réponses ------------
# -------------------- et le concatène au tableau ----------------------
# ----------------------------------------------------------------------
   
def compute_embeddings_batch(df_input):
    """
    Parcourt le DataFrame et génère les embeddings pour chaque réponse.
    """

    df_input["reponse_emb"] = df_input["reponse_txt"].apply(llm_clients.call_embedding_model)
    
    return df_input

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

    # On groupe pour avoir les X langues ensemble pour chaque question ET chaque modèle
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