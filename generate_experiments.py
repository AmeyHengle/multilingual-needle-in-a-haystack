import json
import os
import argparse


def save_dict_to_json(dictionary, output_file):
    """
    Saves a dictionary to a JSON file.

    Args:
        dictionary (dict): The dictionary to be saved.
        output_file (str): The path to the output JSON file.
    """
    with open(output_file, "w") as json_file:
        json.dump(dictionary, json_file, indent=4)


def main(output_file:str):
    ### Experiment Variables

    var_context_lang = ["en", "vi", "ar", "es", "zh", "hi", "de"]
    var_question_lang = ["en", "vi", "ar", "es", "zh", "hi", "de"]
    # var_noise_type = ['monolingual', 'bilingual', 'multilingual']
    var_noise_type = ["en", "vi", "ar", "es", "zh", "hi", "de", "multilingual"]
    # var_retrieval_type = ['related', 'unrelated']
    var_retrieval_type = ["semantic_similarity", "random"]
    var_needle_position = ["start", "end", "middle"]

    # Formulate exhaustive permutations of experiments.
    experiments = []
    for context_lang in var_context_lang:
        for question_lang in var_question_lang:
            for noise_type in var_noise_type:
                for retrieval_type in var_retrieval_type:
                    for needle_position in var_needle_position:
                        experiment_dict = {
                            "context_lang": context_lang,
                            "question_lang": question_lang,
                            "retrieval_type": retrieval_type,
                            "noise_type": noise_type,
                            "needle_position": needle_position,
                        }
                        experiments.append(experiment_dict)

    print(f"Sample Experiment:", experiments[0])
    print(f"Total Experiments:", len(experiments))
    save_dict_to_json(experiments, output_file)
    print(f"Dictionary saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save dictionary to JSON file")
    parser.add_argument(
        "--output_file", "-o", type=str, required=True, help="Output JSON file path"
    )
    args = parser.parse_args()
    main(output_file=args.output_file)


"""
Sample usage
python generate_experiments.py --output_file=/home/amey/long-context-llms/data/experiments.json

"""

