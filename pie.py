import os
import glob
#from openai import OpenAI
import json
#import random

#import numpy as np
#import torch
#from tqdm import tqdm
#from sentence_transformers import SentenceTransformer
#import pandas as pd
#import faiss
#from time import sleep
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize

#random.seed = 42

from gem5 import simulator

with open("trustml_creds.json") as fp:
    creds = json.load(fp)
ORGANIZATION = creds["ORGANIZATION"]
PROJECT_KEY = creds["OPEN_AI_KEY"]

def load_programs_and_explanations():
    """
    Load the programs and GPT-4 explanations from the JSON files.

    Returns:
        tuple[dict, dict, dict, dict, dict, dict, dict]: Tuple containing the source programs, target programs, generated programs, explanations of why target is faster than source, explanations of why generated is faster than source, explanations of why generated is faster than target, and explanations of the programs in natural language.
    """
    with open("src_progs.json", "r") as fp:
        src_progs = json.load(fp)
    with open("target_progs.json", "r") as fp:
        target_progs = json.load(fp)
    with open("gen_progs.json", "r") as fp:
        gen_progs = json.load(fp)
    with open("explanation_tgt_src.json", "r") as fp:
        explanation_tgt_src = json.load(fp)
    with open("explanation_gen_src.json", "r") as fp:
        explanation_gen_src = json.load(fp)
    with open("explanation_gen_tgt.json", "r") as fp:
        explanation_gen_tgt = json.load(fp)
    with open("src_prog_exps.json", "r") as fp:
        src_prog_exps = json.load(fp)
    return (src_progs, target_progs, gen_progs, explanation_tgt_src, explanation_gen_src, explanation_gen_tgt, src_prog_exps)

def get_problem_id(src_id: str):
    """
    Get the problem ID from the source code ID.

    Args:
        src_id (str): source code ID of src_code

    Returns:
        str: problem ID which can be used for retrieving test cases
    """
    with open("PIE_dataset/test.jsonl") as testf:
        testdata = [json.loads(line) for line in testf]
        for testd in testdata:
            if testd["src_id"] == src_id:
                return (str(testd["problem_id"]), testd["tests"])
    return ("Not Found", ["Not Found"])

def check_generated_progs(p_decomp_folder: str = "program_decompositions", natlang_decomp_folder: str = "natlang_decompositions", cpp_folder: str = "cpp_folder", all_test_case_folder: str = "merged_test_cases") -> None:
    """
    Check the generated programs for correctness and performance, and store the values in a JSON file.

    Args:
        p_decomp_folder (str, optional): Folder containing the generated programs. Defaults to "program_decompositions".
    """
    src_progs, target_progs, gen_progs, explanation_tgt_src, explanation_gen_src, explanation_gen_tgt, src_prog_exps = load_programs_and_explanations()
    env = simulator.make(timeout_seconds_gem5=120, verbose=True, use_logical_cpus=True, port=80, workers=80, exit_early_on_fail=True)
    problem_id_dict = {}
    test_cases_dict = {}
    code_list = []
    test_cases_list = []
    problem_id_list = []
    for key in src_progs.keys():
       with open(f"{natlang_decomp_folder}/{key}_natlang_dec_src_tgt.json", 'r') as fp:
            steps_dict = json.load(fp)
            for step in steps_dict.keys():
                prog = json.load(open(f"{p_decomp_folder}/{key}_{step}_decomposition.json", 'r'))["optimized_code"]
                problem_id, test_cases = get_problem_id(key.split("_")[0])
                problem_id_dict[key] = problem_id
                test_cases_dict[key] = test_cases
                if problem_id!="Not Found":
                    problem_id_list.append(problem_id)
                    code_list.append(prog)
                    test_cases_list.append(test_cases[:6])
                    print(problem_id, test_cases)
    results = env.submit_multiple_single_submissions(code_list, test_cases_list, problem_id_list, "gem5")  
    with open("problem_id_dict.json", "w") as f1:
        json.dump(problem_id_dict, f1)        
    with open("test_cases_dict.json", "w") as f2:
        json.dump(test_cases_dict, f2)
    with open("five_testcase_result.txt", "w") as f3:
        f3.write(str(results))
    with open("five_testcase_result.json", "w") as f4:
        dict_results = [result.to_dict() for result in results]
        json.dump(dict_results, f4)

def generate_comparative_perf_edits(testcase_result_json: str = "five_testcase_result.json", problem_id_json: str = "problem_id_dict.json", natlang_decomp_folder: str = "natlang_decompositions", p_decomp_folder: str = "program_decompositions"):
    src_progs, target_progs, gen_progs, explanation_tgt_src, explanation_gen_src, explanation_gen_tgt, src_prog_exps = load_programs_and_explanations()
    with open(testcase_result_json, "r") as fp:
        results = json.load(fp)
    with open(problem_id_json, "r") as fp:
        problem_id_dict = json.load(fp)
    prog_list = []
    for key in src_progs.keys():
       with open(f"{natlang_decomp_folder}/{key}_natlang_dec_src_tgt.json", 'r') as fp:
            steps_dict = json.load(fp)
            for step in steps_dict.keys():
                prog = json.load(open(f"{p_decomp_folder}/{key}_{step}_decomposition.json", 'r'))["optimized_code"]
                prog_list.append([prog, problem_id_dict[key], step, key])
    for result_ind in range(len(results)):
        prog_list[result_ind].extend([results[result_ind]["mean_acc"], results[result_ind]["agg_runtime"]])
    accuracy_dict = {}
    perf_dict = {}
    for result_list in prog_list:
        if result_list[3] not in accuracy_dict:
            accuracy_dict[result_list[3]] = {str(result_list[2]): result_list[4]}
            perf_dict[result_list[3]] = {str(result_list[2]): result_list[5]}
        else:
            accuracy_dict[result_list[3]][str(result_list[2])] = result_list[4]
            perf_dict[result_list[3]][str(result_list[2])] = result_list[5]
    wrong = 0
    correctly_optimized = 0
    for program in accuracy_dict.keys():
        correct = True
        for step in accuracy_dict[program].keys():
            if accuracy_dict[program][step] != 1.0:
                wrong += 1
                correct = False
                break
        if correct:
            max_step = 0
            correct_perf = True
            for step in perf_dict[program].keys():
                max_step = max(max_step, int(step))
            for step in range(1, max_step):
                if perf_dict[program][str(step)] < perf_dict[program][str(step+1)]: #performance in earlier step is strictly better than post-optimization
                    correct_perf = False
                    break
            if correct_perf:
                correctly_optimized += 1
    print(f"Number of programs whose optimizations are eventually inaccurate on some test case: {wrong}/{len(accuracy_dict)}")
    print(f"Number of programs whose optimizations are both accurate and in increasing order of optimization: {correctly_optimized}/{len(accuracy_dict)-wrong}")
    with open("accuracy_dict.json", "w") as f1:
        json.dump(accuracy_dict, f1)
    with open("perf_dict.json", "w") as f2:
        json.dump(perf_dict, f2)
    with open("final_result.json", "w") as f3:
        json.dump(prog_list, f3)
        

if __name__ == "__main__":
    check_generated_progs()
    generate_comparative_perf_edits()