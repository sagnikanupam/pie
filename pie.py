import os
import glob
from openai import OpenAI
import json
import random

import numpy as np
from tqdm import tqdm
import pandas as pd
from time import sleep

random.seed = 42

from gem5 import simulator

with open("trustml_creds.json") as fp:
    creds = json.load(fp)
ORGANIZATION = creds["ORGANIZATION"]
PROJECT_KEY = creds["OPEN_AI_KEY"]

def load_programs_and_explanations(src_progs_file: str = "src_progs.json", target_progs_file: str = "target_progs.json", gen_progs_file: str = "gen_progs.json", explanation_tgt_src_file: str = "explanation_tgt_src.json", explanation_gen_src_file: str = "explanation_gen_src.json", explanation_gen_tgt_file: str = "explanation_gen_tgt.json", src_prog_exps_file: str = "src_prog_exps.json"):
    """
    Load the programs and GPT-4 explanations from the JSON files.

    Returns:
        tuple[dict, dict, dict, dict, dict, dict, dict]: Tuple containing the source programs, target programs, generated programs, explanations of why target is faster than source, explanations of why generated is faster than source, explanations of why generated is faster than target, and explanations of the programs in natural language.
    """
    with open(src_progs_file, "r") as fp:
        src_progs = json.load(fp)
    with open(target_progs_file, "r") as fp:
        target_progs = json.load(fp)
    with open(gen_progs_file, "r") as fp:
        gen_progs = json.load(fp)
    with open(explanation_tgt_src_file, "r") as fp:
        explanation_tgt_src = json.load(fp)
    with open(explanation_gen_src_file, "r") as fp:
        explanation_gen_src = json.load(fp)
    with open(explanation_gen_tgt_file, "r") as fp:
        explanation_gen_tgt = json.load(fp)
    with open(src_prog_exps_file, "r") as fp:
        src_prog_exps = json.load(fp)
    return (src_progs, target_progs, gen_progs, explanation_tgt_src, explanation_gen_src, explanation_gen_tgt, src_prog_exps)

def get_programs_and_explanations(pfolder: str = "src_target_gpt_summary", type: str = "html"):
    """
    Gets the programs and GPT-4 explanations from the folder containing the PIE dataset programs + explanations

    Args:
        pfolder (str): PIE dataset folder path
        type (str, optional): Whether to extract from HTML or some other source. Defaults to "html".
        
    Returns:
        tuple[dict, dict, dict, dict, dict, dict, dict]: Tuple containing the source programs, target programs, generated programs, explanations of why target is faster than source, explanations of why generated is faster than source, explanations of why generated is faster than target, and explanations of the programs in natural language.
    """
    target_progs = {}
    src_progs = {}
    gen_progs = {}
    explanation_tgt_src = {}
    explanation_gen_src = {}
    explanation_gen_tgt = {}
    src_prog_exps = {}
    if type=="html":
        html_files = [f for f in os.listdir(pfolder) if os.path.isfile(os.path.join(pfolder, f))]
        for file in html_files:
            with open(os.path.join(pfolder, file), "r") as f:
                html = f.read()
                src_prog = html.split("Source Program</h2>")[1].split("</div>")[0].replace("<br>", "\n").replace("<pre>", "").replace("</pre>", "")
                target_prog = html.split("Target Program</h2>")[1].split("</div>")[0].replace("<br>", "\n").replace("<pre>", "").replace("</pre>", "")
                gen_prog = html.split("Generated Program</h2>")[1].split("</div>")[0].replace("<br>", "\n").replace("<pre>", "").replace("</pre>", "") 
                tgt_src_exp = html.split("Explanation of why the tgt program is faster than the source program:</h3>")[1].split("</div>")[0].replace("<p>", "").replace("</p>", "")
                gen_src_exp = html.split("Explanation of why the generated program is faster than the source program:</h3>")[1].split("</div>")[0].replace("<p>", "").replace("</p>", "")
                gen_tgt_exp = html.split("Explanation of why the generated program is faster than the target program:</h3>")[1].split("</div>")[0].replace("<p>", "").replace("</p>", "")
                prog_exp = html.split("Explanation of what the program does in natural language:</h3>")[1].split("</div>")[0].replace("<p>", "").replace("</p>", "")
                src_progs[file] = src_prog
                target_progs[file] = target_prog
                gen_progs[file] = gen_prog
                explanation_tgt_src[file] = tgt_src_exp
                explanation_gen_src[file] = gen_src_exp
                explanation_gen_tgt[file] = gen_tgt_exp
                src_prog_exps[file] = prog_exp
                print(f"Program {file} extracted.")
        with open("src_progs.json", "w") as fp:
            json.dump(src_progs, fp)
        with open("target_progs.json", "w") as fp:
            json.dump(target_progs, fp)
        with open("gen_progs.json", "w") as fp:
            json.dump(gen_progs, fp)
        with open("explanation_tgt_src.json", "w") as fp:
            json.dump(explanation_tgt_src, fp)
        with open("explanation_gen_src.json", "w") as fp:
            json.dump(explanation_gen_src, fp)
        with open("explanation_gen_tgt.json", "w") as fp:
            json.dump(explanation_gen_tgt, fp)
        with open("src_prog_exps.json", "w") as fp:
            json.dump(src_prog_exps, fp)
    return (src_progs, target_progs, gen_progs, explanation_tgt_src, explanation_gen_src, explanation_gen_tgt, src_prog_exps)

def llm_call(final_prompt_string: str, index: str, json_output_folder: str, suffix: str, model: str = "gpt-4o") -> None:
    """
    Calls the LLM model with the final prompt string.

    Args:
        final_prompt_string (str): String passed as prompt to the LLM model.index (str): Index that uniquely identifies record in dataset being iterated over.
        json_output_folder (str): Folder to save the JSON output.
        suffix (str): Suffix to append to the JSON output file.
    """
    MODEL = model
    client = OpenAI(api_key=PROJECT_KEY, organization=ORGANIZATION)
    if model.find("o1") != -1:
        response = client.chat.completions.create(
        model = MODEL,
        messages= [{"role": "user", "content": final_prompt_string}]
        )
        print(f"LLM call for {index} succeeded with response {response}.")
        if (response.choices[0].message.content != None):
            with open(f"{json_output_folder}/{index}_{suffix}.txt", 'w') as fp:
                fp.write(response.choices[0].message.content)
    else:
        response = client.chat.completions.create(
        model = MODEL,
        response_format={ "type": "json_object" },
        messages=[{"role": "system", "content": final_prompt_string}] 
        )
        print(f"LLM call for {index} succeeded with response {response}.")
        if (response.choices[0].message.content != None):
            with open(f"{json_output_folder}/{index}_{suffix}.json", 'w') as fp:
                loaded_response = json.loads(response.choices[0].message.content)
                json.dump(loaded_response, fp)
    sleep(1)

def decompose_exps(natlang_decomp_folder: str = "natlang_decompositions", src_progs_file: str = "src_progs.json", model: str = "gpt-4o") -> None:
    """
    Decompose the natural language description of performance edits to JSON-style lists. Must run get_programs_and_explanations() prior to running this function.

    Args:
        natlang_decomp_folder (str, optional): Folder for saving LLM call outputs for natural language decompositions. Defaults to "natlang_decompositions".
    """
    src_progs, target_progs, gen_progs, explanation_tgt_src, explanation_gen_src, explanation_gen_tgt, src_prog_exps = load_programs_and_explanations(src_progs_file=src_progs_file)
    print("Generating natural language decompositions...")
    if (model.find("o1") != -1):
        with open("decompose_nat_lang_not_preexisting.txt", "r") as f:
            base_prompt = f.read()
            for key, _ in tqdm(src_progs.items()):
                prompt_string = base_prompt + " Natural Language Description of Edits: \n" + explanation_tgt_src[key] + "\n Source Program: \n" + src_progs[key] + "\n Target Program: \n" + target_progs[key]
                llm_call(prompt_string, key, natlang_decomp_folder, "natlang_dec_src_tgt", model) 
    else:
        with open("decompose_nat_lang_preexisting.txt", "r") as f:
            base_prompt = f.read()
            for key, _ in tqdm(src_progs.items()):
                prompt_string = base_prompt + " Natural Language Description of Edits: \n" + explanation_tgt_src[key] + "\n Source Program: \n" + src_progs[key] + "\n Target Program: \n" + target_progs[key]
                llm_call(prompt_string, key, natlang_decomp_folder, "natlang_dec_src_tgt", model)

def generate_intermediates(natlang_decomp_folder: str = "natlang_decompositions", p_decomp_folder: str = "program_decompositions", src_progs_file: str = "src_progs.json", model="gpt-4o") -> None:
    """
    Use GPT to generate decompositions of the programs in the PIE dataset using the steps discovered in the natural language decompositions.  Must run get_programs_and_explanations() and decompose_exps() prior to running this function.
    
    Args:
        natlang_decomp_folder (str, optional): Folder containing the natural language decompositions produced by decompose_exps(). Defaults to "natlang_decompositions".
        p_decomp_folder (str, optional): Folder to save the program decompositions. Defaults to "program_decompositions".
    """
    src_progs, target_progs, gen_progs, explanation_tgt_src, explanation_gen_src, explanation_gen_tgt, src_prog_exps = load_programs_and_explanations(src_progs_file=src_progs_file)
    print("Generating program decompositions...")
    #error_dict_tmp = {"s962695246_spd=50.20986204921044_acc=1.0.html": 1, " s394872425_spd=50.224882520959994_acc=1.0.html": 2, "s735918365_spd=27.299248624779842_acc=1.0.html": 4, "s871862468_spd=50.395747586140544_acc=1.0.html": 4, "s717194930_spd=27.43189016177244_acc=1.0.html": 2, "s437790328_spd=48.00901188136315_acc=1.0.html": 3, "s766949236_spd=43.521725831102984_acc=1.0.html": 7}
    #error_dict_tmp ={"s717194930_spd=27.43189016177244_acc=1.0.html": 2, "s735918365_spd=27.299248624779842_acc=1.0.html": 11}
    error_dict = {}
    with open("decompose_prog.txt", "r") as f:
        base_prompt = f.read()
        for key, _ in tqdm(src_progs.items()):
        #for key, _ in tqdm(error_dict_tmp.items()):
            try:
                with open(f"{natlang_decomp_folder}/{key}_natlang_dec_src_tgt.json", 'r') as fp:
                    steps_dict = json.load(fp)
                    step_list = [int(s) for s in steps_dict.keys()]
                    step_list.sort()
                    start_prog = src_progs[key]
                    # if error_dict_tmp[key] == 1:
                    #     start_prog = json.load(open(f"{p_decomp_folder}/{key}_{error_dict_tmp[key]-1}_decomposition.json", 'r'))["optimized_code"]
                    # for step in range(error_dict_tmp[key], max(step_list)+1):
                    for step in step_list:
                        optimization = steps_dict[str(step)]
                        if model.find("o1") != -1:
                            base_prompt = "You are an expert programmer who needs to optimize a given program. You are given the description of the optimization to be performed as well as the source code of the program. Rewrite the source code in a way that incorporates the optimization, and the rewritten code. Do not output anything other than C++ code."
                            prompt_string = base_prompt + " Source Program: \n" + start_prog + "\n Optimization: \n" + optimization
                            llm_call(prompt_string, key, p_decomp_folder, f"{step}_decomposition", model)
                            if f"{p_decomp_folder}/{key}_{step}_decomposition.txt" not in glob.glob(f"{p_decomp_folder}/*.txt"):
                                error_dict[key] = step
                            start_prog = open(f"{p_decomp_folder}/{key}_{step}_decomposition.txt", 'r').read()
                        else:
                            prompt_string = base_prompt + " Source Program: \n" + start_prog + "\n Optimization: \n" + optimization
                            llm_call(prompt_string, key, p_decomp_folder, f"{step}_decomposition", model)
                            if f"{p_decomp_folder}/{key}_{step}_decomposition.json" not in glob.glob(f"{p_decomp_folder}/*.json"):
                                error_dict[key] = step
                            start_prog = json.load(open(f"{p_decomp_folder}/{key}_{step}_decomposition.json", 'r'))["optimized_code"]
            except Exception as e:
                print(f"Error in generating decompositions for {key}: {e}")
                continue
            
def identify_problem_ids_tests() -> None:   
    """
    Identify the problem IDs and test cases for each program in the PIE dataset and store them in JSON files. Must run get_programs_and_explanations() prior to running this function.
    """
    
    src_progs, target_progs, gen_progs, explanation_tgt_src, explanation_gen_src, explanation_gen_tgt, src_prog_exps = load_programs_and_explanations()
    problem_id_dict = {}
    test_cases_dict = {}
    for key in src_progs.keys():
        problem_id, test_cases = get_problem_id(key.split("_")[0])
        problem_id_dict[key] = problem_id
        test_cases_dict[key] = test_cases
    with open("problem_id_dict.json", "w") as f1:
        json.dump(problem_id_dict, f1)        
    with open("test_cases_dict.json", "w") as f2:
        json.dump(test_cases_dict, f2)
 
def o1_prog_txt_to_json_conversion(natlang_decomp_folder: str = "natlang_decompositions", p_decomp_folder: str = "program_decompositions", src_progs_file: str = "o1_src_progs.json"):
    """
    Convert the text files generated by o1 to JSON files. Must run generate_intermediates() prior to running this function.
    
    Args:
        natlang_decomp_folder (str, optional): Folder containing o1's natural language decompositions in JSON format. Defaults to "natlang_decompositions".
        p_decomp_folder (str, optional): Folder containing o1's decompositions in .txt format, and also output folder for .json programs. Defaults to "program_decompositions".
        src_progs_file (str, optional): Contains the source programs optimized by the o1 model. Defaults to "src_progs.json".
    """
    
    src_progs, target_progs, gen_progs, explanation_tgt_src, explanation_gen_src, explanation_gen_tgt, src_prog_exps = load_programs_and_explanations(src_progs_file=src_progs_file)
    print("Converting o1 program decompositions...")
    for key, _ in tqdm(src_progs.items()):
        with open(f"{natlang_decomp_folder}/{key}_natlang_dec_src_tgt.json", 'r') as fp:
            steps_dict = json.load(fp)
            step_list = [int(s) for s in steps_dict.keys()]
            step_list.sort()
            for step in step_list:
                code_lines = open(f"{p_decomp_folder}/{key}_{step}_decomposition.txt", 'r').readlines()
                actual_code = code_lines[1:-1]
                code_dict = {"optimized_code": " \n ".join(actual_code)}
                with open(f"{p_decomp_folder}/{key}_{step}_decomposition.json", 'w') as fp:
                    json.dump(code_dict, fp)

def o1_natlang_txt_to_json_conversion(natlang_decomp_folder: str = "natlang_decompositions", src_progs_file: str = "src_progs.json"):
    """
    Convert the text files generated by o1 to JSON files. Must run decompose_exps() prior to running this function.
    
    Args:
        natlang_decomp_folder (str, optional): Folder containing o1's natural language decompositions in .txt format, and also output folder for .json programs. Defaults to "natlang_decompositions".
        src_progs_file (str, optional): Contains the source programs optimized by the o1 model. Defaults to "src_progs.json".
    """
    
    src_progs, target_progs, gen_progs, explanation_tgt_src, explanation_gen_src, explanation_gen_tgt, src_prog_exps = load_programs_and_explanations(src_progs_file=src_progs_file)
    print("Converting o1 natural language decompositions...")
    for key, _ in tqdm(src_progs.items()):
        code_lines = open(f"{natlang_decomp_folder}/{key}_natlang_dec_src_tgt.txt", 'r').readlines()
        actual_code = code_lines[1:-1]
        json_string = "".join(actual_code)
        with open(f"{natlang_decomp_folder}/{key}_natlang_dec_src_tgt.json", 'w') as fp:
            fp.write(json_string)
            
def check_generated_progs(p_decomp_folder: str = "program_decompositions", natlang_decomp_folder: str = "natlang_decompositions", src_progs_file: str = "src_progs.json") -> None:
    """
    Check the generated programs for correctness and performance, and store the values in a JSON file. Must run get_programs_and_explanations(), decompose_exps(), generate_intermediates() prior to running this function.

    Args:
        p_decomp_folder (str, optional): Folder containing the generated decompositions of programs by calling generate_intermediates. Defaults to "program_decompositions".
        natlang_decomp_folder (str, optional): Folder containing the natural language decompositions produced by decompose_exps(). Defaults to "natlang_decompositions".
    """
    src_progs, target_progs, gen_progs, explanation_tgt_src, explanation_gen_src, explanation_gen_tgt, src_prog_exps = load_programs_and_explanations(src_progs_file=src_progs_file)
    env = simulator.make(timeout_seconds_gem5=120, verbose=True, use_logical_cpus=True, port=80, workers=80, exit_early_on_fail=True)
    code_list = []
    test_cases_list = []
    problem_id_list = []
    for key, _ in tqdm(src_progs.items()):
       with open(f"{natlang_decomp_folder}/{key}_natlang_dec_src_tgt.json", 'r') as fp:
            with open(f"problem_id_dict.json", 'r') as fp2:
                problem_id_dict = json.load(fp2)
                with open(f"test_cases_dict.json", 'r') as fp3:
                    test_cases_dict = json.load(fp3)
                    steps_dict = json.load(fp)
                    problem_id = problem_id_dict[key]
                    test_cases = test_cases_dict[key]
                    
                    #Add source program at very start
                    problem_id_list.append(problem_id)
                    code_list.append(src_progs[key])
                    test_cases_list.append(test_cases[:6])
                    
                    #Add decomposition steps
                    for step in steps_dict.keys():
                        prog = json.load(open(f"{p_decomp_folder}/{key}_{step}_decomposition.json", 'r'))["optimized_code"]
                        problem_id_list.append(problem_id)
                        code_list.append(prog)
                        test_cases_list.append(test_cases[:6])
                    
                    #Add target program at end
                    problem_id_list.append(problem_id)
                    code_list.append(target_progs[key])
                    test_cases_list.append(test_cases[:6])
                    
    results = env.submit_multiple_single_submissions(code_list, test_cases_list, problem_id_list, "gem5")  
    with open("six_testcase_result.txt", "w") as f1:
        f1.write(str(results))
    with open("six_testcase_result.json", "w") as f2:
        dict_results = [result.to_dict() for result in results]
        json.dump(dict_results, f2)

def evaluate_comparative_perf_edits(testcase_result_json: str = "six_testcase_result.json", problem_id_json: str = "problem_id_dict.json", natlang_decomp_folder: str = "natlang_decompositions", p_decomp_folder: str = "program_decompositions", src_progs_file: str = "src_progs.json"):
    """
    Generate human-readable JSONs from PIE result files. Must run check_generated_progs() prior to running this function.

    Args:
        testcase_result_json (str, optional): Result of running decompositions on testcases using gem5. Defaults to "five_testcase_result.json".
        problem_id_json (str, optional): JSON file containing problem ids of each source code. Defaults to "problem_id_dict.json".
        natlang_decomp_folder (str, optional): Folder containing natural_language decompositions of the source code. Defaults to "natlang_decompositions".
        p_decomp_folder (str, optional): Program decomposition folder containing program decompositions. Defaults to "program_decompositions".
    """
    src_progs, target_progs, gen_progs, explanation_tgt_src, explanation_gen_src, explanation_gen_tgt, src_prog_exps = load_programs_and_explanations(src_progs_file=src_progs_file)
    with open(testcase_result_json, "r") as fp:
        results = json.load(fp)
    with open(problem_id_json, "r") as fp:
        problem_id_dict = json.load(fp)
    prog_list = []
    for key in src_progs.keys():
       with open(f"{natlang_decomp_folder}/{key}_natlang_dec_src_tgt.json", 'r') as fp:
            steps_dict = json.load(fp)
            #Add source program at very start
            prog_list.append([src_progs[key], problem_id_dict[key], 0, key])
            
            #Add decomposition steps 
            for step in steps_dict.keys():
                prog = json.load(open(f"{p_decomp_folder}/{key}_{step}_decomposition.json", 'r'))["optimized_code"]
                prog_list.append([prog, problem_id_dict[key], step, key])
            
            #Add target program at end
            prog_list.append([target_progs[key], problem_id_dict[key], len(steps_dict.keys())+1, key])

    for result_ind in range(len(results)):
        prog_list[result_ind].extend([results[result_ind]["mean_acc"], results[result_ind]["agg_runtime"], results[result_ind]["errors"]])
    accuracy_dict = {}
    perf_dict = {}
    error_dict = {}
    summary_src_target_best = {}
    for result_list in prog_list:
        if result_list[3] not in accuracy_dict:
            accuracy_dict[result_list[3]] = {str(result_list[2]): result_list[4]}
            perf_dict[result_list[3]] = {str(result_list[2]): result_list[5]}
            error_dict[result_list[3]] = {str(result_list[2]): result_list[6]}
        else:
            accuracy_dict[result_list[3]][str(result_list[2])] = result_list[4]
            perf_dict[result_list[3]][str(result_list[2])] = result_list[5]
            error_dict[result_list[3]][str(result_list[2])] = result_list[6]
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
            for step in range(1, max_step-1): #-1 to exclude target program, we only check performance for src >= ....decomps >= final_decomp
                if perf_dict[program][str(step)] < perf_dict[program][str(step+1)]: #performance in earlier step is strictly better than post-optimization
                    correct_perf = False
                    break
            if correct_perf:
                correctly_optimized += 1
    decomp_over_target = 0
    for program in perf_dict.keys():
        src_perf = perf_dict[program]["0"]
        best_perf = perf_dict[program]["1"]
        for step in range(1, len(perf_dict[program])-1):
            if perf_dict[program][str(step)] != "Infinity":
                if best_perf == "Infinity" or best_perf > perf_dict[program][str(step)]:
                    best_perf = perf_dict[program][str(step)]
        target_perf = perf_dict[program][str(len(perf_dict[program])-1)]
        print(f"Program: {program}, Source Performance: {src_perf}, Best Performance: {best_perf}, Target Performance: {target_perf}")
        summary_src_target_best[program] = {"source": src_perf, "best": best_perf, "target": target_perf}
        if target_perf > best_perf:
            decomp_over_target += 1
    print(f"Number of programs for whom at least one decomposition is better than the target performance: {decomp_over_target}/{len(perf_dict)}")
    print(f"Number of programs whose optimizations are eventually inaccurate on some test case: {wrong}/{len(accuracy_dict)}")
    print(f"Number of programs whose optimizations are both accurate and in increasing order of optimization: {correctly_optimized}/{len(accuracy_dict)-wrong}")
    with open("src_target_best_perf.json", "w") as f:
        json.dump(summary_src_target_best, f)
    with open("accuracy_dict.json", "w") as f1:
        json.dump(accuracy_dict, f1)
    with open("perf_dict.json", "w") as f2:
        json.dump(perf_dict, f2)
    with open("error_dict.json", "w") as f3:
        json.dump(error_dict, f3)
    with open("final_result.json", "w") as f4:
        json.dump(prog_list, f4)

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

if __name__== "__main__":
    #decompose_exps(model="o1-mini")
    #o1_natlang_txt_to_json_conversion()
    generate_intermediates(model="o1-mini")
    o1_prog_txt_to_json_conversion()
    #check_generated_progs()
    #evaluate_comparative_perf_edits()
