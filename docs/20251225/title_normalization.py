import json
from collections import Counter
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import math
import time
from rapidfuzz import fuzz
from collections import defaultdict
import re
from llm_review_helper_simple import run_llm_review_simple

client = OpenAI(api_key="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")

def unique_list(lst):
    unique = []
    for item in lst:
        if item not in unique:
            unique.append(item)
    return unique


def grouping_by_lowercase(texts):
    groups = {}          # final result: {original_text: [variants]}
    lower_to_key = {}    # mapping: lowercase → original key

    for txt in texts:
        low = txt.lower()

        # If lowercase exists → append to the group
        if low in lower_to_key:
            key = lower_to_key[low]
            
            # append only if unique
            if txt not in groups[key]:
                groups[key].append(txt)

        # If not exist → create new group using original text as the key
        else:
            groups[txt] = [txt]
            lower_to_key[low] = txt

    return groups

# Define stopwords or filler words to remove
STOPWORDS = [
    "and", "of", "the", "for", "in", "on", "at", "to", "by",
    "department", "division", "section", "unit", "office", "group", "team"
]

def normalize_title(title):
    """
    Normalize job title:
    - Lowercase
    - Remove punctuation
    - Remove stopwords
    - Strip extra spaces
    """
    title = title.lower()
    title = re.sub(r"[^\w\s]", " ", title)  # remove punctuation
    words = [w for w in title.split() if w not in STOPWORDS]
    return " ".join(words).strip()

JOB_LEVEL_GROUPS = {
    "executive": [
        "ceo", "chief", "president", "founder", "cofounder", "owner",
        "chairman", "chairwoman", "chairperson", "executive director",
        "managing director", "vp", "vice president", "svp", "avp"
    ],
    "head": [
        "head", "lead", "leader", "team lead", "team leader", "department head"
    ],
    "director": [
        "director", "associate director", "assistant director", "deputy director"
    ],
    "manager": [
        "manager", "supervisor", "controller", "administrator", "foreman"
    ],
    "senior": [
        "senior", "sr", "specialist", "expert", "advisor", "consultant"
    ],
    "mid": [
        "associate", "intermediate", "coordinator", "executive"  # (executive = non-management corporate staff)
    ],
    "junior": [
        "junior", "jr", "entry", "trainee", "support", "assistant"
    ],
    "staff": [
        "staff", "officer", "employee", "representative", "clerk"
    ],
    "intern": [
        "intern", "apprentice", "student", "cadet"
    ],
}


def extract_pure_title(text):
    """Remove known level words to get pure job title."""
    pure_words = []
    for w in text.split():
        if not any(w in group for group in JOB_LEVEL_GROUPS.values()):
            pure_words.append(w)
    return pure_words

def get_job_level_group(text):
    """Return ALL job level groups appearing in the text."""
    levels = []
    for group_name, keywords in JOB_LEVEL_GROUPS.items():
        for kw in keywords:
            if kw in text and group_name:
                levels.append(group_name)

    return levels if levels else None

def is_same_job_title(title1, title2):
    """Compare two job titles by pure job title and same-level hierarchy."""
    t1, t2 = normalize_title(title1), normalize_title(title2)
    pure1, pure2 = extract_pure_title(t1), extract_pure_title(t2)
    level1, level2 = get_job_level_group(t1), get_job_level_group(t2)


    if not pure1 or not pure2:
        return False
    if not level1 or not level2:
        if level1 == ["staff"] or level2 == ["staff"]:
            level1 = ["staff"]
            level2 = ["staff"]
        else:
            return False
        
    return set(pure1) == set(pure2) and set(level1) == set(level2)

def grouping_by_same_level(job_titles):
    titles = job_titles.copy()
    groups = {}

    while titles:
        current = titles.pop(0)
        same_group = [t for t in titles if is_same_job_title(current, t)]
        titles = [t for t in titles if t not in same_group]

        if same_group:
            groups[current] = [current] + same_group
        else:
            groups[current] = [current]

    return groups

def merge_dict(dict1, dict2):
    for k in dict2:
        merged_vals = []
        for val in dict2[k]:
            # If the value is a key in dict01, merge its values
            if val in dict1:
                merged_vals.extend(dict1[val])
            else:
                merged_vals.append(val)
        # remove duplicates and preserve order
        unique = []
        for v in merged_vals:
            if v not in unique:
                unique.append(v)
        dict2[k] = unique

    return dict2

def save_matrix(sim_matrix, filename):
    np.save(filename, sim_matrix)  # saves as .npy file
    print(f"Saved: {filename}")

def load_matrix(filename="similarity.npy"):
    return np.load(filename)

# Choose an embedding model
EMBED_MODEL = "text-embedding-3-large"   # or "text-embedding-3-small"

# Batch size — tune based on your workload
BATCH_SIZE = 100


def embed_texts_in_batches(text_list, batch_size=100):
    """Embed a large list of texts safely in batches."""
    embeddings = []

    num_batches = math.ceil(len(text_list) / batch_size)
    print(f"Total batches: {num_batches}")

    for i in range(num_batches):
        batch = text_list[i * batch_size : (i + 1) * batch_size]

        # Call OpenAI embeddings API
        try:
            response = client.embeddings.create(
                model=EMBED_MODEL,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

        except Exception as e:
            print(f"Error on batch {i+1}: {e}")
            print("Retrying...")
            time.sleep(3)
            continue

        # print(f"Batch {i+1}/{num_batches} done.")

    return embeddings

def grouping_by_similarity(texts, sim_matrix, threshold=0.9):
    """
    Group strings by similarity.
    After grouping one set, remove them from the list before next check.
    """
    groups = []
    remaining = list(range(len(texts)))  # keep indexes only

    while remaining:
        base = remaining[0]   # take first index
        group = [base]

        # compare base with all other remaining indexes
        to_remove = []
        for idx in remaining[1:]:
            if sim_matrix[base][idx] >= threshold:
                group.append(idx)
                to_remove.append(idx)

        # remove base and matched ones
        remaining = [x for x in remaining if x not in group]

        # final group (convert to strings)
        groups.append([texts[i] for i in group])

    return groups

def match_list1_to_list2(list1, list2, sim_matrix):
    results = []

    for i, text1 in enumerate(list1):
        row = sim_matrix[i]
        best_idx = np.argmax(row)
        best_score = float(row[best_idx])
        results.append((text1, list2[best_idx], best_score))

    return results

def merge_to_main_result(main_dict, title_groups):
    key_to_del = []
    for lst in title_groups:
        if len(lst) > 1:
            curr_key = lst[0]
            for k in lst[1:]:
                main_dict[curr_key].extend(main_dict[k])
                key_to_del.append(k)

    for key in key_to_del:
        main_dict.pop(key, None)

    return main_dict

def is_same_pure_diff_level(title1, title2):
    """Same pure job title but different level."""
    t1, t2 = normalize_title(title1), normalize_title(title2)
    pure1, pure2 = extract_pure_title(t1), extract_pure_title(t2)
    level1, level2 = get_job_level_group(t1), get_job_level_group(t2)

    if not pure1 or not pure2:
        return False
    if not level1 or not level2:
        return False

    # Must have same pure job but different (non-null) levels
    return pure1 == pure2 and level1 != level2 and (level1 or level2)


def diff_level_lists(list1, list2):
    """
    For each title in list1, find titles in list2 with same pure title but different level.
    Returns dict: { title_in_list1: [matching_titles_in_list2] }
    """
    results = []
    for title1 in list1:
        for title2 in list2:
            if is_same_pure_diff_level(title1, title2):
                # results.append((title1, title2))
                results.append(title1)
                break
    return results

def grouping_by_fuzzy(texts, threshold=85):
    groups = []  # each group = [list of strings]

    for text in texts:
        placed = False

        for group in groups:
            # Compare with the first entry of the group (group representative)
            rep = group[0]
            score = fuzz.token_set_ratio(text.lower(), rep.lower())

            if score >= threshold:
                group.append(text)
                placed = True
                break

        if not placed:
            groups.append([text])

    return groups

def normalize_dict(list_data:list[str], dict_data:dict)->dict:
    # Step 1: rename keys if not in its own value list
    new_dict = {}
    for key, vals in dict_data.items():
        if key in vals:
            new_key = key
        else:
            new_key = vals[0]  # pick the first value as new key
        new_dict[new_key] = vals[:]  # copy list
    dict_data = new_dict

    # Step 2: remove duplicate values across keys
    seen = set()
    for k in dict_data:
        unique_vals = []
        for v in dict_data[k]:
            if v not in seen:
                unique_vals.append(v)
                seen.add(v)
        dict_data[k] = unique_vals

    # Step 3: add missing values to the first key
    missing = [x for x in list_data if x not in seen]
    if missing:
        first_key = next(iter(dict_data))  # pick first key
        dict_data[first_key].extend(missing)

    return dict_data


def grouping_by_llm(choice:str, list_title:list[str], job_function:str, job_title=None):

    if choice == "cwf":
        system_prompt = """
You are an expert HR analyst. Your task is to group job task titles by meaning.

Context Rules:
- All tasks belong to a specific job role and a specific skill.
- Group tasks that describe the same meaning.
- Each task must appear in exactly one group.
- Single-item groups are allowed.
- The representative must be one of the tasks in the same group.
- Output strictly valid JSON. No explanation, no comments.

Output format:
{
  "Representative task": [
    "task1",
    "task2",
    ...
  ],
  ...
}
        """.strip()
        
        user_prompt = (
            "We have the following data"
            f"Job Function: {job_function}\n"
            f"Job Title: {job_title}\n"
            f"Task Titles:\n{json.dumps(list_title, ensure_ascii=False)}\n"
            "Please group task titles by same meaning."
        )

    elif choice == "skill":
        system_prompt = """
You are an expert HR AI specialized in grouping job skill titles based on meaning and semantic similarity.

Context:
- All provided skill titles are skills for a specific job role.

Rules:
- Group skill titles that have the same meaning or represent the same task.
- Each skill title must appear in exactly one group.
- Single-item groups are allowed.
- Representative must be one of the items in the group.
- Output must be strictly valid JSON.
- No comments or explanations.

Output format:
{
  "Representative skill title": ["skill1", "skill2", ...],
  ...
}
        """.strip()
        
        user_prompt = (
            "We have the following data"
            f"Job Function: {job_function}\n"
            f"Skill Titles:\n{json.dumps(list_title, ensure_ascii=False)}\n"
            "Please group skill titles by same meaning."
        )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0
    )

    raw_output = response.choices[0].message.content.strip()

    try:
        grouped = json.loads(raw_output)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON returned:\n{raw_output}")

    # Validate + Auto-repair
    grouped = normalize_dict(list_data=list_title, dict_data=grouped)

    return grouped

def job_norm_after_llm_check(file_path, main_job_title, to_check_llm):
    df_result_llm = pd.read_csv(f"{file_path}")
    result_llm = list(df_result_llm.itertuples(index=False, name=None))

    # result_llm
    for result in result_llm:
        consider_title = result[0]
        best_match_title = result[1]
        decision = result[3]

        if decision == "NEW_CANONICAL":
            main_job_title[consider_title] = to_check_llm[consider_title]["list"]
        elif decision == "MERGE":
            main_job_title[best_match_title].extend(to_check_llm[consider_title]["list"])
        else:
            print("ERROR")

    return main_job_title

def job_normalization(considered_list:list[str]):

    # unique
    uniq_list = unique_list(considered_list)

    # group by lowercase
    dict_group_lower = grouping_by_lowercase(uniq_list)

    # count and sort all title in list for reference of order
    counts_all = Counter(considered_list)
    sorted_all = sorted(counts_all.keys(), key=lambda x: counts_all[x], reverse=True)
    index_map = {value: idx for idx, value in enumerate(sorted_all)}

    # sorted data
    curr_job_list = list(dict_group_lower.keys())    
    sorted_job_title = sorted(curr_job_list, key=lambda x: index_map.get(x, float('inf')))

    # group by same level
    result_group_level = grouping_by_same_level(sorted_job_title)
    dict_group_level = merge_dict(dict_group_lower, result_group_level)

    # Separate main and remaining list by checking count more than 10
    lookup = {v: k for k, values in dict_group_level.items() for v in values}
    updated_list_title = [lookup[item] for item in considered_list]
    counts_update_title = Counter(updated_list_title)

    main_job_title = {}
    list_all_title = []
    
    for item, count in counts_update_title.items():
        if count > 1:
            main_job_title[item] = dict_group_level[item]
            list_all_title.extend(dict_group_level[item])

    list_main_title = list(main_job_title.keys())
    list_remaining_title = [item for item in dict_group_level.keys() if item not in list_all_title]


    # create embedding and similarity matrix
    print("==== Embedding by OpenAI ====")
    emb_list_main = embed_texts_in_batches(list_main_title)
    emb_list_remaining = embed_texts_in_batches(list_remaining_title)
    
    sim_matrix_main = cosine_similarity(emb_list_main)
    sim_matrix_remaining = cosine_similarity(emb_list_remaining)
    sim_matrix_both = cosine_similarity(emb_list_remaining, emb_list_main)

    # group titles by checking similarity
    print("==== Similarity ====")
    groups_main = grouping_by_similarity(list_main_title, sim_matrix_main, threshold=0.9)
    groups_remaining = grouping_by_similarity(list_remaining_title, sim_matrix_remaining, threshold=0.9)
    matches = match_list1_to_list2(list_remaining_title, list_main_title, sim_matrix_both)

    # group main_dict by similarity score > 0.9
    group_main_dict = {}
    for lst in groups_main:
        key = min(lst, key=lambda x: sorted_job_title.index(x))
        group_main_dict[key] = lst
    
    for key, value in group_main_dict.items():
        current_list = value
    
        if len(current_list) > 1:
            print(current_list)
            merged_values = []
            for t in current_list:
                merged_values.extend(main_job_title[t])            
                if t != key:
                    main_job_title.pop(t, None)
                
            main_job_title[key] = merged_values

    # create remaining_dict by similarity score > 0.9
    remaining_job_title = {}
    for lst in groups_remaining:
        key = min(lst, key=lambda x: sorted_job_title.index(x))
        
        new_group = []
        for t in lst:
            new_group.extend(dict_group_level[t])
    
        remaining_job_title[key] = list(set(new_group))

    # group between remaining and main dictionaries
    group_main_lookup = {v: k for k, values in group_main_dict.items() for v in values}
    new_matches = [(a, group_main_lookup[b], c) for a, b, c in matches]
    curr_list_remaining = list(remaining_job_title.keys())

    key_remaining_to_del = []
    for m in new_matches:
        curr_remaining = m[0]
        current_match = m[1]
        current_score = m[2]
        
        if curr_remaining in curr_list_remaining:
            if current_score > 0.9:
                main_job_title[current_match].extend(remaining_job_title[curr_remaining])
                key_remaining_to_del.append(curr_remaining)
            elif current_score < 0.6:
                main_job_title[curr_remaining] = remaining_job_title[curr_remaining]
                key_remaining_to_del.append(curr_remaining)
            else:
                remaining_job_title[curr_remaining].append(m)

    for key in key_remaining_to_del:
        remaining_job_title.pop(key, None)

    # separate for each in remaining diff from main by job level
    # and create to_check_llm dictionary
    curr_list_main = list(main_job_title.keys())
    curr_list_remaining = list(remaining_job_title.keys())
    result_diff_level = diff_level_lists(curr_list_remaining, curr_list_main)
    
    # to_check_llm = {}
    # key_remaining_to_del = []
    # to_csv = []
    # for title in curr_list_remaining:
    #     if title in result_diff_level:
    #         main_job_title[title] = remaining_job_title[title][:-1]
    #         key_remaining_to_del.append(title)
    #     else:
    #         list_value = remaining_job_title[title][:-1]
    #         check_llm = remaining_job_title[title][-1]
    #         to_csv.append(check_llm)
    #         to_check_llm[title] = {"list": list_value, "check_llm": check_llm}

    key_remaining_to_del = []
    list_result_llm = []
    for title in curr_list_remaining:
        if title in result_diff_level:
            main_job_title[title] = remaining_job_title[title][:-1]
            key_remaining_to_del.append(title)
        else:
            curr_rare_item = remaining_job_title[title][-1][0]
            curr_best_match = remaining_job_title[title][-1][1]
            curr_similarity = remaining_job_title[title][-1][2]
            result_llm = run_llm_review_simple(curr_rare_item, curr_best_match, curr_similarity)
            list_result_llm.append(result_llm)
            
            decision = result_llm["decision"]
            if decision == "NEW_CANONICAL":
                main_job_title[curr_rare_item] = remaining_job_title[curr_rare_item][:-1]
            elif decision == "MERGE":
                main_job_title[curr_best_match].append(curr_rare_item)

    # for key in key_remaining_to_del:
    #     remaining_job_title.pop(key, None)

    total_main = 0
    for key, value in main_job_title.items():
        total_main += len(value)
        
    print("==== check total ====")
    print(f"unique total: {len(uniq_list)}")
    print(f"total after normalization: {total_main}")

    return main_job_title

def skill_normalization(considered_list:list[str], choice, job_function):

    # unique
    uniq_list = unique_list(considered_list)

    # count and sort all title in list for reference of order
    counts_all = Counter(considered_list)
    sorted_all = sorted(counts_all.keys(), key=lambda x: counts_all[x], reverse=True)

    # create first main result in dictionary
    main_title_result = grouping_by_lowercase(uniq_list)

    # sort data of main title result
    index_map = {value: idx for idx, value in enumerate(sorted_all)}
    current_key_title = list(main_title_result.keys())
    sorted_curr_key = sorted(current_key_title, key=lambda x: index_map.get(x, float('inf')))

    # create embedding and similarity matrix
    print("==== Embedding by OpenAI ====")
    emb_title = embed_texts_in_batches(sorted_curr_key)
    sim_matrix = cosine_similarity(emb_title)
    # save_matrix(sim_matrix=sim_matrix, filename="File/similarity_skill_title.npy")

    # group titles by checking similarity
    print("==== Similarity ====")
    groups_by_sim = grouping_by_similarity(sorted_curr_key, sim_matrix, threshold=0.85)
    main_title_result = merge_to_main_result(main_title_result, groups_by_sim)

    # group titles by fuzzy match
    print("==== Fuzzy Match ====")
    current_key_title = list(main_title_result.keys())
    groups_by_fuzzy = grouping_by_fuzzy(current_key_title, threshold=85)
    main_title_result = merge_to_main_result(main_title_result, groups_by_fuzzy)

    # group titles by LLM
    print("==== LLM ====")
    update_dict = []
    del_list = []
    for key, value in main_title_result.items():
        curr_list_title = value
        if len(curr_list_title) > 1:
            del_list.extend(curr_list_title)
            groups_by_llm = grouping_by_llm(choice, curr_list_title, job_function)
            update_dict.append(groups_by_llm)

    for key in del_list:
        main_title_result.pop(key, None)

    for u in update_dict:
        main_title_result.update(u)


    count_skill = 0
    for key, value in main_title_result.items():
        count_skill += len(value)
    print("==== check total ====")
    print(f"unique total: {len(uniq_list)}")
    print(f"total after normalization: {count_skill}")

    return main_title_result

def cwf_normalization(considered_list:list[str], choice, job_function, job_title):

    # unique
    uniq_list = unique_list(considered_list)

    # count and sort all title in list for reference of order
    counts_all = Counter(considered_list)
    sorted_all = sorted(counts_all.keys(), key=lambda x: counts_all[x], reverse=True)

    # create first main result in dictionary
    main_title_result = grouping_by_lowercase(uniq_list)

    # sort data of main title result
    index_map = {value: idx for idx, value in enumerate(sorted_all)}
    current_key_title = list(main_title_result.keys())
    sorted_curr_key = sorted(current_key_title, key=lambda x: index_map.get(x, float('inf')))

    # create embedding and similarity matrix
    emb_title = embed_texts_in_batches(sorted_curr_key)
    sim_matrix = cosine_similarity(emb_title)
    # save_matrix(sim_matrix=sim_matrix, filename="File/similarity_skill_title.npy")

    # group titles by checking similarity
    groups_by_sim = grouping_by_similarity(sorted_curr_key, sim_matrix, threshold=0.85)
    main_title_result = merge_to_main_result(main_title_result, groups_by_sim)

    # group titles by fuzzy match
    current_key_title = list(main_title_result.keys())
    groups_by_fuzzy = grouping_by_fuzzy(current_key_title, threshold=85)
    main_title_result = merge_to_main_result(main_title_result, groups_by_fuzzy)

    # group titles by LLM
    update_dict = []
    del_list = []
    for key, value in main_title_result.items():
        curr_list_title = value
        if len(curr_list_title) > 1:
            del_list.extend(curr_list_title)
            groups_by_llm = grouping_by_llm(choice, curr_list_title, job_function, job_title)
            update_dict.append(groups_by_llm)

    for key in del_list:
        main_title_result.pop(key, None)

    for u in update_dict:
        main_title_result.update(u)


    count_skill = 0
    for key, value in main_title_result.items():
        count_skill += len(value)
    print("==== check total ====")
    print(f"unique total: {len(uniq_list)}")
    print(f"total after normalization: {count_skill}")

    return main_title_result