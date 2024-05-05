from nltk import pos_tag
from nltk.corpus import stopwords
import re
import random
import string
from collections import defaultdict
from SPARQLWrapper import SPARQLWrapper, JSON
import gensim.downloader as api
import itertools
import spacy_dbpedia_spotlight
import spacy

NLP = spacy.load('en_core_web_trf')
NLP.add_pipe('dbpedia_spotlight')
HIGH_FREQ_WORDS = set(stopwords.words('english'))
WORD2VEC_MODEL = api.load('word2vec-google-news-300')
VOCABS = defaultdict(list)
pattern = re.compile(r'^[a-zA-Z0-9_.\']+$')
for vocab in WORD2VEC_MODEL.index_to_key:
    if bool(pattern.match(vocab)) and vocab.lower() not in HIGH_FREQ_WORDS:
        VOCABS[pos_tag([vocab])[0][1]].append(vocab)


def string_matching(word1, word2):
    word1_lower = word1.lower()
    word2_lower = word2.lower()

    if word1_lower in word2_lower or word2_lower in word1_lower:
        return True

    return False

def remove_matched_string_in_a_list(strings):
    if len(strings) < 2:
        return strings
    strings = sorted(strings, key=lambda x: len(x), reverse=True)
    new_strings=[strings[0]]
    for i in range(1, len(strings)):
        flag = True
        for ans_str in new_strings:
            if string_matching(ans_str, strings[i]):
                flag = False
                break
        if flag:
            new_strings.append(strings[i])

    return new_strings


# search wikipedia entities with same attributions
def get_wikipedia_entities(ent, types, limit):

    def find_entities_with_all_types(types_list):
        types_filter = "\n".join(f"?entity rdf:type <{type_uri}> ." for type_uri in types_list)
        sparql = SPARQLWrapper("https://dbpedia.org/sparql")
        sparql.setQuery(f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT DISTINCT ?entity ?label (COUNT(DISTINCT ?ref) as ?popularity)
        WHERE {{
          {types_filter}
          ?entity rdfs:label ?label .
          FILTER (lang(?label) = "en") .
          OPTIONAL {{?ref ?p ?entity . }}
        }}
        GROUP BY ?entity ?label
        ORDER BY DESC(?popularity)
        LIMIT {limit*2}
        """)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        unique_wiki_entities = remove_matched_string_in_a_list([result["label"]["value"] for result in results["results"]["bindings"] if not string_matching(ent, result["label"]["value"])])
            
        return unique_wiki_entities[:limit]

    #  types string to type list
    types_all = []
    for t in types.split(","):
        if "Wiki" in t:
            types_all.append(f"http://www.wikidata.org/entity/{t.split(':')[-1]}")
        elif "Schema" in t:
            types_all.append(f"http://schema.org/{t.split(':')[-1]}")
        elif "DBpedia" in t:
            types_all.append(f"http://dbpedia.org/ontology/{t.split(':')[-1]}")

    # search the entities with decreased attribution combination
    found_entities = []
    for i in range(len(types_all), 0, -1):
        if len(found_entities) >= limit:
            break
        for types_sub in itertools.combinations(types_all, i):
            # find entities with all the attribution in subset
            matched_entities = find_entities_with_all_types(list(types_sub))
            if matched_entities:
                found_entities.extend(matched_entities)
        found_entities=remove_matched_string_in_a_list(found_entities)

    return random.sample(found_entities, limit) if len(found_entities) > limit else found_entities


# get words with similiar embedding
def get_words_with_simi_embed(word, limit):

    def search_word2vec_model(ww, threshold = 0.9):
        try: 
            neighbors = WORD2VEC_MODEL.most_similar(ww, topn=1000)
        except KeyError:
            return []
        return [w for w, s in neighbors if s < threshold]

    if len(word.split()) > 1:
        word = '_'.join(word.split())

    found_words = search_word2vec_model(word)
    if found_words == [] and word.lower() != word:
        found_words = search_word2vec_model(word.lower())
    
    found_words = remove_matched_string_in_a_list([w.replace('_',' ') for w in found_words if not string_matching(w, word)])

    return found_words[:limit] if len(found_words) > limit else found_words

# get random word with the same POS
def get_words_random(word, limit):
    pos = pos_tag([word])[0][1]

    if pos in ['NN', 'NNP', 'NNPS', 'NNS','CD']:
        search_space = VOCABS['NN'] + VOCABS['NNP'] + VOCABS['NNS'] + VOCABS['CD'] + VOCABS['NNPS']
        found_words = random.sample(search_space, limit*2) if len(search_space) > limit*2 else search_space
        found_words = remove_matched_string_in_a_list([w.replace('_',' ') for w in found_words])
        return random.sample(found_words, limit) if len(found_words)>limit else found_words

    elif pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBZ','VBP']:
        search_space = VOCABS['VB'] + VOCABS['VBD'] + VOCABS['VBG']+VOCABS['VBN'] + VOCABS['VBZ']+VOCABS['VBP']
        found_words = random.sample(search_space, limit*2) if len(search_space) > limit*2 else search_space
        found_words = remove_matched_string_in_a_list([w.replace('_',' ') for w in found_words])
        return random.sample(found_words, limit) if len(found_words)>limit else found_words

    elif pos in ['JJ', 'JJR', 'JJS','RBR','RBS','RB']:
        search_space = VOCABS['JJ'] + VOCABS['JJR'] + VOCABS['JJS']
        found_words = random.sample(search_space, limit*2) if len(search_space) > limit*2 else search_space
        found_words = remove_matched_string_in_a_list([w.replace('_',' ') for w in found_words])
        return random.sample(found_words, limit) if len(found_words) > limit else found_words

    else: 
        search_space = []
        for value in VOCABS.values():
            search_space.extend(value)
        found_words = random.sample(search_space, limit) if len(search_space) > limit*2 else search_space
        found_words = remove_matched_string_in_a_list([w.replace('_',' ') for w in found_words])
        return random.sample(found_words, limit) if len(found_words) > limit else found_words

def get_random_number_with_same_format(word, limit):

    def replace_digits(match):
        digits = match.group()
        new_digits = ''.join(random.choice('0123456789') for _ in range(len(digits)))
        while new_digits == digits:
            new_digits = ''.join(random.choice('0123456789') for _ in range(len(digits)))
        return new_digits

    found_words = []
    flag=0
    while len(found_words)<limit:
        found_word=re.sub(r'\d+', replace_digits, word)
        if found_word in found_words:
            flag=flag+1 
        if flag > 50:
            break
        if found_word not in found_words and found_word != word:
            found_words.append(found_word)
            flag=0
         
    return found_words

def get_date(date, limit):
    candidate_list=[]
    months_list = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December"
    ]

    four_digit_numbers=re.findall(r'\b\d{4}\b', date)
    two_one_digit_numbers=re.findall(r'\b\d{2}\b', date)+re.findall(r'\b\d{1}\b', date)
    date_list=date.split()

    for dd in date_list:
        dd=dd.rstrip(string.punctuation)
        if dd in months_list:
            for month in months_list[11:]:
                if dd != month[:3]:
                    candidate_list.append(date.replace(dd, month))

    if four_digit_numbers:
        for num in four_digit_numbers:
            if int(num)-(limit-limit//2)>=0:
                four_digits=[f"{i:04}" for i in range(int(num)-(limit-limit//2),int(num))]+[f"{i:04}" for i in range(int(num)+1, int(num)+limit//2+1)]
            else:
                four_digits=[f"{i:04}"  for i in range(0,int(num))]+[f"{i:04}"  for i in range(int(num)+1, limit+1)]    
            for i in four_digits:
                candidate_list.append(date.replace(num, i))

    if two_one_digit_numbers:
        for num in two_one_digit_numbers:
            two_one_digits=[str(i) for i in range(1,31)]
            for i in two_one_digits:
                candidate_list.append(date.replace(num, i))
                
    return random.sample(candidate_list, limit) if len(candidate_list) > limit else candidate_list

def contains_digits(word):
    return any(c.isdigit() for c in word)

def distractors_for_entities(ent_list, limit):
    distractors_dict=defaultdict(list)
    # search for the whole ent (search source: wiki)
    for ent_text, ent_label, dbpedia_raw_result in ent_list:
        if ent_label == "DBPEDIA_ENT":
            distractors_dict[ent_text].extend(get_wikipedia_entities(ent_text, dbpedia_raw_result["@types"], limit))
        if ent_label == "DATE":
            distractors_dict[ent_text].extend(get_date(ent_text,limit))

    keys_to_delete = [key for key, value in distractors_dict.items() if not value]
    for key in keys_to_delete:
        del distractors_dict[key]
        
    # search for the whole ent (search source: embedding simi)
    replace_total_num=sum(len(value) for value in distractors_dict.values())
    if distractors_dict=={} or replace_total_num<limit:
        for ent_text, _, _ in ent_list:
            distractors_dict[ent_text].extend(get_words_with_simi_embed(ent_text, limit))
            
    keys_to_delete = [key for key, value in distractors_dict.items() if not value]
    for key in keys_to_delete:
        del distractors_dict[key]
    
    # search for each word in the ent (embedding simi)
    replace_total_num=sum(len(value) for value in distractors_dict.values())
    if distractors_dict=={} or replace_total_num<limit:
        for ent_text, _, _ in ent_list:
            ent_split=ent_text.split()
            if len(ent_split)>1:
                for token in list(set(ent_split)):
                    if token.lower() not in HIGH_FREQ_WORDS:
                        candidates=get_words_with_simi_embed(token, limit)
                        distractors_dict[ent_text].extend([ent_text.replace(token, candidate) for candidate in candidates])
    
    keys_to_delete = [key for key, value in distractors_dict.items() if not value]
    for key in keys_to_delete:
        del distractors_dict[key]

    # search randomly
    replace_total_num=sum(len(value) for value in distractors_dict.values())
    if distractors_dict=={} or replace_total_num<limit:
        for ent_text, _, _ in ent_list:
            if contains_digits(ent_text):
                distractors_dict[ent_text].extend(get_random_number_with_same_format(ent_text, limit))
            distractors_dict[ent_text].extend(get_words_random(ent_text,limit))
    
    keys_to_delete = [key for key, value in distractors_dict.items() if not value]
    for key in keys_to_delete:
        del distractors_dict[key]

    return distractors_dict
    
def distractors_for_words(word_list, limit):
    distractors_dict=defaultdict(list)
    # search based on embedding simi
    for token in word_list:
        distractors_dict[token].extend(get_words_with_simi_embed(token,limit))

    keys_to_delete = [key for key, value in distractors_dict.items() if not value]
    for key in keys_to_delete:
        del distractors_dict[key]

    # search randomly
    replace_total_num=sum(len(value) for value in distractors_dict.values())
    if distractors_dict=={} or replace_total_num<limit:
        for token in word_list:
            if contains_digits(token):
                distractors_dict[token].extend(get_random_number_with_same_format(token, limit))
            distractors_dict[token].extend(get_words_random(token,limit))

    keys_to_delete = [key for key, value in distractors_dict.items() if not value]
    for key in keys_to_delete:
        del distractors_dict[key]

    return distractors_dict

from collections import defaultdict

def merge_defaultdicts(dd1, dd2):
    merged = defaultdict(list)
    for key, value in dd1.items():
        merged[key].extend(value)
    for key, value in dd2.items():
        merged[key].extend(value)

    return merged

def distractor_generate(question, answer, limit):
    
    doc_answer = NLP(answer)
    doc_question = NLP(question)

    ents_question = set([ent.text for ent in doc_question.ents])
    unique_ent = set()
    ents_only_ans=[]
    ents_also_que=[]
    for i, ent in enumerate(doc_answer.ents):
        if ent.text in unique_ent:
            continue
        flag=True
        for ent_q in ents_question:
            if string_matching(ent_q,ent.text):
                flag=False
                break
        if flag:
            ents_only_ans.append((ent.text, ent.label_, ent._.dbpedia_raw_result))
        else:
            ents_also_que.append((ent.text, ent.label_, ent._.dbpedia_raw_result))
        unique_ent.add(ent.text)

    token_question = set([token.text.lower() for token in doc_question if not token.is_punct])
    words_only_ans = list(set([token.text for token in doc_answer if not token.is_punct and token.text.lower() not in token_question and token.text.lower() not in HIGH_FREQ_WORDS]))
    words_also_que = list(set([token.text for token in doc_answer if not token.is_punct and token.text.lower() in token_question and token.text not in HIGH_FREQ_WORDS]))
    
    NN_CD_only_ans = []
    VB_JJ_only_ans = []
    for token in words_only_ans:
        if pos_tag([token])[0][1] in ['NN', 'NNP', 'NNPS', 'NNS','CD']:
            NN_CD_only_ans.append(token)
        elif pos_tag([token])[0][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBZ','VBP','JJ', 'JJR', 'JJS','RBR','RBS','RB']:
            VB_JJ_only_ans.append(token)

    NN_CD_also_que = []
    VB_JJ_also_que = []
    for token in words_also_que:
        if pos_tag([token])[0][1] in ['NN', 'NNP', 'NNPS', 'NNS','CD']:
            NN_CD_only_ans.append(token)
        elif pos_tag([token])[0][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBZ','VBP','JJ', 'JJR', 'JJS','RBR','RBS','RB']:
            VB_JJ_only_ans.append(token)
    
    level_1=ents_only_ans
    level_2=NN_CD_only_ans
    level_3=VB_JJ_only_ans+NN_CD_also_que+ents_also_que
    level_4=list(set([token.text for token in doc_answer if not token.is_punct]))

    replace_dict= defaultdict(list)

    if level_1:
        # to replace ents_only_ans
        replace_dict = distractors_for_entities(level_1, limit)

    elif level_2:
        # to replace NN_CD_only_ans
        replace_dict = distractors_for_words(level_2,limit)

    elif level_3:
        # to replace VB_JJ_only_ans + NN_CD_also_que + ents_also_que

        if VB_JJ_only_ans:
            replace_dict = distractors_for_words(VB_JJ_only_ans, limit)

        replace_total_num=sum(len(value) for value in replace_dict.values())
        if (replace_dict=={} or replace_total_num<limit) and ents_also_que:
            replace_dict_ans = distractors_for_entities(ents_also_que, limit)
            replace_dict= merge_defaultdicts(replace_dict,replace_dict_ans)

        replace_total_num=sum(len(value) for value in replace_dict.values())
        if (replace_dict=={} or replace_total_num<limit) and NN_CD_also_que:
            replace_dict_ans = distractors_for_words(NN_CD_also_que, limit)
            replace_dict= merge_defaultdicts(replace_dict,replace_dict_ans)

    elif level_4:
        replace_dict = distractors_for_words(level_4,limit)

    else:
        print("I did not find the token that needs to be replaced.")

    distractors=[]
    for key, value in replace_dict.items():
        for v in value:
            distractors.append(answer.replace(key,v))
    distractors=remove_matched_string_in_a_list(distractors)

    return distractors
