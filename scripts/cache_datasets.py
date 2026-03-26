#!/usr/bin/env python3
"""Run once on a machine with internet access to populate the HF datasets arrow cache."""
from datasets import load_dataset

tasks = [
    ('juletxara/mgsm', ['bn','de','en','es','fr','ja','ru','sw','te','th','zh'], 'test'),
    ('xcopa', ['et','ht','id','it','qu','sw','ta','th','tr','vi','zh'], 'test'),
    ('juletxara/xstory_cloze', ['ar','es','eu','hi','id','my','ru','sw','te','zh'], 'eval'),
    ('americas_nli', ['aym','bzd','cni','gn','hch','nah','oto','quy','shp','tar'], 'test'),
    ('mkqa', [None], 'train'),
    ('Maxwell-Jia/AIME_2024', [None], 'train'),
    ('livecodebench/code_generation_lite', [None], 'test'),
    ('xnli', ['ar','bg','de','el','en','es','fr','hi','ru','sw','th','tr','ur','vi','zh'], 'test'),
    ('facebook/belebele', ['eng_Latn','arb_Arab','zho_Hans','deu_Latn','ben_Beng','fra_Latn',
                           'hin_Deva','swh_Latn','jpn_Jpan','kor_Hang','rus_Cyrl','spa_Latn',
                           'por_Latn','ita_Latn','tur_Latn','vie_Latn','ind_Latn','tha_Thai',
                           'pol_Latn','nld_Latn','ron_Latn','ukr_Cyrl','ces_Latn','fin_Latn'], 'test'),
    #('GAIR/OlympiadBench', ['OE_TO_maths_en_COMP','OE_TO_maths_zh_COMP'], 'test'),
    ('Idavidrein/gpqa', ['gpqa_diamond'], 'train'),
]

for name, subsets, split in tasks:
    for subset in subsets:
        try:
            if subset:
                ds = load_dataset(name, subset, split=split, trust_remote_code=True)
            else:
                ds = load_dataset(name, split=split, trust_remote_code=True)
            print(f'OK: {name}/{subset} - {len(ds)} examples')
        except Exception as e:
            print(f'FAIL: {name}/{subset}: {e}')
