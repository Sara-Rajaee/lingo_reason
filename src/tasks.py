from datasets import load_dataset
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Optional
import sacrebleu
import iso639
#from comet import download_model, load_from_checkpoint
from math_verify import parse, StringExtractionConfig, LatexExtractionConfig, verify
import re

# ---------- Base Classes ----------

class BaseBenchmark(ABC):
    """Abstract base class for all benchmarks"""
    
    def __init__(self, task_config, subset):
        self.task_config = task_config
        self.subset = subset
        self.defaults = task_config.get('defaults', {})
        self.dataset = None
    
    @abstractmethod
    def load_data(self):
        """Load the benchmark dataset"""
        pass
    
    @abstractmethod
    def prepare_prompt(self, example):
        """Prepare prompt for a single example"""
        pass
    
    @abstractmethod
    def evaluate(self, predictions, references, eval_types=None, points=None):
        """Evaluate predictions against references"""
        pass
    
    def get_generation_params(self, model_params):
        """Merge task defaults with model params"""
        params = self.defaults.copy()
        # Model params override task defaults
        params.update(model_params)
        return params


class BenchmarkFactory:
    @staticmethod
    def get_benchmark(benchmark_type, task_config, subset):
        if benchmark_type == 'mmmlu':
            return MMMLUBenchmark(task_config, subset)
        elif benchmark_type == 'wmt24pp':
            return WMT24PPBenchmark(task_config, subset)
        elif benchmark_type == 'polymath':
            return PolyMathBenchmark(task_config, subset)
        elif benchmark_type == 'linguini':
            return LinguiniBenchmark(task_config, subset)
        elif benchmark_type == 'mulr':
            return MuLRBenchmark(task_config, subset)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_type}")


# ---------- MMMLU ----------

@dataclass
class MMMLUExample:
    id: str
    question: str
    A: str
    B: str
    C: str
    D: str
    answer: str
    subject: str


def extract_choice(text):
    """Extract A/B/C/D choice from model output"""
    for c in text.strip():
        if c in "ABCDabcd":
            return c
    return ""


class MMMLUBenchmark(BaseBenchmark):
    """MMMLU (Massive Multitask Multilingual Language Understanding) benchmark"""
    
    def load_data(self):
        """Load MMMLU dataset for specific subset"""
        name = self.task_config['dataset_name']
        split = self.task_config['split']
        limit = self.defaults.get('limit_per_subset')
        
        print(f"Loading {name} ({self.subset})...")
        ds = load_dataset(name, self.subset, split=split)
        
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        
        examples = []
        for i, r in enumerate(ds):
            examples.append(MMMLUExample(
                id=f"{self.subset}_{i}",
                question=r["question"],
                A=r["option_a"], 
                B=r["option_b"], 
                C=r["option_c"], 
                D=r["option_d"],
                answer=r["answer"],
                subject=r["subject"],
            ))
        
        self.dataset = examples
        print(f"Loaded {len(examples)} examples from {self.subset}")
        return examples
    
    def prepare_prompt(self, example):
        """Prepare MMMLU prompt"""
        return (
            "The following is a multi-choice question. Answer with only the letter (A, B, C, or D). Do not include explanations in your final answer.\n\n"
            f"{example.question}\n"
            f"A) {example.A}\nB) {example.B}\nC) {example.C}\nD) {example.D}\n\n"
        )
    
    def evaluate(self, predictions, references, eval_types=None, points=None):
        """Evaluate MMMLU predictions"""
        correct = 0
        total = len(predictions)
        
        extracted_predictions = [extract_choice(pred) for pred in predictions]
        scores = []
        for pred, ref in zip(extracted_predictions, references):
            if pred == ref:
                correct += 1
                scores.append(1)
            else:
                scores.append(0)

        accuracy = (correct / total * 100) if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'per_example_accuracy': scores,
        }

# ---------- PolyMath ----------

poly_math_instruction = {
    'en': 'Note: Please put the final answer in the $\\boxed{{}}$',
    'zh': '注意：请将最终答案放在 $\\boxed{{}}$ 中。',
    'ar': '.$\\boxed{{}}$ مالحظة: يُرجى وضع اإلجابة النهائية في',
    'bn': 'ব িঃদ্রিঃ: অনুগ্রহ করে চূ ড়ান্ত উত্তেটি $\\boxed{{}}$ এে মরযে ে়াখুন।.',
    'de': 'Hinweis: Bitte setzen Sie die endgültige Antwort in $\\boxed{{}}$.',
    'es': 'Nota: Por favor, coloque la respuesta final en el $\\boxed{{}}$.',
    'fr': 'Remarque : Veuillez mettre la réponse finale dans le $\\boxed{{}}$.',
    'id': 'Catatan: Silakan letakkan jawaban akhir di dalam $\\boxed{{}}$.',
    'it': 'Nota: Per favore, metti la risposta finale nel $\\boxed{{}}$.',
    'ja': '注意：最終的な答えを $\\boxed{{}}$ に入れてください。',
    'ko': '참고: 최종 답안을 $\\boxed{{}}$ 안에 넣어 주세요.',
    'ms': 'Nota: Sila letakkan jawapan akhir dalam $\\boxed{{}}$.',
    'pt': 'Nota: Por favor, coloque a resposta final no $\\boxed{{}}$.',
    'ru': 'Примечание: Пожалуйста, поместите окончательный ответ в $\\boxed{{}}$.',
    'sw': 'Kumbuka: Tafadhali weka jibu la mwisho katika $\\boxed{{}}$.',
    'te': 'గమనిక: దయచేసి తుది జవాబును $\\boxed{{}}$ లో ఉంచండి.',
    'th': 'หมายเหตุ: กรุณาใส่ค าตอบสุดท้ายใน $\\boxed{{}}$.',
    'vi': 'Lưu ý: Vui lòng đặt câu trả lời cuối cùng trong $\\boxed{{}}$.'
}

@dataclass
class PolyMathExample:
    id: str
    question: str
    answer: str

def normalize_latex(text):
    # Unwrap font/text commands FIRST: \text{x} -> x
    for cmd in [r'\\text', r'\\mathrm', r'\\mathbf', r'\\mathit', r'\\mathsf', r'\\mbox']:
        text = re.sub(cmd + r'\{([^}]+)\}', r'\1', text)

    # NOW normalize double braces (content is clean of \text{}): \boxed{{x}} -> \boxed{x}
    text = re.sub(r'\\boxed\{\{([^{}]+)\}\}', r'\\boxed{\1}', text)

    # Remove \displaystyle
    text = re.sub(r'\\displaystyle\s*', '', text)

    # Remove currency symbols inside \boxed{}
    text = re.sub(r'(\\boxed\{)[\\]?\$\s*', r'\1', text)

    # Remove % and \% symbol inside \boxed{}
    text = re.sub(r'(\\boxed\{[^}]*?)\\%(\})', r'\1\2', text)
    text = re.sub(r'(\\boxed\{[^}]*?)(?<!\\)%(\})', r'\1\2', text)

    # Remove trailing unit text inside \boxed{} (any Unicode, with or without space)
    text = re.sub(r'(\\boxed\{-?[\d.,]+)\s*[^}0-9.,\s][^}]*(\})', r'\1\2', text)  # ← updated line

    # Remove English thousand separators globally: 70,000 -> 70000
    for _ in range(3):
        text = re.sub(r'(\d),(\d{3})', r'\1\2', text)

    def normalize_european_number(m):
        inner = m.group(1)
        # German/European thousands separator: 57.500 -> 57500
        inner = re.sub(r'(\d)\.(\d{3})(?!\d)', r'\1\2', inner)
        # German decimal comma: 26,00 -> 26.00
        inner = re.sub(r'(\d),(\d{1,2})$', r'\1.\2', inner)
        return r'\boxed{' + inner + '}'

    text = re.sub(r'\\boxed\{([^}]+)\}', normalize_european_number, text)

    # Remove trailing .0+ inside \boxed{}: \boxed{28.00} -> \boxed{28}
    text = re.sub(r'(\\boxed\{-?\d+)\.0+(\})', r'\1\2', text)

    return text



def extract_boxed_answer(text):
    """Extract answer within the \\boxed{{}}"""
    normalized = normalize_latex(text)

    return parse(
            normalized, 
            extraction_config=[
                LatexExtractionConfig(
                    boxed_match_priority=0, 
                    try_extract_without_anchor=True,
                ),
            ]
        )


class PolyMathBenchmark(BaseBenchmark):
    """PolyMath: Evaluating Mathematical Reasoning in Multilingual Contexts"""
    
    def load_data(self):
        """Load polymath dataset for specific subset"""
        name = self.task_config['dataset_name']
        split = self.task_config['split']
        limit = self.defaults.get('limit_per_subset')
        
        print(f"Loading {name} ({self.subset})...")
        ds = load_dataset(name, self.subset, split=split)
        
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        
        examples = []
        for i, r in enumerate(ds):
            examples.append(PolyMathExample(
                id=f"{self.subset}_{i}",
                question=r["question"],
                answer=r["answer"],
            ))
        
        self.dataset = examples
        print(f"Loaded {len(examples)} examples from {self.subset}")
        return examples
    
    def prepare_prompt(self, example):
        """Prepare PolyMath prompt"""
        return (
            f"{example.question}\n"
            f"{poly_math_instruction[self.subset]}\n\n"
        )
    
    def evaluate(self, predictions, references, eval_types=None, points=None):
        """Evaluate MMMLU predictions"""
        correct = 0
        total = len(predictions)
        
        extracted_predictions = [extract_boxed_answer(pred) for pred in predictions]
        scores = []
        for pred, ref in zip(extracted_predictions, references):
            if verify(ref, pred):
                correct += 1
                scores.append(1)
            else:
                scores.append(0)
        
        
        accuracy = (correct / total * 100) if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'per_example_accuracy': scores
        }

# ---------- Linguini ----------

@dataclass
class LinguiniExample:
    id: str
    context: str          # The linguistic puzzle context (word pairs, etc.)
    question: str         # Individual question text
    answer: str           # Gold answer (exact match target)
    task_type: str        # e.g. "translation", "match_letter", "fill_blanks"
    task_lang: str         # The low-resource language being tested
    problem_id: str       # Parent problem ID (multiple questions share one problem)
    eval_type: str


class LinguiniBenchmark(BaseBenchmark):
    """Linguini benchmark for language-agnostic linguistic reasoning.

    Based on International Linguistic Olympiad (IOL) problems.
    Each 'problem' contains a context (word pairs in an unknown language)
    and multiple questions requiring deductive linguistic reasoning.
    Evaluated via exact match across 894 questions / 160 problems.

    HuggingFace dataset: facebook/linguini
    - Single config: 'default'
    - Single split: 'test'
    - 160 rows (one per problem); each row contains a list of questions+answers.
    """

    def load_data(self):
        name = self.task_config['dataset_name']       # "facebook/linguini"
        split = self.task_config.get('split', 'test')
        limit = self.defaults.get('limit_per_subset')

        print(f"Loading {name} ({self.subset})...")
        # Linguini has a single config ('default') and a single split ('test').
        # self.subset is expected to be 'default' when running this benchmark.
        ds = load_dataset(name, split=split)

        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        examples = []
        for problem in ds:
            # Each row is a full problem with (potentially) multiple questions.
            # Field names below reflect the most likely schema; adjust if needed.
            problem_id   = str(problem.get('problem_id', problem.get('id', '')))
            context      = problem.get('context', problem.get('context', ''))
            task_type    = problem.get('task_type', problem.get('task_type', 'unknown'))
            task_lang    = problem.get('task_lang', problem.get('task_lang', 'unknown'))

            # Questions and answers are stored as parallel lists inside each row.
            query = problem.get('query', [])
            answers   = problem.get('answers', [])

        
            examples.append(LinguiniExample(
                id=f"{problem_id}_q{q_idx}",
                context=context,
                question=q,
                answer=str(a).strip(),
                task_type=task_type,
                task_lang=task_lang,
                problem_id=problem_id,
            ))

        self.dataset = examples
        print(f"Loaded {len(examples)} question instances from {self.subset}")
        return examples

    def prepare_prompt(self, example: LinguiniExample) -> str:
        """Build a zero-shot prompt for a Linguini linguistic reasoning question.

        The context contains the key information needed to solve the puzzle
        (the model should NOT rely on prior knowledge of the language).
        """
        return (
            "You are solving a linguistic puzzle. "
            "All the information you need is contained in the context below — "
            "no prior knowledge of this language is required.\n\n"
            f"Context:\n{example.context}\n\n"
            f"Question:\n{example.question}\n\n"
            "Answer with only the requested word or phrase. "
            "Do not include explanations in your final answer.\n\n"
        )

    def evaluate(self, predictions: List[str], references: List[str], 
                 eval_types: Optional[List[str]]=None, points: Optional[List[float]] =None) -> dict:
        """Evaluate using exact match (case-insensitive) + chrF score."""
        assert len(predictions) == len(references), (
            f"Mismatch: {len(predictions)} predictions vs {len(references)} references"
        )

        scores = []
        correct = 0
        chrf_scores = []

        for pred, ref in zip(predictions, references):
            pred_norm = pred.strip().lower()
            ref_norm  = ref.strip().lower()

            # Exact match
            match = int(pred_norm == ref_norm)
            scores.append(match)
            correct += match

            # chrF per example (character n-gram F-score)
            chrf_scores.append(_chrf(pred_norm, ref_norm))

        total    = len(predictions)
        accuracy = (correct / total * 100) if total > 0 else 0.0
        chrf_avg = (sum(chrf_scores) / total * 100) if total > 0 else 0.0

        return {
            'accuracy': accuracy,
            'chrf': chrf_avg,
            'correct': correct,
            'total': total,
            "per_example_scores": {
                "accuracy": scores,
                "chrf": chrf_scores,
            },
        }

# ---------- WMT24++ ----------

LANGUAGE_PAIRS = (
    "en-ar_EG", "en-ar_SA", "en-bg_BG", "en-bn_IN", "en-ca_ES", "en-cs_CZ", "en-da_DK", "en-de_DE",
    "en-el_GR", "en-es_MX", "en-et_EE", "en-fa_IR", "en-fi_FI", "en-fil_PH", "en-fr_CA", "en-fr_FR",
    "en-gu_IN", "en-he_IL", "en-hi_IN", "en-hr_HR", "en-hu_HU", "en-id_ID", "en-is_IS", "en-it_IT",
    "en-ja_JP", "en-kn_IN", "en-ko_KR", "en-lt_LT", "en-lv_LV", "en-ml_IN", "en-mr_IN", "en-nl_NL",
    "en-no_NO", "en-pa_IN", "en-pl_PL", "en-pt_BR", "en-pt_PT", "en-ro_RO", "en-ru_RU", "en-sk_SK",
    "en-sl_SI", "en-sr_RS", "en-sv_SE", "en-sw_KE", "en-sw_TZ", "en-ta_IN", "en-te_IN", "en-th_TH",
    "en-tr_TR", "en-uk_UA", "en-ur_PK", "en-vi_VN", "en-zh_CN", "en-zh_TW", "en-zu_ZA",
)

LANGUAGE_BY_CODE = {
    "ar_EG": "Arabic", "ar_SA": "Arabic", "bg_BG": "Bulgarian", "bn_BD": "Bengali", "bn_IN": "Bengali",
    "ca_ES": "Catalan", "cs_CZ": "Czech", "da_DK": "Danish", "de_DE": "German", "el_GR": "Greek",
    "es_MX": "Spanish", "et_EE": "Estonian", "fa_IR": "Farsi", "fi_FI": "Finnish", "fil_PH": "Filipino",
    "fr_CA": "French", "fr_FR": "French", "gu_IN": "Gujarati", "he_IL": "Hebrew", "hi_IN": "Hindi",
    "hr_HR": "Croatian", "hu_HU": "Hungarian", "id_ID": "Indonesian", "is_IS": "Icelandic",
    "it_IT": "Italian", "ja_JP": "Japanese", "kn_IN": "Kannada", "ko_KR": "Korean", "lt_LT": "Lithuanian",
    "lv_LV": "Latvian", "ml_IN": "Malayalam", "mr_IN": "Marathi", "nl_NL": "Dutch", "no_NO": "Norwegian",
    "pa_IN": "Punjabi", "pl_PL": "Polish", "pt_BR": "Portuguese", "pt_PT": "Portuguese", "ro_RO": "Romanian",
    "ru_RU": "Russian", "sk_SK": "Slovak", "sl_SI": "Slovenian", "sr_RS": "Serbian", "sv_SE": "Swedish",
    "sw_KE": "Swahili", "sw_TZ": "Swahili", "ta_IN": "Tamil", "te_IN": "Telugu", "th_TH": "Thai",
    "tr_TR": "Turkish", "uk_UA": "Ukrainian", "ur_PK": "Urdu", "vi_VN": "Vietnamese",
    "zh_CN": "Mandarin", "zh_TW": "Mandarin", "zu_ZA": "Zulu",
}

REGION_BY_CODE = {
    "ar_EG": "Egypt", "ar_SA": "Saudi Arabia", "bg_BG": "Bulgaria", "bn_BD": "Bangladesh", "bn_IN": "India",
    "ca_ES": "Spain", "cs_CZ": "Czechia", "da_DK": "Denmark", "de_DE": "Germany", "el_GR": "Greece",
    "es_MX": "Mexico", "et_EE": "Estonia", "fa_IR": "Iran", "fi_FI": "Finland", "fil_PH": "Philippines",
    "fr_CA": "Canada", "fr_FR": "France", "gu_IN": "India", "he_IL": "Israel", "hi_IN": "India",
    "hr_HR": "Croatia", "hu_HU": "Hungary", "id_ID": "Indonesia", "is_IS": "Iceland", "it_IT": "Italy",
    "ja_JP": "Japan", "kn_IN": "India", "ko_KR": "South Korea", "lt_LT": "Lithuania", "lv_LV": "Latvia",
    "ml_IN": "India", "mr_IN": "India", "nl_NL": "Netherlands", "no_NO": "Norway", "pa_IN": "India",
    "pl_PL": "Poland", "pt_BR": "Brazil", "pt_PT": "Portugal", "ro_RO": "Romania", "ru_RU": "Russia",
    "sk_SK": "Slovakia", "sl_SI": "Slovenia", "sr_RS": "Serbia", "sv_SE": "Sweden", "sw_KE": "Kenya",
    "sw_TZ": "Tanzania", "ta_IN": "India", "te_IN": "India", "th_TH": "Thailand", "tr_TR": "Turkey",
    "uk_UA": "Ukraine", "ur_PK": "Pakistan", "vi_VN": "Vietnam", "zh_CN": "China", "zh_TW": "Taiwan",
    "zu_ZA": "South Africa",
}

WMT24PP_PROMPT = """You are a professional {src_lang} to {tgt_lang} translator, tasked with providing translations suitable for use in {tgt_lang} ({tgt_region}). Your goal is to accurately convey the meaning and nuances of the original {src_lang} text while adhering to {tgt_lang} grammar, vocabulary, and cultural sensitivities.
Produce only the {tgt_lang} translation, without any additional explanations or commentary.
Please translate the following {src_lang} text into {tgt_lang} ({tgt_region}):
{input_text}"""


@dataclass
class MTExample:
    id: str
    source: str
    reference: str


class WMT24PPBenchmark(BaseBenchmark):
    """WMT24++ Machine Translation benchmark"""
    
    def __init__(self, task_config, subset):
        super().__init__(task_config, subset)
        self.comet_model = None
        self.include_comet = task_config['defaults'].get("include_comet", False)
        self.sources = []  # Store sources for COMET evaluation
    
    def load_data(self):
        """Load WMT24++ dataset for specific language pair"""
        name = self.task_config['dataset_name']
        split = self.task_config['split']
        limit = self.defaults.get('limit_per_subset')
        filter_bad = self.defaults.get('filter_bad', False)
        
        print(f"Loading {name} ({self.subset})...")
        ds = load_dataset(name, self.subset, split=split)
        
        if filter_bad and "is_bad_source" in ds.column_names:
            ds = ds.filter(lambda x: not x["is_bad_source"])

        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        
        examples = [
            MTExample(f"{self.subset}_{i}", r["source"], r["target"]) 
            for i, r in enumerate(ds)
        ]

        # Store sources for COMET
        self.sources = [ex.source for ex in examples]
        
        self.dataset = examples
        print(f"Loaded {len(examples)} examples from {self.subset}")
        return examples
    
    def prepare_prompt(self, example):
        """Prepare WMT24++ translation prompt"""
        src_lang = self.defaults.get('src_lang', 'English')
        
        if self.subset not in LANGUAGE_PAIRS:
            tgt_code = self.subset.split("-", 1)[1]
            tgt_lang = LANGUAGE_BY_CODE.get(tgt_code, tgt_code)
            tgt_region = REGION_BY_CODE.get(tgt_code, tgt_code)
        else:
            tgt_code = self.subset.split("-", 1)[1]
            tgt_lang = LANGUAGE_BY_CODE[tgt_code]
            tgt_region = REGION_BY_CODE[tgt_code]
        
        return WMT24PP_PROMPT.format(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            tgt_region=tgt_region,
            tgt_code=tgt_code,
            input_text=example.source,
        )
    
    def _load_comet_model(self):
        """Load xCOMET-xl model (lazy loading)"""
        if self.comet_model is None:
            print("Loading xCOMET-xl model...")
            model_path = download_model("Unbabel/XCOMET-XL")
            self.comet_model = load_from_checkpoint(model_path)
            print("xCOMET-xl model loaded!")
        return self.comet_model
    
    def evaluate(self, predictions, references, eval_types=None, points=None):
        """Evaluate WMT24++ predictions using xCOMET, BLEU, and chrF++"""

        assert len(predictions) == len(references), \
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"

        # Calculate per-example BLEU and chrF++, setting None for empty predictions
        per_example_bleu = []
        per_example_chrf = []
        for pred, ref in zip(predictions, references):
            if not pred or not pred.strip():
                per_example_bleu.append(None)
                per_example_chrf.append(None)
            else:
                per_example_bleu.append(sacrebleu.sentence_bleu(pred, [ref]).score)
                per_example_chrf.append(sacrebleu.sentence_chrf(pred, [ref], word_order=2).score)

        # Filter to valid examples for corpus-level metrics
        valid = [
            (pred, ref, src)
            for pred, ref, src in zip(predictions, references, self.sources)
            if pred and pred.strip()
        ]

        if len(valid) < len(predictions):
            print(f"Warning: {len(predictions) - len(valid)} empty predictions excluded from corpus metrics")

        valid_preds, valid_refs, valid_srcs = map(list, zip(*valid))

        # Calculate corpus BLEU and chrF++
        bleu = sacrebleu.corpus_bleu(valid_preds, [valid_refs])
        chrf = sacrebleu.corpus_chrf(valid_preds, [valid_refs], word_order=2)

        results = {
            "bleu": bleu.score,
            "chrfpp": chrf.score,
            "num_predictions": len(predictions),
            "per_example_scores": {
                "bleu": per_example_bleu,
                "chrfpp": per_example_chrf,
            },
        }

        # Calculate xCOMET score
        if self.include_comet:
            print("Computing xCOMET scores...")
            comet_model = self._load_comet_model()
            
            # Prepare data for COMET
            data = [
                {
                    "src": src,
                    "mt": pred,
                    "ref": ref
                }
                for src, pred, ref in zip(self.sources, predictions, references)
            ]
            out = comet_model.predict(data, batch_size=64, gpus=1)
            # Extract per-example scores robustly, but ALWAYS aggregate as mean(scores)
            if hasattr(out, "scores"):
                comet_scores = [float(s) for s in out.scores]
            elif isinstance(out, dict) and "scores" in out:
                comet_scores = [float(s) for s in out["scores"]]
            else:
                comet_scores = [float(s) for s in out]

            comet_mean = sum(comet_scores) / len(comet_scores)
            # Std (population std, consistent with your previous formula)
            comet_var = sum((s - comet_mean) ** 2 for s in comet_scores) / len(comet_scores)
            comet_std = comet_var ** 0.5

            results.update({"xcomet-xl": comet_mean, "xcomet-xl_std": comet_std})
            results["per_example_scores"].update({"xcomet-xl": comet_scores})

        return results
    

# ---- WMT Multilingual Linguistic Reasoning (MuLR) ----
@dataclass
class MuLRExample:
    id: str
    prompt: str           # The linguistic puzzle prompt including instruction and task context
    answer: str           # Gold answer
    task_type: str        # one of "translation", "mapping", "fill-in-blanks", "classification"
    task_lang: str        # The low-resource language being tested
    eval_type: str        # either "chrF" or "exact match"
    points: float         # How many points this task scores (proxy of difficulty)
    meta: str             # Author and task information


class MuLRBenchmark(BaseBenchmark):
    """Multilingual linguistic reasoning benchmark from WMT MIST 2025.

    Based on International Linguistic Olympiad (IOL) problems.
    Each 'problem' contains a context (word pairs in an unknown language)
    and multiple questions requiring deductive linguistic reasoning.
    These are reorganized into individual prompts and answers (90).
    Evaluated via ChrF and exact match, weighted by points.

    # TODO upload on huggingface
    Local dataset: json
    """

    def load_data(self):
        name = self.task_config['dataset_name']       # "mulr"
        split = self.task_config.get('split', 'train')  # no train split, just default
        limit = self.defaults.get('limit_per_subset')

        print(f"Loading {name} ({self.subset})...")
        ds = load_dataset('json', data_files="data/linguistic_reasoning_wmt_2025_test.json", split=split)

        # Select language.        
        def select_language(example):
            return iso639.Language.from_name(example['problem_language']).part1 == self.subset 
        
        ds = ds.filter(select_language)

        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        examples = []
        for problem in ds:
            # Each row is a full problem with one question.
            problem_id   = problem['id']
            prompt       = problem['prompt']
            task_type    = problem['type']
            eval_type    = problem['eval_type']
            task_lang    = problem['instruction_language']
            answer       = problem['answer']
            meta         = problem['meta']
            points       = problem['points']
        
            examples.append(MuLRExample(id=problem_id, prompt=prompt, answer=answer, task_type=task_type, 
                                        task_lang=task_lang, eval_type=eval_type, points=points, meta=meta))
            

        self.dataset = examples
        print(f"Loaded {len(examples)} question instances from {self.subset}")
        return examples
    
    def prepare_prompt(self, example):
        # This benchmark has prompts readily formatted.
        return example.prompt

    def evaluate(self, predictions: List[str], references: List[str], eval_types: Optional[List[str]]=None, points: Optional[List[float]] =None) -> dict:
        """Evaluate using exact match (case-insensitive) and chrF score, weighted by points."""
        assert len(predictions) == len(references), (
            f"Mismatch: {len(predictions)} predictions vs {len(references)} references"
        )

        def ends_with_bracketed_line(text):
            lines = text.splitlines()
            if not lines:
                return False
            last_line = lines[-1].strip()
            return last_line.startswith('[') and last_line.endswith(']')

        def extract_text_between_brackets(input_string):
            last_line = input_string.splitlines()[-1].strip()
            pattern = r"\[(.*?)\]"
            matches = re.findall(pattern, last_line)
            return matches[-1] if matches else last_line
        
        def extract_answer(generation):
            valid_format = ends_with_bracketed_line(generation)
            return extract_text_between_brackets(generation), valid_format

        total_points = []
        valid_formats = []
        model_answers = []
        chrf = sacrebleu.metrics.CHRF() 

        assert eval_types
        assert points
        assert len(predictions) == len(eval_types) == len(points)

        for pred, ref, eval_type, point in zip(predictions, references, eval_types, points):
            pred_norm = pred.strip().lower()
            ref_norm  = ref.strip().lower()
            model_answer, valid = extract_answer(pred_norm)
            valid_formats.append(valid)
            model_answers.append(model_answer)

            if eval_type.lower() == 'exact match':
                correct = float(ref_norm == model_answer)
            elif eval_type.lower() == 'chrf':
                correct = chrf.sentence_score(model_answer, [ref_norm]).score/100
            total_points.append(correct*point)

        return {
            'total_points': sum(total_points),
            'valid_format_rate': sum(valid_formats)/len(valid_formats),
            "per_example_stats": {
                "points": total_points,
                "valid_format": valid_formats,
                "extracted_answer": model_answers
            },
        }
