import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../third_party/PolyMath/eval"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../third_party/PolyMath"))

from run_eval import extract_boxed_content
from scripts import math_equal

from datasets import load_dataset, get_dataset_config_names
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Optional
import sacrebleu
import iso639
#from comet import download_model, load_from_checkpoint
try:
    from math_verify import parse, StringExtractionConfig, LatexExtractionConfig, verify
except ImportError:
    parse = StringExtractionConfig = LatexExtractionConfig = verify = None
import re
import subprocess
import tempfile
import os
from instruction import query_dic

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


    def prepare_system_prompt(self):
        """Prepare system prompt for a single example"""
        return None

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
        elif benchmark_type == 'mgsm':
            return MGSMBenchmark(task_config, subset)
        elif benchmark_type == 'xcopa':
            return XCOPABenchmark(task_config, subset)
        elif benchmark_type == 'xstory_cloze':
            return XStoryCloze(task_config, subset)
        elif benchmark_type == 'xnli':
            return XNLIBenchmark(task_config, subset)
        elif benchmark_type == 'americas_nli':
            return AmericasNLIBenchmark(task_config, subset)
        elif benchmark_type == 'sib200':
            return SIB200Benchmark(task_config, subset)
        elif benchmark_type == 'belebele':
            return BelebeleBenchmark(task_config, subset)
        elif benchmark_type == 'mkqa':
            return MKQABenchmark(task_config, subset)
        elif benchmark_type == 'olympiad_bench':
            return OlympiadBenchBenchmark(task_config, subset)
        elif benchmark_type == 'gpqa':
            return GPQABenchmark(task_config, subset)
        elif benchmark_type == 'livecodebench':
            return LiveCodeBenchBenchmark(task_config, subset)
        elif benchmark_type == 'frontiermath':
            return FrontierMathBenchmark(task_config, subset)
        elif benchmark_type == 'flores200':
            return FLORES200Benchmark(task_config, subset)
        elif benchmark_type == 'aime2025':
            return AIME2025Benchmark(task_config, subset)
        elif benchmark_type == 'absencebench':
            return AbsenceBenchmark(task_config, subset)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_type}")

    @staticmethod
    def get_all_subsets(benchmark_type, task_config):
        """Return all available subsets for benchmarks that use subsets: 'all'."""
        dataset_name = task_config['dataset_name']
        if benchmark_type == 'flores200':
            src_lang = task_config['defaults'].get('src_lang', 'eng_Latn')
            configs = get_dataset_config_names(dataset_name)
            return [c for c in configs if c != src_lang]
        elif benchmark_type == 'sib200':
            return get_dataset_config_names(dataset_name)
        else:
            return get_dataset_config_names(dataset_name)


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
            f"{query_dic[self.subset]}\n\n"
        )
    
    def evaluate(self, predictions, references, eval_types=None, points=None):
        """Evaluate MMMLU predictions"""
        correct = 0
        total = len(predictions)
        scores = []

        for pred_text, ref in zip(predictions, references):
            normalized = normalize_latex(pred_text)
            extracted = extract_boxed_content(normalized)
            extracted = extracted[0] if extracted else None
            is_correct = math_equal(extracted, ref)
            correct += int(is_correct)
            scores.append(1 if is_correct else 0)

        accuracy = round(correct / total * 100, 1) if total > 0 else 0

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'per_example_accuracy': scores,
        }
# ---------- AIME 2025 ---------
@dataclass
class AIMEExample:
    id: str
    question: str
    answer: str


class AIME2025Benchmark(BaseBenchmark):
    """AIME 2025 - MathArena dataset"""

    def load_data(self):
        name = self.task_config['dataset_name']  # MathArena/aime_2025
        split = self.task_config['split']        # train
        limit = self.defaults.get('limit_per_subset')

        print(f"Loading {name}...")
        ds = load_dataset(name, split=split)

        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        examples = []
        for i, r in enumerate(ds):
            examples.append(AIMEExample(
                id=f"aime2025_{i}",
                question=r["problem"],
                answer=str(r["answer"]),  # int in dataset, cast to str
            ))

        self.dataset = examples
        print(f"Loaded {len(examples)} examples")
        return examples

    def prepare_prompt(self, example):
        return (
            f"{example.question}\n"
            f"Note: Please put the final answer in the $\\boxed{{}}$.\n\n"
        )

    def evaluate(self, predictions, references, eval_types=None, points=None):
        correct = 0
        total = len(predictions)
        scores = []

        for pred_text, ref in zip(predictions, references):
            extracted = extract_boxed_answer(pred_text)
            is_correct = verify(ref, extracted)
            correct += int(is_correct)
            scores.append(1 if is_correct else 0)

        accuracy = round(correct / total * 100, 1) if total > 0 else 0

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'per_example_accuracy': scores,
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


def _chrf(hypothesis, reference):
    """Compute sentence-level chrF score."""
    return sacrebleu.sentence_chrf(hypothesis, [reference]).score / 100.0


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
            problem_id   = str(problem.get('id', ''))
            context      = problem.get('context', '')
            task_type    = problem.get('task_type', 'unknown')
            task_lang    = problem.get('task_lang', 'unknown')

            query = problem.get('query', '')
            answer_raw = problem.get('answer', '')
            eval_type = problem.get('eval_type', 'single')

            # Parse answer: stored as string repr of a list, e.g. "['a', 'b']"
            try:
                import ast
                answer_list = ast.literal_eval(answer_raw)
                if isinstance(answer_list, list):
                    answer = '\n'.join(str(a).strip() for a in answer_list)
                else:
                    answer = str(answer_list).strip()
            except (ValueError, SyntaxError):
                answer = str(answer_raw).strip()

            examples.append(LinguiniExample(
                id=problem_id,
                context=context,
                question=query,
                answer=str(answer).strip(),
                task_type=task_type,
                task_lang=task_lang,
                problem_id=problem_id,
                eval_type=eval_type,
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
        """Evaluate using exact match, line-level accuracy, and chrF score."""
        assert len(predictions) == len(references), (
            f"Mismatch: {len(predictions)} predictions vs {len(references)} references"
        )

        def normalize(text):
            """Normalize whitespace and case for comparison."""
            return ' '.join(text.strip().lower().split())

        def extract_answer_lines(text):
            """Extract clean answer lines, stripping numbering/bullets."""
            lines = []
            for line in text.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                # Strip common numbering: "1. ", "1) ", "(1) ", "- "
                cleaned = re.sub(r'^(\d+[\.\)]\s*|\(\d+\)\s*|-\s*)', '', line).strip()
                # Strip trailing whitespace artifacts
                cleaned = cleaned.rstrip()
                if cleaned:
                    lines.append(cleaned)
            return lines

        def format_check(pred, ref):
            """Check if prediction has consistent format.
            Returns (is_clean, has_reasoning_leak, line_count_match)."""
            reasoning_words = ['because', 'therefore', 'the pattern', 'let me',
                             'step 1', 'i notice', 'looking at', 'we can see',
                             'first,', 'analysis', 'observe that']
            has_leak = any(w in pred.lower() for w in reasoning_words)
            pred_lines = extract_answer_lines(pred)
            ref_lines = extract_answer_lines(ref)
            count_match = len(pred_lines) == len(ref_lines)
            is_clean = not has_leak
            return is_clean, not has_leak, count_match

        scores = []
        correct = 0
        chrf_scores = []
        line_correct_list = []
        line_total_list = []
        format_clean = 0
        format_line_match = 0

        for pred, ref in zip(predictions, references):
            pred_norm = normalize(pred)
            ref_norm = normalize(ref)

            # Exact match (full answer)
            match = int(pred_norm == ref_norm)
            scores.append(match)
            correct += match

            # chrF (full answer)
            chrf_scores.append(_chrf(pred_norm, ref_norm))

            # Line-level accuracy (per-answer-element)
            pred_lines = extract_answer_lines(pred)
            ref_lines = extract_answer_lines(ref)
            line_correct = 0
            line_total = len(ref_lines)
            for i, ref_line in enumerate(ref_lines):
                if i < len(pred_lines):
                    if normalize(pred_lines[i]) == normalize(ref_line):
                        line_correct += 1
            line_correct_list.append(line_correct)
            line_total_list.append(line_total)

            # Format verification
            is_clean, _, count_match = format_check(pred, ref)
            format_clean += int(is_clean)
            format_line_match += int(count_match)

        total = len(predictions)
        accuracy = (correct / total * 100) if total > 0 else 0.0
        chrf_avg = (sum(chrf_scores) / total * 100) if total > 0 else 0.0
        total_lines = sum(line_total_list)
        total_line_correct = sum(line_correct_list)
        line_accuracy = (total_line_correct / total_lines * 100) if total_lines > 0 else 0.0

        return {
            'accuracy': accuracy,
            'chrf': chrf_avg,
            'line_accuracy': line_accuracy,
            'correct': correct,
            'total': total,
            'line_correct': total_line_correct,
            'line_total': total_lines,
            'format_clean_rate': (format_clean / total * 100) if total > 0 else 0.0,
            'format_line_match_rate': (format_line_match / total * 100) if total > 0 else 0.0,
            "per_example_scores": {
                "accuracy": scores,
                "chrf": chrf_scores,
                "line_correct": line_correct_list,
                "line_total": line_total_list,
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

mulr_instruction = {
    'en': "\n\nIMPORTANT: You MUST put your final answer inside square brackets on the last line. Example: [your answer here]\nDo NOT write anything after the brackets. Only the text inside [...] will be graded.\n\n",
    'de': "\n\nWICHTIG: Sie MÜSSEN Ihre endgültige Antwort in eckige Klammern auf der letzten Zeile setzen. Beispiel: [Ihre Antwort hier]\nSchreiben Sie NICHTS nach den Klammern. Nur der Text innerhalb von [...] wird bewertet.\n\n",
    'zh': "\n\n重要：你必须将最终答案放在最后一行的方括号内。示例：[你的答案]\n方括号后不要写任何内容。只有 [...] 内的文本会被评分。\n\n",
    'fr': "\n\nIMPORTANT : Vous DEVEZ mettre votre réponse finale entre crochets sur la dernière ligne. Exemple : [votre réponse ici]\nN'écrivez RIEN après les crochets. Seul le texte entre [...] sera évalué.\n\n",
    'ja': "\n\n重要：最終回答は必ず最後の行の角括弧内に記入してください。例：[あなたの回答]\n括弧の後には何も書かないでください。[...] 内のテキストのみが採点されます。\n\n",
    'ko': "\n\n중요: 최종 답변을 반드시 마지막 줄에 대괄호 안에 넣어 주십시오. 예: [답변]\n대괄호 뒤에는 아무것도 쓰지 마십시오. [...] 안의 텍스트만 채점됩니다.\n\n",
    'pt': "\n\nIMPORTANTE: Você DEVE colocar sua resposta final entre colchetes na última linha. Exemplo: [sua resposta aqui]\nNÃO escreva nada após os colchetes. Apenas o texto dentro de [...] será avaliado.\n\n",
    'es': "\n\nIMPORTANTE: DEBE poner su respuesta final entre corchetes en la última línea. Ejemplo: [su respuesta aquí]\nNO escriba nada después de los corchetes. Solo el texto dentro de [...] será evaluado.\n\n"


}

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
            return iso639.to_iso639_1(example['problem_language']) == self.subset 
        
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
        return example.prompt + mulr_instruction[self.subset]

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
            if len(pred_norm) > 0:
                model_answer, valid = extract_answer(pred_norm)
            else:
                model_answer, valid = "", False
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
                "valid_formats": valid_formats,
                "extracted_answers": model_answers
            },
        }


# ---------- Shared helpers for new benchmarks ----------

def extract_binary_choice(text):
    """Extract 1 or 2 from model output (XCOPA, XStoryCloze)."""
    for char in text.strip():
        if char in "12":
            return char
    return ""


def extract_integer_answer(text):
    """Extract the last integer from model output (MGSM)."""
    matches = re.findall(r'-?\d+', text.replace(',', ''))
    return matches[-1] if matches else ""


def extract_nli_label(text):
    """Extract NLI label from model output (AmericasNLI)."""
    text_lower = text.strip().lower()
    for label in ["entailment", "contradiction", "neutral"]:
        if label in text_lower:
            return label
    return ""


def extract_sib200_topic(text):
    """Extract SIB-200 topic category from model output."""
    text_lower = text.strip().lower()
    for topic in ["science_technology", "travel", "politics",
                  "sports", "health", "entertainment", "geography"]:
        if topic in text_lower:
            return topic
    return ""


def extract_code_block(text):
    """Strip markdown code fences from model output (LiveCodeBench)."""
    match = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    return match.group(1) if match else text


def extract_aime_answer(text):
    """Extract integer answer for AIME, trying \\boxed{} first."""
    boxed = extract_boxed_answer(text)
    if boxed:
        try:
            val = re.sub(r'[^\d-]', '', str(boxed[0]) if isinstance(boxed, list) else str(boxed))
            return str(int(val)) if val else ""
        except Exception:
            pass
    matches = re.findall(r'\b(\d{1,3})\b', text)
    return matches[-1] if matches else ""


def _run_python_code(code, test_input, timeout=10):
    """Execute Python code with given stdin; return stdout or error sentinel."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            ["python3", fname], input=test_input,
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "__TIMEOUT__"
    except Exception:
        return "__ERROR__"
    finally:
        os.unlink(fname)


# ---------- MGSM ----------

@dataclass
class MGSMExample:
    id: str
    question: str
    answer: str


class MGSMBenchmark(BaseBenchmark):
    """Multilingual Grade School Math benchmark (juletxara/mgsm)."""

    def load_data(self):
        name = self.task_config['dataset_name']
        split = self.task_config['split']
        limit = self.defaults.get('limit_per_subset')

        print(f"Loading {name} ({self.subset})...")
        ds = load_dataset(name, self.subset, split=split)

        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        examples = []
        for i, r in enumerate(ds):
            examples.append(MGSMExample(
                id=f"{self.subset}_{i}",
                question=r["question"],
                answer=str(r["answer_number"]),
            ))

        self.dataset = examples
        print(f"Loaded {len(examples)} examples from {self.subset}")
        return examples

    def prepare_prompt(self, example):
        return (
            "Solve the following math problem step by step. "
            "At the end, write only the final integer answer on the last line.\n\n"
            f"{example.question}\n\n"
            "Answer:"
        )

    def evaluate(self, predictions, references, eval_types=None, points=None):
        correct = 0
        scores = []
        for pred, ref in zip(predictions, references):
            extracted = extract_integer_answer(pred)
            match = int(extracted == ref.strip())
            scores.append(match)
            correct += match
        total = len(predictions)
        return {
            'accuracy': (correct / total * 100) if total > 0 else 0,
            'correct': correct,
            'total': total,
            'per_example_accuracy': scores,
        }


# ---------- XCOPA ----------

@dataclass
class XCOPAExample:
    id: str
    premise: str
    choice1: str
    choice2: str
    question: str   # "cause" or "effect"
    answer: str     # "1" or "2"


class XCOPABenchmark(BaseBenchmark):
    """Cross-lingual Choice Of Plausible Alternatives (xcopa)."""

    def load_data(self):
        name = self.task_config['dataset_name']
        split = self.task_config['split']
        limit = self.defaults.get('limit_per_subset')

        print(f"Loading {name} ({self.subset})...")
        ds = load_dataset(name, self.subset, split=split)

        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        examples = []
        for i, r in enumerate(ds):
            examples.append(XCOPAExample(
                id=f"{self.subset}_{i}",
                premise=r["premise"],
                choice1=r["choice1"],
                choice2=r["choice2"],
                question=r["question"],
                answer=str(r["label"] + 1),  # 0-indexed → "1" or "2"
            ))

        self.dataset = examples
        print(f"Loaded {len(examples)} examples from {self.subset}")
        return examples

    def prepare_prompt(self, example):
        connector = (
            "What was the CAUSE of this?"
            if example.question == "cause"
            else "What happened as a RESULT?"
        )
        return (
            f"Premise: {example.premise}\n"
            f"{connector}\n\n"
            f"Choice 1: {example.choice1}\n"
            f"Choice 2: {example.choice2}\n\n"
            "Answer with only the number 1 or 2."
        )

    def evaluate(self, predictions, references, eval_types=None, points=None):
        correct = 0
        scores = []
        for pred, ref in zip(predictions, references):
            extracted = extract_binary_choice(pred)
            match = int(extracted == ref.strip())
            scores.append(match)
            correct += match
        total = len(predictions)
        return {
            'accuracy': (correct / total * 100) if total > 0 else 0,
            'correct': correct,
            'total': total,
            'per_example_accuracy': scores,
        }


# ---------- XStoryCloze ----------

@dataclass
class XStoryClozeExample:
    id: str
    context: str
    choice1: str
    choice2: str
    question: str   # constant sentinel for eval.py routing
    answer: str     # "1" or "2"


class XStoryCloze(BaseBenchmark):
    """Cross-lingual Story Cloze benchmark (juletxara/xstory_cloze)."""

    def load_data(self):
        name = self.task_config['dataset_name']
        split = self.task_config['split']
        limit = self.defaults.get('limit_per_subset')

        print(f"Loading {name} ({self.subset})...")
        ds = load_dataset(name, self.subset, split=split)

        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        examples = []
        for i, r in enumerate(ds):
            context = " ".join([
                r["input_sentence_1"], r["input_sentence_2"],
                r["input_sentence_3"], r["input_sentence_4"],
            ])
            examples.append(XStoryClozeExample(
                id=f"{self.subset}_{i}",
                context=context,
                choice1=r["sentence_quiz1"],
                choice2=r["sentence_quiz2"],
                question="story_cloze",
                answer=str(r["answer_right_ending"]),  # already 1 or 2
            ))

        self.dataset = examples
        print(f"Loaded {len(examples)} examples from {self.subset}")
        return examples

    def prepare_prompt(self, example):
        return (
            "Read the following story and choose the best ending sentence.\n\n"
            f"Story: {example.context}\n\n"
            f"Ending 1: {example.choice1}\n"
            f"Ending 2: {example.choice2}\n\n"
            "Answer with only the number 1 or 2."
        )

    def evaluate(self, predictions, references, eval_types=None, points=None):
        correct = 0
        scores = []
        for pred, ref in zip(predictions, references):
            extracted = extract_binary_choice(pred)
            match = int(extracted == ref.strip())
            scores.append(match)
            correct += match
        total = len(predictions)
        return {
            'accuracy': (correct / total * 100) if total > 0 else 0,
            'correct': correct,
            'total': total,
            'per_example_accuracy': scores,
        }


# ---------- AmericasNLI ----------

_NLI_LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}


@dataclass
class AmericasNLIExample:
    id: str
    premise: str
    hypothesis: str
    question: str   # constant sentinel
    answer: str     # "entailment", "neutral", or "contradiction"


class AmericasNLIBenchmark(BaseBenchmark):
    """Natural Language Inference for indigenous American languages (americas_nli)."""

    def load_data(self):
        name = self.task_config['dataset_name']
        split = self.task_config['split']
        limit = self.defaults.get('limit_per_subset')

        print(f"Loading {name} ({self.subset})...")
        ds = load_dataset(name, self.subset, split=split)

        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        examples = []
        for i, r in enumerate(ds):
            examples.append(AmericasNLIExample(
                id=f"{self.subset}_{i}",
                premise=r["premise"],
                hypothesis=r["hypothesis"],
                question="nli",
                answer=_NLI_LABEL_MAP[r["label"]],
            ))

        self.dataset = examples
        print(f"Loaded {len(examples)} examples from {self.subset}")
        return examples

    def prepare_prompt(self, example):
        return (
            "Determine the logical relationship between the following two sentences.\n\n"
            f"Premise: {example.premise}\n"
            f"Hypothesis: {example.hypothesis}\n\n"
            "Does the premise entail, contradict, or is neutral towards the hypothesis?\n"
            "Answer with only one word: entailment, neutral, or contradiction."
        )

    def evaluate(self, predictions, references, eval_types=None, points=None):
        correct = 0
        scores = []
        for pred, ref in zip(predictions, references):
            extracted = extract_nli_label(pred)
            match = int(extracted == ref.strip())
            scores.append(match)
            correct += match
        total = len(predictions)
        return {
            'accuracy': (correct / total * 100) if total > 0 else 0,
            'correct': correct,
            'total': total,
            'per_example_accuracy': scores,
        }


# ---------- SIB-200 ----------

_SIB200_TOPICS = [
    "science_technology", "travel", "politics",
    "sports", "health", "entertainment", "geography",
]


@dataclass
class SIB200Example:
    id: str
    text: str
    question: str   # constant sentinel
    answer: str     # one of 7 topic strings


class SIB200Benchmark(BaseBenchmark):
    """Topic classification across 200 languages (Davlan/sib200)."""

    def load_data(self):
        name = self.task_config['dataset_name']
        split = self.task_config['split']
        limit = self.defaults.get('limit_per_subset')

        print(f"Loading {name} ({self.subset})...")
        ds = load_dataset(name, self.subset, split=split)

        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        examples = []
        for i, r in enumerate(ds):
            examples.append(SIB200Example(
                id=f"{self.subset}_{i}",
                text=r["text"],
                question="topic_classification",
                answer=r["category"],
            ))

        self.dataset = examples
        print(f"Loaded {len(examples)} examples from {self.subset}")
        return examples

    def prepare_prompt(self, example):
        topics = ", ".join(_SIB200_TOPICS)
        return (
            f"Classify the following text into one of these topics:\n{topics}\n\n"
            f"Text: {example.text}\n\n"
            "Answer with only the topic name."
        )

    def evaluate(self, predictions, references, eval_types=None, points=None):
        correct = 0
        scores = []
        for pred, ref in zip(predictions, references):
            extracted = extract_sib200_topic(pred)
            match = int(extracted == ref.strip())
            scores.append(match)
            correct += match
        total = len(predictions)
        return {
            'accuracy': (correct / total * 100) if total > 0 else 0,
            'correct': correct,
            'total': total,
            'per_example_accuracy': scores,
        }


# ---------- MKQA ----------

@dataclass
class MKQAExample:
    id: str
    question: str
    answer: str     # pipe-delimited gold answers, or "__NULL__" if unanswerable


class MKQABenchmark(BaseBenchmark):
    """Multilingual Knowledge Questions and Answers (mkqa)."""

    def load_data(self):
        name = self.task_config['dataset_name']
        split = self.task_config['split']
        limit = self.defaults.get('limit_per_subset')

        print(f"Loading {name} ({self.subset})...")
        ds = load_dataset(name, split=split)

        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        examples = []
        for i, r in enumerate(ds):
            lang_answers = r["answers"].get(self.subset, [])
            all_texts = []
            for ans in lang_answers:
                if ans.get("text"):
                    all_texts.append(ans["text"])
                all_texts.extend(ans.get("aliases", []))
            all_texts = [t for t in all_texts if t]

            examples.append(MKQAExample(
                id=f"{self.subset}_{i}",
                question=r["query"],
                answer="|".join(all_texts) if all_texts else "__NULL__",
            ))

        self.dataset = examples
        print(f"Loaded {len(examples)} examples from {self.subset}")
        return examples

    def prepare_prompt(self, example):
        return (
            "Answer the following question briefly. "
            "If you cannot answer, write \"unanswerable\".\n\n"
            f"Question: {example.question}\n\n"
            "Answer:"
        )

    def evaluate(self, predictions, references, eval_types=None, points=None):
        _NULL_INDICATORS = {"unanswerable", "none", "unknown", "n/a", ""}
        correct = 0
        scores = []
        for pred, ref in zip(predictions, references):
            pred_norm = pred.strip().lower()
            if ref == "__NULL__":
                match = int(
                    any(ni in pred_norm for ni in _NULL_INDICATORS)
                    or len(pred_norm) < 5
                )
            else:
                gold_set = [g.strip().lower() for g in ref.split("|")]
                match = int(
                    pred_norm in gold_set
                    or any(g in pred_norm or pred_norm in g for g in gold_set if g)
                )
            scores.append(match)
            correct += match
        total = len(predictions)
        return {
            'accuracy': (correct / total * 100) if total > 0 else 0,
            'correct': correct,
            'total': total,
            'per_example_accuracy': scores,
        }


# ---------- LiveCodeBench ----------

@dataclass
class LiveCodeBenchExample:
    id: str
    question: str
    answer: str         # sentinel "__code__"
    test_cases: list    # list of {"input": str, "output": str}


class LiveCodeBenchBenchmark(BaseBenchmark):
    """Code generation with pass@1 evaluation (livecodebench/code_generation_lite)."""

    def __init__(self, task_config, subset):
        super().__init__(task_config, subset)
        self.test_cases_per_example = []

    def load_data(self):
        name = self.task_config['dataset_name']
        split = self.task_config['split']
        limit = self.defaults.get('limit_per_subset')

        print(f"Loading {name} ({self.subset})...")
        ds = load_dataset(name, split=split)

        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        self.test_cases_per_example = []
        examples = []
        for i, r in enumerate(ds):
            tests = r.get("public_tests", [])
            self.test_cases_per_example.append(tests)
            examples.append(LiveCodeBenchExample(
                id=f"{self.subset}_{i}",
                question=r["question_content"],
                answer="__code__",
                test_cases=tests,
            ))

        self.dataset = examples
        print(f"Loaded {len(examples)} examples from {self.subset}")
        return examples

    def prepare_prompt(self, example):
        return (
            "Write a Python solution for the following competitive programming problem. "
            "Return ONLY the Python code without any explanation or markdown formatting.\n\n"
            f"{example.question}"
        )

    def evaluate(self, predictions, references, eval_types=None, points=None):
        scores = []
        for i, pred in enumerate(predictions):
            code = extract_code_block(pred)
            tests = self.test_cases_per_example[i] if i < len(self.test_cases_per_example) else []
            if not tests:
                scores.append(0)
                continue
            passed = sum(
                1 for tc in tests
                if _run_python_code(code, tc.get("input", "")) == tc.get("output", "").strip()
            )
            scores.append(1 if passed == len(tests) else 0)

        correct = sum(scores)
        total = len(predictions)
        return {
            'pass_at_1': (correct / total * 100) if total > 0 else 0,
            'correct': correct,
            'total': total,
            'per_example_accuracy': scores,
        }


# ---------- FrontierMath ----------

@dataclass
class FrontierMathExample:
    id: str
    question: str
    answer: str


class FrontierMathBenchmark(BaseBenchmark):
    """Expert-level mathematics benchmark (EleutherAI/frontiermath). Requires HF access approval."""

    def load_data(self):
        name = self.task_config['dataset_name']
        split = self.task_config['split']
        limit = self.defaults.get('limit_per_subset')

        print(f"Loading {name} ({self.subset})...")
        try:
            ds = load_dataset(name, split=split)
        except Exception as e:
            raise RuntimeError(
                f"Could not load FrontierMath dataset '{name}'. "
                "This dataset requires approved access at "
                "https://huggingface.co/datasets/EleutherAI/frontiermath\n"
                f"Original error: {e}"
            )

        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        examples = []
        for i, r in enumerate(ds):
            examples.append(FrontierMathExample(
                id=f"{self.subset}_{i}",
                question=r["problem"],
                answer=r["answer"],
            ))

        self.dataset = examples
        print(f"Loaded {len(examples)} examples from {self.subset}")
        return examples

    def prepare_prompt(self, example):
        return (
            "Solve the following advanced mathematics problem. "
            "Express your final answer in \\boxed{}.\n\n"
            f"{example.question}"
        )

    def evaluate(self, predictions, references, eval_types=None, points=None):
        correct = 0
        scores = []
        extracted_predictions = [extract_boxed_answer(pred) for pred in predictions]
        for pred, ref in zip(extracted_predictions, references):
            if verify(ref, pred):
                correct += 1
                scores.append(1)
            else:
                scores.append(0)
        total = len(predictions)
        return {
            'accuracy': (correct / total * 100) if total > 0 else 0,
            'correct': correct,
            'total': total,
            'per_example_accuracy': scores,
        }


# ---------- FLORES-200 ----------

FLORES_LANG_NAMES = {
    "deu_Latn": ("German", "Germany"),
    "fra_Latn": ("French", "France"),
    "spa_Latn": ("Spanish", "Spain"),
    "zho_Hans": ("Mandarin Chinese", "China"),
    "zho_Hant": ("Traditional Chinese", "Taiwan"),
    "arb_Arab": ("Arabic", "Middle East"),
    "ben_Beng": ("Bengali", "Bangladesh"),
    "rus_Cyrl": ("Russian", "Russia"),
    "jpn_Jpan": ("Japanese", "Japan"),
    "swh_Latn": ("Swahili", "East Africa"),
    "tur_Latn": ("Turkish", "Turkey"),
    "kor_Hang": ("Korean", "South Korea"),
    "hin_Deva": ("Hindi", "India"),
    "vie_Latn": ("Vietnamese", "Vietnam"),
    "por_Latn": ("Portuguese", "Portugal"),
    "ind_Latn": ("Indonesian", "Indonesia"),
    "ita_Latn": ("Italian", "Italy"),
    "pol_Latn": ("Polish", "Poland"),
    "nld_Latn": ("Dutch", "Netherlands"),
    "tha_Thai": ("Thai", "Thailand"),
    "ron_Latn": ("Romanian", "Romania"),
    "ukr_Cyrl": ("Ukrainian", "Ukraine"),
    "ces_Latn": ("Czech", "Czechia"),
    "fin_Latn": ("Finnish", "Finland"),
    "heb_Hebr": ("Hebrew", "Israel"),
    "tel_Telu": ("Telugu", "India"),
    "tam_Taml": ("Tamil", "India"),
    "mlt_Latn": ("Maltese", "Malta"),
    "amh_Ethi": ("Amharic", "Ethiopia"),
    "yor_Latn": ("Yoruba", "Nigeria"),
    "swa_Latn": ("Swahili", "East Africa"),
}


class FLORES200Benchmark(BaseBenchmark):
    """FLORES-200 machine translation benchmark (Muennighoff/flores200)."""

    def __init__(self, task_config, subset):
        super().__init__(task_config, subset)
        self.comet_model = None
        self.include_comet = task_config['defaults'].get("include_comet", False)
        self.sources = []

    def load_data(self):
        name = self.task_config['dataset_name']
        split = self.task_config['split']
        limit = self.defaults.get('limit_per_subset')
        src_lang = self.defaults.get('src_lang', 'eng_Latn')

        print(f"Loading {name} ({src_lang} → {self.subset})...")
        src_ds = load_dataset(name, src_lang, split=split)
        tgt_ds = load_dataset(name, self.subset, split=split)

        assert len(src_ds) == len(tgt_ds), (
            f"Source/target length mismatch: {len(src_ds)} vs {len(tgt_ds)}"
        )

        if limit:
            src_ds = src_ds.select(range(min(limit, len(src_ds))))
            tgt_ds = tgt_ds.select(range(min(limit, len(tgt_ds))))

        examples = []
        for i, (src_row, tgt_row) in enumerate(zip(src_ds, tgt_ds)):
            examples.append(MTExample(
                id=f"{self.subset}_{i}",
                source=src_row["sentence"],
                reference=tgt_row["sentence"],
            ))

        self.sources = [ex.source for ex in examples]
        self.dataset = examples
        print(f"Loaded {len(examples)} examples from {self.subset}")
        return examples

    def prepare_prompt(self, example):
        tgt_lang, tgt_region = FLORES_LANG_NAMES.get(self.subset, (self.subset, self.subset))
        return WMT24PP_PROMPT.format(
            src_lang="English",
            tgt_lang=tgt_lang,
            tgt_region=tgt_region,
            tgt_code=self.subset,
            input_text=example.source,
        )

    def evaluate(self, predictions, references, eval_types=None, points=None):
        assert len(predictions) == len(references), (
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )

        per_example_bleu = []
        per_example_chrf = []
        for pred, ref in zip(predictions, references):
            if not pred or not pred.strip():
                per_example_bleu.append(None)
                per_example_chrf.append(None)
            else:
                per_example_bleu.append(sacrebleu.sentence_bleu(pred, [ref]).score)
                per_example_chrf.append(sacrebleu.sentence_chrf(pred, [ref], word_order=2).score)

        valid = [
            (pred, ref, src)
            for pred, ref, src in zip(predictions, references, self.sources)
            if pred and pred.strip()
        ]

        if not valid:
            return {
                "bleu": 0.0, "chrfpp": 0.0, "num_predictions": len(predictions),
                "per_example_scores": {"bleu": per_example_bleu, "chrfpp": per_example_chrf},
            }

        if len(valid) < len(predictions):
            print(f"Warning: {len(predictions) - len(valid)} empty predictions excluded")

        valid_preds, valid_refs, _ = map(list, zip(*valid))
        bleu = sacrebleu.corpus_bleu(valid_preds, [valid_refs])
        chrf = sacrebleu.corpus_chrf(valid_preds, [valid_refs], word_order=2)

        return {
            "bleu": bleu.score,
            "chrfpp": chrf.score,
            "num_predictions": len(predictions),
            "per_example_scores": {"bleu": per_example_bleu, "chrfpp": per_example_chrf},
        }


# ---------- XNLI ----------

_NLI_LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}


@dataclass
class XNLIExample:
    id: str
    question: str   # premise + hypothesis concatenated for source logging
    premise: str
    hypothesis: str
    answer: str     # "entailment", "neutral", or "contradiction"


class XNLIBenchmark(BaseBenchmark):
    """Cross-lingual Natural Language Inference (xnli)."""

    def load_data(self):
        name = self.task_config['dataset_name']
        split = self.task_config['split']
        limit = self.defaults.get('limit_per_subset')

        print(f"Loading {name} ({self.subset})...")
        ds = load_dataset(name, self.subset, split=split)

        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        examples = []
        for i, r in enumerate(ds):
            examples.append(XNLIExample(
                id=f"{self.subset}_{i}",
                question=f"{r['premise']} | {r['hypothesis']}",
                premise=r["premise"],
                hypothesis=r["hypothesis"],
                answer=_NLI_LABEL_MAP[r["label"]],
            ))

        self.dataset = examples
        print(f"Loaded {len(examples)} examples from {self.subset}")
        return examples

    def prepare_prompt(self, example):
        return (
            "Determine the logical relationship between the following two sentences.\n\n"
            f"Premise: {example.premise}\n"
            f"Hypothesis: {example.hypothesis}\n\n"
            "Does the premise entail, contradict, or is neutral towards the hypothesis?\n"
            "Answer with only one word: entailment, neutral, or contradiction."
        )

    def evaluate(self, predictions, references, eval_types=None, points=None):
        correct = 0
        scores = []
        for pred, ref in zip(predictions, references):
            extracted = extract_nli_label(pred)
            match = int(extracted == ref.strip())
            scores.append(match)
            correct += match
        total = len(predictions)
        return {
            'accuracy': (correct / total * 100) if total > 0 else 0,
            'correct': correct,
            'total': total,
            'per_example_accuracy': scores,
        }


# ---------- Belebele ----------

_NUM_TO_LETTER = {1: "A", 2: "B", 3: "C", 4: "D", "1": "A", "2": "B", "3": "C", "4": "D"}


@dataclass
class BelebeleExample:
    id: str
    question: str   # passage + question for source logging
    passage: str
    q: str
    A: str
    B: str
    C: str
    D: str
    answer: str     # "A", "B", "C", or "D"


class BelebeleBenchmark(BaseBenchmark):
    """Belebele: multilingual reading comprehension MCQ (facebook/belebele)."""

    def load_data(self):
        name = self.task_config['dataset_name']
        split = self.task_config['split']
        limit = self.defaults.get('limit_per_subset')

        print(f"Loading {name} ({self.subset})...")
        ds = load_dataset(name, self.subset, split=split)

        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        examples = []
        for i, r in enumerate(ds):
            correct_letter = _NUM_TO_LETTER[r["correct_answer_num"]]
            examples.append(BelebeleExample(
                id=f"{self.subset}_{i}",
                question=f"{r['flores_passage']}\n{r['question']}",
                passage=r["flores_passage"],
                q=r["question"],
                A=r["mc_answer1"],
                B=r["mc_answer2"],
                C=r["mc_answer3"],
                D=r["mc_answer4"],
                answer=correct_letter,
            ))

        self.dataset = examples
        print(f"Loaded {len(examples)} examples from {self.subset}")
        return examples

    def prepare_prompt(self, example):
        return (
            f"Read the passage and answer the question by selecting the correct option.\n\n"
            f"Passage: {example.passage}\n\n"
            f"Question: {example.q}\n"
            f"A) {example.A}\nB) {example.B}\nC) {example.C}\nD) {example.D}\n\n"
            "Answer with only the letter (A, B, C, or D)."
        )

    def evaluate(self, predictions, references, eval_types=None, points=None):
        correct = 0
        scores = []
        for pred, ref in zip(predictions, references):
            extracted = extract_choice(pred).upper()
            match = int(extracted == ref.strip().upper())
            scores.append(match)
            correct += match
        total = len(predictions)
        return {
            'accuracy': (correct / total * 100) if total > 0 else 0,
            'correct': correct,
            'total': total,
            'per_example_accuracy': scores,
        }


# ---- AbsenceBench ----
@dataclass
class AbsenceBenchExample:
    id: str
    original_context: str
    modified_context: str
    omitted_context: list   # list of strings (poetry, github_prs)
    omitted_index: list     # list of ints
    metadata: dict

ABSENCE_SYSTEM_PROMPTS = {
    "poetry": (
        "You are helping a student practice memorizing poems. \n"
        "The student will recite a poem, but they may have missed some lines. \n"
        "Your task is to identify exactly which lines are missing from their recitation.\n"
        "List only the missing lines, nothing else."
    ),
    "numerical": (
        "You are helping a student practice reciting sequences. \n"
        "The student will recite a sequence, but they may have missed some numbers. \n"
        "Your task is to identify exactly which numbers are missing from their recitation.\n"
        "List only the missing numbers, nothing else."
    ),
    "github_prs": (
        "You are helping a software developer determine if their merge"
        " of a pull request was successful. "
        "The developer had to edit the commit history and just wants to make sure"
        " that they have not changed what will be merged. "
        "They will list the changed lines. "
        "Your job is to figure out if they have missed any "
        "insertions or deletions from the original merge. "
        "Only pay attention to the insertions and deletions (ignore the context of the diff)."
    ),
}

# ── Official evaluate_response per domain ─────────────────────────────────────

def evaluate_response_poetry(response: str, example: AbsenceBenchExample) -> dict:
    """Exact port of test_llms_poetry.py evaluate_response (non-needle mode)."""
    original_lines = example.original_context.split('\n')
    omitted_indices = set(example.omitted_index)

    tp, fp, fn = 0, 0, 0

    for idx, line in enumerate(original_lines):
        clean_line = line.strip().lower()
        if clean_line and clean_line in response.lower():
            if idx in omitted_indices:
                tp += 1
            else:
                fp += 1
        elif clean_line and clean_line not in response.lower():
            if idx in omitted_indices:   # ← only count as FN if it was omitted
                fn += 1

    try:
        micro_f1 = 2 * tp / (2 * tp + fp + fn)
    except ZeroDivisionError:
        micro_f1 = 0

    if len(example.omitted_index) == 0:
        micro_f1 = 1 - fp / len(original_lines)

    return {"tp": tp, "fp": fp, "fn": fn, "micro_f1": micro_f1}


def evaluate_response_numerical(response: str, example: AbsenceBenchExample) -> dict:
    """Exact port of test_llms_numerical.py evaluate_response."""
    og_sequence = example.original_context.split('\n')
    omitted_indices = set(example.omitted_index)

    tp, fp, fn = 0, 0, 0
    response_lines = [x.strip() for x in response.split('\n')]

    for idx, element in enumerate(og_sequence):
        str_element = str(element)
        if str_element in response_lines:
            if idx in omitted_indices:
                tp += 1
            else:
                fp += 1
        else:
            if idx in omitted_indices:
                fn += 1

    try:
        micro_f1 = 2 * tp / (2 * tp + fp + fn)
    except ZeroDivisionError:
        micro_f1 = 0

    if len(example.omitted_index) == 0:
        micro_f1 = 1 - fp / len(og_sequence)

    return {"tp": tp, "fp": fp, "fn": fn, "micro_f1": micro_f1}


def evaluate_response_github(response: str, example: AbsenceBenchExample) -> dict:
    """Exact port of test_llms_github_prs.py evaluate_response (non-needle mode)."""
    original_lines = example.original_context.split('\n')
    omitted_indices = set(example.omitted_index)

    tp, fp, fn = 0, 0, 0

    # handle repeated lines as FP
    repeat_lines = list(set([l for l in original_lines if original_lines.count(l) != 1]))
    for line in repeat_lines:
        line_count = min(
            response.lower().count("\n" + line.strip().lower() + "\n"),
            original_lines.count(line)
        )
        fp += line_count

    for idx, line in enumerate(original_lines):
        if line in repeat_lines:
            continue
        clean_line = line.strip().lower()
        if clean_line and clean_line in response.lower():
            if idx in omitted_indices:
                tp += 1
            else:
                fp += 1
        elif clean_line and clean_line not in response.lower():
            if idx in omitted_indices:   # ← only count as FN if it was omitted
                fn += 1

    try:
        micro_f1 = 2 * tp / (2 * tp + fp + fn)
    except ZeroDivisionError:
        micro_f1 = 0

    if len(example.omitted_index) == 0:
        micro_f1 = 1 - fp / len(original_lines)

    return {"tp": tp, "fp": fp, "fn": fn, "micro_f1": micro_f1}


# ── Dispatcher ─────────────────────────────────────────────────────────────────

EVALUATE_FN = {
    "poetry":     evaluate_response_poetry,
    "numerical":  evaluate_response_numerical,
    "github_prs": evaluate_response_github,
}

class AbsenceBenchmark(BaseBenchmark):
    """AbsenceBench: Language Models Can't Tell What's Missing
    Subsets: poetry, numerical, github_prs
    """

    def load_data(self):
        name  = self.task_config['dataset_name']  # harveyfin/AbsenceBench
        split = self.task_config['split']         # validation
        limit = self.defaults.get('limit_per_subset')

        print(f"Loading {name} ({self.subset})...")
        ds = load_dataset(name, self.subset, split=split)

        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        examples = []
        for i, r in enumerate(ds):
            examples.append(AbsenceBenchExample(
                id=f"{self.subset}_{i}",
                original_context=r["original_context"],
                modified_context=r["modified_context"],
                omitted_context=r.get("omitted_context", []),
                omitted_index=r["omitted_index"],
                metadata=r.get("metadata", {}),
            ))

        self.dataset = examples
        print(f"Loaded {len(examples)} examples from {self.subset}")
        return examples

    def prepare_prompt(self, example):
        if self.subset == "poetry":
            return (
                f"Here is the complete original poem:\n\n"
                f"{example.original_context}\n\n"
                f"Now, here is my recitation which may be missing some lines:\n\n"
                f"{example.modified_context}\n\n"
                f"What lines did I miss? Please list only the missing lines, nothing else."
            )
        elif self.subset == "numerical":
            return (
                f"Here is a sequence of numbers:\n\n"
                f"{example.original_context}\n\n"
                f"Now, here is my recitation of the sequence which may be missing some numbers:\n\n"
                f"{example.modified_context}\n\n"
                f"What numbers did I miss? Please list only the missing numbers, nothing else."
            )
        elif self.subset == "github_prs":
            return (
                f"Here is the complete original diff:\n\n"
                f"{example.original_context}\n\n"
                f"And here is the merge diff after the developer fixed the commit history:\n\n"
                f"{example.modified_context}\n\n"
                f"What changed lines (insertions or deletions) present "
                f"in the original diff are missing in the merge diff (if any)?\n"
                f"List only the missing changed lines, nothing else."
            )

    def prepare_system_prompt(self):
        return ABSENCE_SYSTEM_PROMPTS[self.subset]

    def evaluate(self, predictions, references, eval_types=None, points=None):
        """
        predictions: list of raw model output strings
        references:  list of AbsenceBenchExample (passed through from runner)
                     OR list of omitted_index lists — depends on your runner.
        Note: we need the full example for evaluation, not just references.
        So references here should be the list of AbsenceBenchExample objects.
        """
        evaluate_fn = EVALUATE_FN[self.subset]
        total = len(predictions)

        total_tp, total_fp, total_fn = 0, 0, 0
        per_example_f1 = []

        for pred_text, example in zip(predictions, self.dataset):
            result = evaluate_fn(pred_text, example)
            total_tp += result["tp"]
            total_fp += result["fp"]
            total_fn += result["fn"]
            per_example_f1.append(result["micro_f1"])

        macro_f1 = sum(per_example_f1) / total if total > 0 else 0.0

        try:
            micro_f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn)
        except ZeroDivisionError:
            micro_f1 = 0.0

        return {
            'f1':             round(macro_f1 * 100, 1),
            'micro_f1':       round(micro_f1 * 100, 1),
            'total':          total,
            'per_example_f1': per_example_f1,
        }


# ---------- OlympiadBench ----------

_OLYMPIAD_CONFIG_MAP = {
    "en": "OE_TO_maths_en_COMP",
    "zh": "OE_TO_maths_zh_COMP",
}


@dataclass
class OlympiadExample:
    id: str
    question: str
    answer: str


class OlympiadBenchBenchmark(BaseBenchmark):
    """OlympiadBench: bilingual olympiad mathematics (GAIR/OlympiadBench)."""

    def load_data(self):
        name = self.task_config['dataset_name']
        split = self.task_config['split']
        limit = self.defaults.get('limit_per_subset')
        config = _OLYMPIAD_CONFIG_MAP.get(self.subset, self.subset)

        print(f"Loading {name} ({config})...")
        ds = load_dataset(name, config, split=split)

        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        examples = []
        for i, r in enumerate(ds):
            final_answer = r.get("final_answer", [""])
            answer = final_answer[0] if isinstance(final_answer, list) and final_answer else str(final_answer)
            examples.append(OlympiadExample(
                id=f"{self.subset}_{i}",
                question=r["question"],
                answer=answer,
            ))

        self.dataset = examples
        print(f"Loaded {len(examples)} examples from {self.subset}")
        return examples

    def prepare_prompt(self, example):
        instruction = poly_math_instruction.get(self.subset, poly_math_instruction['en'])
        return f"{example.question}\n{instruction}\n\n"

    def evaluate(self, predictions, references, eval_types=None, points=None):
        correct = 0
        scores = []
        for pred, ref in zip(predictions, references):
            pred_parsed = extract_boxed_answer(pred)
            ref_parsed = extract_boxed_answer(f"\\boxed{{{ref}}}")
            match = int(verify(ref_parsed, pred_parsed))
            scores.append(match)
            correct += match
        total = len(predictions)
        return {
            'accuracy': (correct / total * 100) if total > 0 else 0,
            'correct': correct,
            'total': total,
            'per_example_accuracy': scores,
        }


# ---------- GPQA ----------

@dataclass
class GPQAExample:
    id: str
    question: str
    A: str
    B: str
    C: str
    D: str
    answer: str     # "A", "B", "C", or "D"


class GPQABenchmark(BaseBenchmark):
    """GPQA Diamond: graduate-level science MCQ (Idavidrein/gpqa)."""

    def load_data(self):
        name = self.task_config['dataset_name']
        split = self.task_config.get('split', 'train')
        limit = self.defaults.get('limit_per_subset')

        print(f"Loading {name} (gpqa_diamond)...")
        ds = load_dataset(name, "gpqa_diamond", split=split)

        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        examples = []
        for i, r in enumerate(ds):
            correct = r["Correct Answer"]
            wrongs = [r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            # Deterministic ordering: sort all 4 choices alphabetically
            all_opts = sorted([correct] + wrongs)
            letter_map = {opt: chr(ord('A') + j) for j, opt in enumerate(all_opts)}
            examples.append(GPQAExample(
                id=f"gpqa_{i}",
                question=r["Question"],
                A=all_opts[0],
                B=all_opts[1],
                C=all_opts[2],
                D=all_opts[3],
                answer=letter_map[correct],
            ))

        self.dataset = examples
        print(f"Loaded {len(examples)} examples")
        return examples

    def prepare_prompt(self, example):
        return (
            "The following is a graduate-level science question. "
            "Answer with only the letter (A, B, C, or D). "
            "Do not include explanations in your final answer.\n\n"
            f"{example.question}\n"
            f"A) {example.A}\nB) {example.B}\nC) {example.C}\nD) {example.D}\n\n"
        )

    def evaluate(self, predictions, references, eval_types=None, points=None):
        correct = 0
        scores = []
        for pred, ref in zip(predictions, references):
            extracted = extract_choice(pred).upper()
            match = int(extracted == ref.strip().upper())
            scores.append(match)
            correct += match
        total = len(predictions)
        return {
            'accuracy': (correct / total * 100) if total > 0 else 0,
            'correct': correct,
            'total': total,
            'per_example_accuracy': scores,
        }
