from datasets import load_dataset
from dataclasses import dataclass
from abc import ABC, abstractmethod
import sacrebleu
from comet import download_model, load_from_checkpoint
from math_verify import parse, StringExtractionConfig, LatexExtractionConfig, verify


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
    def evaluate(self, predictions, references):
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
    
    def evaluate(self, predictions, references):
        """Evaluate MMMLU predictions"""
        correct = 0
        total = len(predictions)
        
        extracted_predictions = [extract_choice(pred) for pred in predictions]
        
        for pred, ref in zip(extracted_predictions, references):
            if pred == ref:
                correct += 1

        accuracy = (correct / total * 100) if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
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


def extract_boxed_answer(text):
    """Extract answer within the \\boxed{{}}"""
    return parse(
            text, 
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
    
    def evaluate(self, predictions, references):
        """Evaluate MMMLU predictions"""
        correct = 0
        total = len(predictions)
        
        extracted_predictions = [extract_boxed_answer(pred) for pred in predictions]
        for pred, ref in zip(extracted_predictions, references):
            if verify(ref, pred):
                correct += 1
        
        
        accuracy = (correct / total * 100) if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
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
    
    def evaluate(self, predictions, references):
        """Evaluate WMT24++ predictions using xCOMET, BLEU, and chrF++"""
        # Calculate BLEU
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        
        # Calculate chrF++
        chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
        
        # Calculate per-example BLEU and chrF++
        per_example_bleu = []
        per_example_chrf = []
        for pred, ref in zip(predictions, references):
            example_bleu = sacrebleu.sentence_bleu(pred, [ref])
            example_chrf = sacrebleu.sentence_chrf(pred, [ref], word_order=2)
            per_example_bleu.append(example_bleu.score)
            per_example_chrf.append(example_chrf.score)
        
        # Calculate xCOMET score
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

        return {
            "xcomet-xl": comet_mean,         
            "xcomet-xl_std": comet_std,
            "bleu": bleu.score,
            "chrfpp": chrf.score,
            "num_predictions": len(predictions),
            "per_example_scores": {
                "xcomet-xl": comet_scores,
                "bleu": per_example_bleu,
                "chrfpp": per_example_chrf,
            },
        }