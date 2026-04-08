import matplotlib
matplotlib.use("Agg")

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import joblib
import pickle
import spacy
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import xgboost as xgb
from sae_lens import SAE

class TimingCallback(xgb.callback.TrainingCallback):
    def __init__(self, total_rounds=None, print_every=25):
        self.total_rounds = total_rounds
        self.print_every = print_every

    def before_training(self, model):
        return model

    def after_iteration(self, model, epoch, evals_log):
        return False

    def after_training(self, model):
        return model

# Initialize models (DependencyAI & DivEye)
class DependencyAIDetector:
    def __init__(self, vectorizer_path, model_path):
        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)
        self.nlp = spacy.load("ru_core_news_lg")

    def extract_dependency_sequence(self, text):
        doc = self.nlp(text)
        dep_seq = " ".join([token.dep_ for token in doc])
        return dep_seq

    def predict_proba(self, text):
        # Извлечение зависимостей
        dep_seq = self.extract_dependency_sequence(text)
        
        # Векторизация текста
        transformed_text = self.vectorizer.transform([dep_seq])
        
        # Получаем вероятность, что текст сгенерирован ИИ
        return self.model.predict_proba(transformed_text)[0, 1]

class RussianAIDetector:
    def __init__(self, model_path="./local_model", xgb_path="diveye_llmtrace_ru_booster.pkl"):
        self.device = "cpu"
        self.max_length = 1024

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()

        if xgb_path.endswith(".joblib"):
            artifact = joblib.load(xgb_path)
        else:
            with open(xgb_path, "rb") as f:
                artifact = pickle.load(f)

        if hasattr(artifact, "predict_proba"):
            self.clf = artifact
            self.calibrator = None
            self.threshold = 0.5
            self.feature_columns = None
            self.use_booster = False

        elif isinstance(artifact, dict):
            self.clf = artifact["model"]
            self.calibrator = artifact.get("calibrator")
            self.threshold = float(artifact.get("threshold", 0.5))
            self.feature_columns = artifact.get("feature_columns")
            self.use_booster = bool(artifact.get("use_booster", False))
        else:
            raise ValueError("Неизвестный формат файла детектора")

        self.feature_dim = len(self.feature_columns) if self.feature_columns else 9
        print("Модель и токенизатор загружены из локальной папки:", model_path)
        print("DivEye threshold =", self.threshold)
        print("DivEye feature_dim =", self.feature_dim)

    @torch.no_grad()
    def _compute_surprisal(self, text, max_length=None):
        max_length = max_length or self.max_length
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self.device)

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        surprisal = -token_log_probs.detach().cpu().numpy()[0]
        return surprisal.astype(np.float32)

    def _safe_quantile(self, arr, q):
        if arr.size == 0:
            return 0.0
        return float(np.quantile(arr, q))

    def _safe_mean(self, arr):
        return float(np.mean(arr)) if arr.size else 0.0

    def _safe_std(self, arr):
        return float(np.std(arr)) if arr.size else 0.0

    def _safe_var(self, arr):
        return float(np.var(arr)) if arr.size else 0.0

    def _safe_max(self, arr):
        return float(np.max(arr)) if arr.size else 0.0

    def _text_stabilizer_features(self, text, surprisal):
        token_count = int(len(surprisal))

        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        sent_lengths = [len(s.split()) for s in sentences if s.strip()]

        mean_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
        std_sent_len = float(np.std(sent_lengths)) if sent_lengths else 0.0

        punct_ratio = 0.0
        if len(text) > 0:
            punct_ratio = len(re.findall(r"[,:;.!?\-()\"]", text)) / max(len(text), 1)

        words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if words:
            ttr = len(set(words)) / len(words)
        else:
            ttr = 0.0

        return {
            "token_count": float(token_count),
            "mean_sent_len": float(mean_sent_len),
            "std_sent_len": float(std_sent_len),
            "punct_ratio": float(punct_ratio),
            "ttr": float(ttr),
        }

    def _extract_features_dict(self, text, surprisal):
        S = np.asarray(surprisal, dtype=np.float32)
        if S.size == 0:
            base = {
                "s_mean": 0.0, "s_std": 0.0, "s_q90": 0.0,
                "d1_mean_abs": 0.0, "d1_std": 0.0, "d1_q90_abs": 0.0,
                "d2_mean_abs": 0.0, "d2_std": 0.0, "d2_q90_abs": 0.0,
            }
            base.update(self._text_stabilizer_features(text, S))
            return base

        d1 = np.diff(S)
        d2 = np.diff(d1)

        feats = {
            "s_mean": self._safe_mean(S),
            "s_std": self._safe_std(S),
            "s_q90": self._safe_quantile(S, 0.90),

            "d1_mean_abs": self._safe_mean(np.abs(d1)),
            "d1_std": self._safe_std(d1),
            "d1_q90_abs": self._safe_quantile(np.abs(d1), 0.90),

            "d2_mean_abs": self._safe_mean(np.abs(d2)),
            "d2_std": self._safe_std(d2),
            "d2_q90_abs": self._safe_quantile(np.abs(d2), 0.90),
        }

        feats.update(self._text_stabilizer_features(text, S))
        return feats

    def _extract_features(self, surprisal, text=None):
        text = text or ""
        feats_dict = self._extract_features_dict(text, surprisal)

        if self.feature_columns:
            return np.array(
                [float(feats_dict.get(col, 0.0)) for col in self.feature_columns],
                dtype=np.float32
            )

        S = np.asarray(surprisal, dtype=np.float32)
        if S.size == 0:
            return np.zeros(9, dtype=np.float32)

        dS = np.diff(S)
        d2S = np.diff(dS)

        def safe_stats(arr):
            if arr.size == 0:
                return [0.0, 0.0, 0.0]
            return [float(np.mean(arr)), float(np.var(arr)), float(np.max(arr))]

        feats = safe_stats(S) + safe_stats(dS) + safe_stats(d2S)
        return np.array(feats, dtype=np.float32)

    def _predict_raw_proba(self, X):
        proba = float(self.clf.predict_proba(X)[0, 1])

        if self.calibrator is not None:
            try:
                proba = float(self.calibrator.predict(np.array([proba]))[0])
            except Exception:
                pass

        return proba

    def predict_proba(self, text):
        try:
            if not text or len(text.strip()) == 0:
                return 0.5, "Не определено", 0.5

            surprisal_seq = self._compute_surprisal(text)
            features = self._extract_features(surprisal_seq, text=text)
            proba_ai = self._predict_raw_proba(features.reshape(1, -1))

            label = "ИИ-ГЕНЕРИРОВАННЫЙ" if proba_ai >= self.threshold else "Человеческий"

            if proba_ai >= self.threshold:
                confidence = proba_ai
            else:
                confidence = 1.0 - proba_ai

            return proba_ai, label, confidence

        except Exception as e:
            print(f"Ошибка при детекции ИИ: {e}")
            return 0.5, "Не определено", 0.5


class SAEGemmaXGBDetector:
    def __init__(self, config_path="run_config.json", xgb_path="xgb_layer_16.joblib", model_path="gemma-2-2b", hf_token=None):
        self.device = "cpu"
        self.available = False
        self.load_error = None
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)

            self.model_name = cfg["MODEL_NAME"]
            self.sae_release = cfg["SAE_RELEASE"]
            self.sae_id = cfg["SAE_ID"]
            self.max_length = int(cfg.get("MAX_LENGTH", 768))
            self.use_bos_in_sum = bool(cfg.get("USE_BOS_IN_SUM", True))
            self.layer = int(cfg["LAYER"])
            self.model_source = model_path or os.getenv("SAE_XGB_GEMMA_PATH") or self.model_name

            local_only = os.path.isdir(self.model_source)
            common_kwargs = {"local_files_only": local_only}
            if self.hf_token and not local_only:
                common_kwargs["token"] = self.hf_token

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_source, use_fast=True, **common_kwargs)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_source,
                torch_dtype=torch.float32,
                output_hidden_states=True,
                **common_kwargs,
            ).to(self.device)
            self.model.eval()

            self.sae = SAE.from_pretrained(
                release=self.sae_release,
                sae_id=self.sae_id,
                device=self.device,
            )
            self.sae.eval()

            self.clf = joblib.load(xgb_path)
            self.available = True
            print(f"SAE/Gemma/XGB detector loaded: layer={self.layer}, source={self.model_source}")
        except Exception as e:
            self.load_error = str(e)
            print(f"SAE/Gemma/XGB detector unavailable: {e}")

    def _tokenize(self, texts):
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.model.device) for k, v in enc.items()}

    @torch.no_grad()
    def _extract_features(self, texts):
        batch = self._tokenize(texts)
        outputs = self.model(**batch, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        resid = hidden_states[self.layer + 1]
        sae_latents = self.sae.encode(resid)

        attn = batch["attention_mask"].unsqueeze(-1).to(sae_latents.dtype)
        sae_latents = sae_latents * attn

        if not self.use_bos_in_sum and sae_latents.shape[1] > 0:
            sae_latents[:, 0, :] = 0

        pooled = sae_latents.sum(dim=1)
        return pooled.detach().float().cpu().numpy()

    def predict_proba(self, text):
        if not self.available:
            return None
        try:
            if not text or not text.strip():
                return 0.5, "Не определено", 0.5

            X = self._extract_features([text])
            proba_ai = float(self.clf.predict_proba(X)[0, 1])
            label = "ИИ-ГЕНЕРИРОВАННЫЙ" if proba_ai > 0.5 else "Человеческий"
            confidence = proba_ai if proba_ai > 0.5 else (1 - proba_ai)
            return proba_ai, label, confidence
        except Exception as e:
            print(f"Ошибка при SAE/Gemma/XGB детекции: {e}")
            return None

from dataclasses import dataclass


@dataclass
class EnsembleVerdict:
    final_ai_prob: float
    final_human_prob: float
    final_label: str
    confidence_pct: float
    diveye_weight: float
    dependency_weight: float
    rationale: list[str]


def ensemble_ai_verdict(
    dependency_ai_prob: float,
    diveye_ai_prob: float,
    text_length_tokens: int,
    has_repetition: bool = False,
    has_anomalous_tail: bool = False,
    surprisal_is_smooth: bool = False,
    syntactic_is_too_regular: bool = False,
    need_second_opinion: bool = True,
) -> EnsembleVerdict:
    dependency_ai_prob = max(0.0, min(1.0, dependency_ai_prob))
    diveye_ai_prob = max(0.0, min(1.0, diveye_ai_prob))

    diveye_weight = 0.70
    dependency_weight = 0.30
    rationale: list[str] = []

    dependency_says_human = dependency_ai_prob < 0.5
    dependency_human_conf = 1.0 - dependency_ai_prob

    if has_repetition:
        diveye_weight += 0.05
        dependency_weight -= 0.05
        rationale.append("Обнаружены повторы или однотипные фрагменты: увеличен вес DivEye.")

    if has_anomalous_tail:
        diveye_weight += 0.05
        dependency_weight -= 0.05
        rationale.append("На графике surprisal обнаружен аномальный хвост: увеличен вес DivEye.")

    if surprisal_is_smooth:
        diveye_weight += 0.03
        dependency_weight -= 0.03
        rationale.append("График surprisal выглядит сглаженным и повторяемым: увеличен вес DivEye.")

    if text_length_tokens >= 250:
        diveye_weight += 0.03
        dependency_weight -= 0.03
        rationale.append("Текст достаточно длинный для устойчивого ритмического анализа DivEye.")

    if dependency_says_human and dependency_human_conf < 0.75:
        diveye_weight += 0.02
        dependency_weight -= 0.02
        rationale.append("DependencyAI склоняется к человеку без высокой уверенности: вес DivEye увеличен.")

    diveye_weight = min(diveye_weight, 0.85)
    dependency_weight = max(1.0 - diveye_weight, 0.15)

    if text_length_tokens < 120:
        dependency_weight += 0.10
        diveye_weight -= 0.10
        rationale.append("Текст короткий: DivEye может быть менее стабилен, поэтому увеличен вес DependencyAI.")

    if syntactic_is_too_regular:
        dependency_weight += 0.08
        diveye_weight -= 0.08
        rationale.append("Синтаксис выглядит слишком регулярным: увеличен вес DependencyAI.")

    if need_second_opinion:
        rationale.append("DependencyAI сохранён как второе мнение другого типа, не завязанное на surprisal.")

    total_w = diveye_weight + dependency_weight
    diveye_weight /= total_w
    dependency_weight /= total_w

    final_ai_prob = dependency_weight * dependency_ai_prob + diveye_weight * diveye_ai_prob
    final_human_prob = 1.0 - final_ai_prob

    if final_ai_prob >= 0.5:
        final_label = "ИИ"
        confidence_pct = final_ai_prob * 100.0
    else:
        final_label = "Человек"
        confidence_pct = final_human_prob * 100.0

    disagreement = abs(dependency_ai_prob - diveye_ai_prob)
    if disagreement >= 0.40:
        rationale.append("Методы существенно расходятся в оценке; итоговый вердикт следует интерпретировать осторожно.")

    return EnsembleVerdict(
        final_ai_prob=final_ai_prob,
        final_human_prob=final_human_prob,
        final_label=final_label,
        confidence_pct=confidence_pct,
        diveye_weight=diveye_weight,
        dependency_weight=dependency_weight,
        rationale=rationale,
    )


def _detect_repetition(text: str) -> bool:
    paragraphs = [p.strip().lower() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) >= 2:
        unique_ratio = len(set(paragraphs)) / len(paragraphs)
        if unique_ratio < 0.8:
            return True

    sentences = [s.strip().lower() for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(sentences) >= 6:
        unique_ratio = len(set(sentences)) / len(sentences)
        if unique_ratio < 0.85:
            return True

    return False


def _analyze_surprisal_profile(detector, text: str) -> dict:
    surprisal_seq = detector._compute_surprisal(text)
    s = np.asarray(surprisal_seq, dtype=np.float32)

    if s.size == 0:
        return {
            "has_anomalous_tail": False,
            "surprisal_is_smooth": False,
            "text_length_tokens": 0,
        }

    thirds = np.array_split(s, 3)
    first_mean = float(np.mean(thirds[0])) if len(thirds[0]) else 0.0
    last_mean = float(np.mean(thirds[-1])) if len(thirds[-1]) else 0.0

    tail = thirds[-1] if len(thirds[-1]) else s
    near_zero_tail_ratio = float(np.mean(tail < 0.5)) if tail.size else 0.0
    has_anomalous_tail = (first_mean > 0 and last_mean < first_mean * 0.45) or (near_zero_tail_ratio > 0.55)

    std_s = float(np.std(s))
    d1 = np.diff(s)
    smooth_change = float(np.std(d1)) if d1.size else 0.0
    surprisal_is_smooth = std_s < 1.2 or smooth_change < 0.9

    return {
        "has_anomalous_tail": has_anomalous_tail,
        "surprisal_is_smooth": surprisal_is_smooth,
        "text_length_tokens": int(len(s)),
    }


def _syntactic_is_too_regular(detector, text: str) -> bool:
    dep_seq = detector.extract_dependency_sequence(text)
    tags = dep_seq.split()
    if not tags:
        return False

    from collections import Counter
    counts = Counter(tags)
    most_common_ratio = counts.most_common(1)[0][1] / len(tags)
    unique_ratio = len(counts) / len(tags)

    return most_common_ratio > 0.22 or unique_ratio < 0.12

# Combined model class
class CombinedAIDetector:
    def __init__(self, dependency_model, diveye_model, sae_xgb_model=None):
        self.dependency_model = dependency_model
        self.diveye_model = diveye_model
        self.sae_xgb_model = sae_xgb_model

    def predict(self, text):
        prob_dependency = self.dependency_model.predict_proba(text)

        diveye_ai_prob, label_diveye, conf_diveye = self.diveye_model.predict_proba(text)

        repetition_flag = _detect_repetition(text)
        surprisal_meta = _analyze_surprisal_profile(self.diveye_model, text)
        syntactic_regular = _syntactic_is_too_regular(self.dependency_model, text)
        
        sae_result = self.sae_xgb_model.predict_proba(text) if self.sae_xgb_model else None
        sae_available = sae_result is not None
        sae_ai_prob = float(sae_result[0]) if sae_available else None

        ensemble = ensemble_ai_verdict(
        dependency_ai_prob=prob_dependency,
        diveye_ai_prob=diveye_ai_prob,
            text_length_tokens=surprisal_meta["text_length_tokens"],
            has_repetition=repetition_flag,
            has_anomalous_tail=surprisal_meta["has_anomalous_tail"],
            surprisal_is_smooth=surprisal_meta["surprisal_is_smooth"],
            syntactic_is_too_regular=syntactic_regular,
            need_second_opinion=True,
        )

        base_ai_prob = float(ensemble.final_ai_prob)
        rationale = list(ensemble.rationale)

        if sae_available:
            final_ai_prob = 0.6 * sae_ai_prob + 0.4 * base_ai_prob
            rationale.insert(0, "В итог также добавлен SAE/Gemma/XGB детектор, обученный на признаках SAE латентов Gemma.")
            rationale.append("Итоговая вероятность = 60% SAE/Gemma/XGB + 40% ансамбль DependencyAI/DivEye.")
            sae_weight = 0.60
            legacy_weight = 0.40
        else:
            final_ai_prob = base_ai_prob
            sae_weight = 0.0
            legacy_weight = 1.0
            rationale.append("SAE/Gemma/XGB детектор недоступен, поэтому итог построен только на DependencyAI и DivEye.")

        final_human_prob = 1.0 - final_ai_prob
        if final_ai_prob >= 0.5:
            final_label = "ИИ"
            confidence_pct = final_ai_prob * 100.0
        else:
            final_label = "Человек"
            confidence_pct = final_human_prob * 100.0

        return {
            "probability_dependencyAI": prob_dependency,
            "probability_divEye": diveye_ai_prob,
            "probability_sae_xgb": sae_ai_prob,

            "legacy_average_probability": base_ai_prob,
            "average_probability": final_ai_prob,

            "final_prediction": final_label,
            "final_confidence": confidence_pct,

            "diveye_weight": ensemble.diveye_weight,
            "dependency_weight": ensemble.dependency_weight,

            "sae_xgb_available": sae_available,
            "sae_xgb_error": getattr(self.sae_xgb_model, 'load_error', None) if self.sae_xgb_model else None,
            "sae_xgb_weight": sae_weight,
            "legacy_ensemble_weight": legacy_weight,

            "ensemble_rationale": rationale,
}

# Функция для расширенного анализа DependencyAI
def extended_analysis(file_path):
    # Проверка на существование файла
    if not os.path.exists(file_path):
        return "Файл не найден."

    # Загрузка моделей
    try:
        vectorizer = joblib.load('dependency_vectorizer.pkl')
        clf = joblib.load('dependency_model.pkl')
    except FileNotFoundError:
        return "Ошибка: Модели не найдены."

    # Чтение содержимого файла
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Создаем экземпляр класса DependencyAIDetector
    dependency_model = DependencyAIDetector('dependency_vectorizer.pkl', 'dependency_model.pkl')
    
    # Используем метод extract_dependency_sequence
    dep_seq = dependency_model.extract_dependency_sequence(content)

    # Преобразуем вектора текста в матрицу TF-IDF
    tfidf_matrix = vectorizer.transform([dep_seq])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # Извлечение признаков
    tfidf_weights = tfidf_df.iloc[0].values
    global_importances = clf.feature_importances_
    local_contributions = tfidf_weights * global_importances

    # Создаем DataFrame с результатами
    results_df = pd.DataFrame({
        'pattern': vectorizer.get_feature_names_out(),
        'contribution': local_contributions
    }).sort_values(by='contribution', ascending=False)

    # Оставляем только топ-10 по вкладу
    top_results = results_df[results_df['contribution'] > 0].head(10).copy()
    
    # Нормализация вкладов в проценты (среди топ-10)
    total_contribution = top_results['contribution'].sum() if not top_results.empty else 0.0
    top_results['share_pct'] = top_results['contribution'] / total_contribution * 100 if total_contribution > 0 else 0.0

    # Стиль для "человеческих" характеристик
    human_style_map = {
        'punct': 'Стандартная пунктуация (запятые/точки)',
        'cc conj': 'Однотипные перечисления или союзы (И/А/НО)',
        'obl': 'Избыток уточнений (где/когда/почему)',
        'advmod': 'Частое использование обстоятельств (как/каким образом)',
        'root': 'Предсказуемая структура главного действия (глагола)',
        'nmod': 'Нанизывание существительных (канцелярит)',
        'nsubj': 'Типичное подлежащее (кто/что)',
        'amod': 'Обилие описательных прилагательных',
        'conj': 'Сочинительная связь',
        'dep': 'Зависимый элемент (связка)',
        'obj': 'Прямое дополнение',
        'case': 'Предложная конструкция',
        'flat:foreign': 'Иностранные заимствования/слова',
        'appos': 'Поясняющее приложение'
    }

    # Перевод паттернов в читабельный формат
    def translate_pattern(p):
        if p in human_style_map:
            return human_style_map[p]
        tags = p.split()
        return " + ".join([human_style_map.get(t, t) for t in tags])

    top_results['readable_style'] = top_results['pattern'].apply(translate_pattern)

    # Получаем предсказание вероятности для ИИ
    prob = float(clf.predict_proba(tfidf_df)[0][1])  # P(AI) для DependencyAI
    is_ai = prob > 0.5
    confidence = prob if is_ai else (1 - prob)  # уверенность в выбранном вердикте
    verdict_text = "Текст сгенерирован с помощью ИИ" if is_ai else "Текст написан человеком"

    description_text = """
    Метод DependencyAI работает так: он не просто смотрит на слова в тексте, 
    а строит «скелет» каждого предложения (что с чем связано). 
    Подробнее про метод можно почитать здесь: 
    <a href="https://arxiv.org/pdf/2602.15514" target="_blank" rel="noopener">ссылка на статью</a>.
    """

    # Создаем график
    plt.figure(figsize=(18, 10))
    sns.barplot(
        x="share_pct",
        y="readable_style",
        data=top_results.sort_values("share_pct", ascending=True),
        palette="magma",
        hue="readable_style",
        legend=False
    )
    plt.title("Почему модель решила именно так", fontsize=24)
    plt.xlabel("Доля вклада в решение, %", fontsize=18)
    plt.ylabel("Стилистический паттерн (скелет фразы)", fontsize=18)
    plt.grid(axis="x", linestyle="--", alpha=0.35)


    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    img_path = "static/extended_analysis_image.png"
    plt.tight_layout()
    plt.subplots_adjust(left=0.38)
    plt.savefig(img_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Преобразуем top_results в список словарей
    top_results_list = top_results[['readable_style', 'share_pct']].to_dict(orient='records')

    # Возвращаем данные для шаблона
    return {
        "verdict_text": verdict_text,
        "confidence_pct": confidence * 100.0,
        "prob_ai_pct": prob * 100.0,
        "top_results_list": top_results_list,
        "img_path": img_path,
        "description_text": description_text
    }

def extended_analysis_diveye(text, detector, diveye_ai_prob):
    if not text or not text.strip():
        return {
            "verdict_text": "Недостаточно текста для анализа",
            "confidence_pct": 0.0,
            "description_text": "",
            "img_path": None,
            "group_cards": [],
            "top_signals": [],
            "summary_text": ""
        }

    surprisal_seq = detector._compute_surprisal(text)
    features = detector._extract_features(surprisal_seq, text=text)

    score = float(diveye_ai_prob)
    is_ai = score >= getattr(detector, "threshold", 0.5)
    confidence_pct = (score if is_ai else (1.0 - score)) * 100.0
    verdict_text = "Текст сгенерирован с помощью ИИ" if is_ai else "Текст написан человеком"

    description_text = """
    Метод DivEye анализирует не сами слова, а то, как по ходу текста меняется неожиданность токенов для языковой модели.
    В этой версии также учитываются дополнительные стабилизирующие признаки текста и применяется калибровка вероятности.
    Подробнее про метод можно почитать здесь:
    <a href="https://arxiv.org/pdf/2509.18880" target="_blank" rel="noopener">ссылка на статью</a>.
    """

    if hasattr(detector.clf, "feature_importances_"):
        importances = np.asarray(detector.clf.feature_importances_, dtype=np.float32)
    else:
        importances = np.ones(len(features), dtype=np.float32)

    local_scores = np.abs(features) * importances
    total_local = float(local_scores.sum()) if local_scores.size else 0.0

    if detector.feature_columns:
        feature_names = detector.feature_columns
    else:
        feature_names = [
            "s_mean", "s_var", "s_max",
            "d1_mean", "d1_var", "d1_max",
            "d2_mean", "d2_var", "d2_max",
        ]

    readable_map = {
        "s_mean": "Средняя неожиданность текста",
        "s_std": "Разброс неожиданности текста",
        "s_q90": "Высокие пики неожиданности",
        "d1_mean_abs": "Средняя сила локальных изменений",
        "d1_std": "Разброс локальных изменений",
        "d1_q90_abs": "Сильные локальные скачки",
        "d2_mean_abs": "Средняя глубинная неровность",
        "d2_std": "Разброс глубинной неровности",
        "d2_q90_abs": "Сильные вторичные колебания",
        "token_count": "Длина текста в токенах",
        "mean_sent_len": "Средняя длина предложений",
        "std_sent_len": "Разброс длины предложений",
        "punct_ratio": "Доля пунктуации",
        "ttr": "Лексическое разнообразие",
        "base_score": "Оценка базового детектора",
    }

    explanation_map = {
        "s_mean": "Показывает, насколько текст в целом предсказуем для языковой модели.",
        "s_std": "Показывает, насколько текст ровный или неоднородный по уровню неожиданности.",
        "s_q90": "Отражает выраженные неожиданные участки текста.",
        "d1_mean_abs": "Характеризует среднюю силу локальных изменений ритма.",
        "d1_std": "Показывает, насколько неравномерно меняется ритм текста.",
        "d1_q90_abs": "Выделяет сильные локальные переломы ритма.",
        "d2_mean_abs": "Характеризует среднюю глубинную изменчивость ритма.",
        "d2_std": "Показывает, насколько неоднородна глубинная динамика текста.",
        "d2_q90_abs": "Фиксирует самые сильные вторичные колебания ритма.",
        "token_count": "Длинные и короткие тексты ведут себя по-разному; длина помогает стабилизировать решение.",
        "mean_sent_len": "Средняя длина предложений помогает учитывать общий стиль текста.",
        "std_sent_len": "Разброс длины предложений показывает естественность структуры.",
        "punct_ratio": "Доля пунктуации помогает учитывать форму изложения.",
        "ttr": "Лексическое разнообразие помогает учитывать богатство словаря.",
        "base_score": "Сигнал вашего базового детектора, усиленный DivEye.",
    }

    feature_items = []
    for name, raw in zip(feature_names, local_scores):
        share_pct = (float(raw) / total_local * 100.0) if total_local > 0 else 0.0
        feature_items.append({
            "name": readable_map.get(name, name),
            "share_pct": share_pct,
            "explanation": explanation_map.get(name, "Дополнительный признак, влияющий на итоговое решение.")
        })

    feature_items = sorted(feature_items, key=lambda x: x["share_pct"], reverse=True)
    top_signals = feature_items[:5]

    # Группировка
    group_scores = {
        "Ритм surprisal": 0.0,
        "Стабилизаторы текста": 0.0,
        "Базовый детектор": 0.0,
    }

    for name, raw in zip(feature_names, local_scores):
        if name.startswith("s_") or name.startswith("d1_") or name.startswith("d2_"):
            group_scores["Ритм surprisal"] += float(raw)
        elif name == "base_score":
            group_scores["Базовый детектор"] += float(raw)
        else:
            group_scores["Стабилизаторы текста"] += float(raw)

    total_groups = sum(group_scores.values()) or 1.0
    group_cards = [
        {
            "title": k,
            "share_pct": v / total_groups * 100.0,
            "explanation": {
                "Ритм surprisal": "Блок признаков, связанных с динамикой неожиданности текста.",
                "Стабилизаторы текста": "Дополнительные текстовые признаки, которые делают решение устойчивее.",
                "Базовый детектор": "Сигнал исходного детектора, усиленный DivEye.",
            }[k]
        }
        for k, v in group_scores.items()
        if v > 0
    ]

    if is_ai:
        summary_text = (
            "Улучшенный DivEye показал, что текст ближе к ИИ: вклад внесли и ритм surprisal, "
            "и дополнительные стабилизирующие признаки."
        )
    else:
        summary_text = (
            "Улучшенный DivEye показал, что текст ближе к человеческому: ритм surprisal и "
            "дополнительные признаки не дают сильного сигнала в пользу ИИ."
        )

    s = np.asarray(surprisal_seq, dtype=np.float32)
    x = np.arange(1, len(s) + 1)

    if len(s) >= 5:
        window = max(5, min(25, len(s) // 10))
        smooth = pd.Series(s).rolling(window=window, min_periods=1).mean().to_numpy()
    else:
        smooth = s.copy()

    plt.figure(figsize=(20, 9))
    plt.plot(x, s, linewidth=1.8, alpha=0.35, label="Surprisal по токенам")
    plt.plot(x, smooth, linewidth=3.5, label="Сглаженная кривая surprisal")
    plt.title("Ритм неожиданности текста по методу DivEye", fontsize=26, loc="left")
    plt.xlabel("Позиция токена в тексте", fontsize=20)
    plt.ylabel("Неожиданность токена (surprisal)", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend(fontsize=16)

    diveye_img_path = "static/diveye_analysis_image.png"
    plt.tight_layout()
    plt.savefig(diveye_img_path, dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "verdict_text": verdict_text,
        "confidence_pct": confidence_pct,
        "description_text": description_text,
        "img_path": diveye_img_path,
        "group_cards": group_cards,
        "top_signals": top_signals,
        "summary_text": summary_text
    }


def extended_analysis_sae_gemma(text, detector):
    try:
        if detector is None or not getattr(detector, "available", False):
            return {"error": "SAE/Gemma/XGB детектор недоступен."}

        if not text or not text.strip():
            return {"error": "Пустой текст."}

        import os
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import xgboost as xgb
        import torch

        # 1. Токенизация и проход через Gemma
        batch = detector._tokenize([text])

        with torch.no_grad():
            outputs = detector.model(**batch, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            resid = hidden_states[detector.layer + 1]   # [1, T, D]

            sae_latents = detector.sae.encode(resid)    # [1, T, F]

            attn = batch["attention_mask"].unsqueeze(-1).to(sae_latents.dtype)
            sae_latents = sae_latents * attn

            if not detector.use_bos_in_sum and sae_latents.shape[1] > 0:
                sae_latents[:, 0, :] = 0

            pooled = sae_latents.sum(dim=1)             # [1, F]
            X = pooled.detach().float().cpu().numpy()

        # 2. Предсказание и покомпонентные вклады XGBoost
        booster = detector.clf.get_booster()
        dmatrix = xgb.DMatrix(X)

        contribs = booster.predict(dmatrix, pred_contribs=True)  # [1, F+1]
        contribs = contribs[0]

        # последний элемент — bias
        feature_contribs = contribs[:-1]
        bias_term = float(contribs[-1])

        prob_ai = float(detector.clf.predict_proba(X)[0, 1])
        prediction = "ИИ-ГЕНЕРИРОВАННЫЙ" if prob_ai >= 0.5 else "Человеческий"

        # 3. Сырые pooled-активации по признакам
        pooled_vec = X[0]

        # 4. Топ-признаки по абсолютному вкладу
        top_abs_idx = np.argsort(np.abs(feature_contribs))[::-1][:10]

        top_df = pd.DataFrame({
            "feature_idx": top_abs_idx,
            "contribution": feature_contribs[top_abs_idx],
            "abs_contribution": np.abs(feature_contribs[top_abs_idx]),
            "activation": pooled_vec[top_abs_idx],
        }).sort_values("abs_contribution", ascending=True)

        # 5. Отдельно топ AI и топ human признаки
        pos_idx = np.where(feature_contribs > 0)[0]
        neg_idx = np.where(feature_contribs < 0)[0]

        top_ai_idx = pos_idx[np.argsort(feature_contribs[pos_idx])[::-1][:5]] if len(pos_idx) > 0 else np.array([], dtype=int)
        top_human_idx = neg_idx[np.argsort(feature_contribs[neg_idx])[:3]] if len(neg_idx) > 0 else np.array([], dtype=int)

        top_ai_features = []
        for idx in top_ai_idx:
            top_ai_features.append({
                "feature_idx": int(idx),
                "contribution": float(feature_contribs[idx]),
                "activation": float(pooled_vec[idx]),
            })

        top_human_features = []
        for idx in top_human_idx:
            top_human_features.append({
                "feature_idx": int(idx),
                "contribution": float(feature_contribs[idx]),
                "activation": float(pooled_vec[idx]),
            })

        # 6. График вкладов топ-10 признаков
        plt.figure(figsize=(12, 5))
        colors = ["#c0392b" if v > 0 else "#2980b9" for v in top_df["contribution"]]
        plt.barh(
            [f"SAE #{int(i)}" for i in top_df["feature_idx"]],
            top_df["contribution"],
            color=colors
        )
        plt.axvline(0, color="black", linewidth=1)
        plt.title("Топ-10 SAE-признаков по вкладу в решение")
        plt.xlabel("Вклад в решение XGBoost")
        plt.ylabel("SAE-признак")
        plt.tight_layout()

        barplot_path = "static/sae_analysis_barplot.png"
        plt.savefig(barplot_path, bbox_inches="tight")
        plt.close()

        # 7. График активации топ-1 признака по токенам
        if len(top_abs_idx) > 0:
            top_feature_idx = int(top_abs_idx[0])

            token_latents = sae_latents[0, :, top_feature_idx].detach().float().cpu().numpy()
            token_ids = batch["input_ids"][0].detach().cpu().tolist()
            tokens = detector.tokenizer.convert_ids_to_tokens(token_ids)

            # срежем паддинги
            attn_mask = batch["attention_mask"][0].detach().cpu().numpy().astype(bool)
            token_latents = token_latents[attn_mask]
            tokens = [tok for tok, keep in zip(tokens, attn_mask) if keep]

            # подписи токенов лучше прореживать, иначе график будет нечитаемым
            x = np.arange(len(token_latents))

            plt.figure(figsize=(13, 4.5))
            plt.plot(x, token_latents, linewidth=1.8)
            plt.title(f"Активация ключевого SAE-признака #{top_feature_idx} по токенам")
            plt.xlabel("Позиция токена в тексте")
            plt.ylabel("Активация признака")
            plt.tight_layout()

            activation_plot_path = "static/sae_top_feature_trace.png"
            plt.savefig(activation_plot_path, bbox_inches="tight")
            plt.close()

            # Для текста под графиком удобно отдать немного токенов
            token_preview = []
            for i, (tok, val) in enumerate(zip(tokens[:80], token_latents[:80])):
                token_preview.append({
                    "pos": i,
                    "token": tok,
                    "activation": float(val),
                })
        else:
            top_feature_idx = None
            activation_plot_path = None
            token_preview = []

        # 8. Сводные показатели
        total_pos = float(np.sum(feature_contribs[feature_contribs > 0])) if np.any(feature_contribs > 0) else 0.0
        total_neg = float(np.sum(np.abs(feature_contribs[feature_contribs < 0]))) if np.any(feature_contribs < 0) else 0.0
        total_abs = float(np.sum(np.abs(feature_contribs))) if np.sum(np.abs(feature_contribs)) > 0 else 1.0

        ai_signal_pct = round(100.0 * total_pos / total_abs, 1)
        human_signal_pct = round(100.0 * total_neg / total_abs, 1)
        dominance_pct = round(100.0 * np.sum(np.abs(feature_contribs[top_abs_idx[:3]])) / total_abs, 1)

        explanation = {
            "prediction": prediction,
            "ai_score": round(prob_ai, 4),
            "bias_term": round(bias_term, 4),
            "summary": {
                "ai_signal_pct": ai_signal_pct,
                "human_signal_pct": human_signal_pct,
                "dominance_pct": dominance_pct,
            },
            "top_ai_features": top_ai_features,
            "top_human_features": top_human_features,
            "top_feature_idx": top_feature_idx,
            "barplot_path": "/" + barplot_path,
            "activation_plot_path": ("/" + activation_plot_path) if activation_plot_path else None,
            "token_preview": token_preview,
        }
        return explanation

    except Exception as e:
        return {"error": f"Ошибка анализа SAE/Gemma: {str(e)}"}

# Flask App Setup
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', '')
    file = request.files.get('file')

    if file:
        text = file.read().decode('utf-8')

    if text.strip() == "":
        return render_template('index.html', error="Пожалуйста, введите текст или загрузите файл.")

    with open("temp_text.txt", "w", encoding="utf-8") as f:
        f.write(text)

    result = combined_detector.predict(text)
    return render_template('index.html', result=result, text_value=text)

@app.route('/extended_analysis_page', methods=['POST'])
def extended_analysis_page():
    if not os.path.exists("temp_text.txt"):
        return render_template(
            'extended_analysis_page.html',
            dep=None,
            diveye=None,
            sae=None,
            error="Не удалось выполнить расширенный анализ: исходный текст не найден."
        )

    with open("temp_text.txt", "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return render_template(
            'extended_analysis_page.html',
            dep=None,
            diveye=None,
            sae=None,
            error="Не удалось выполнить расширенный анализ: текст пуст."
        )

    diveye_ai_prob = float(request.form.get("diveye_ai_prob", 0.5))

    dep = extended_analysis('temp_text.txt')
    diveye = extended_analysis_diveye(text, diveye_model, diveye_ai_prob)
    sae = extended_analysis_sae_gemma(text, sae_xgb_model)

    return render_template(
        'extended_analysis_page.html',
        dep=dep,
        diveye=diveye,
        sae=sae,
        error=None
    )

if __name__ == '__main__':
    # Initialize the models
    dependency_model = DependencyAIDetector('dependency_vectorizer.pkl', 'dependency_model.pkl')
    diveye_model = RussianAIDetector(xgb_path="diveye_llmtrace_ru_xgb.pkl")
    sae_xgb_model = SAEGemmaXGBDetector(
        config_path='run_config.json',
        xgb_path='xgb_layer_16.joblib',
        model_path='./gemma-2-2b',
        hf_token=None,
    )
    combined_detector = CombinedAIDetector(dependency_model, diveye_model, sae_xgb_model)
    

    app.run(debug=True, port=5001)
