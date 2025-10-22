# ngram_lm.py
# N-Gram Language Modeling and Evaluation

import math, random
from collections import Counter
from tqdm import tqdm
import nltk
nltk.download('punkt')

# ---------- Load Data ----------
def load_data(path):
    with open(path, 'r', encoding='utf8') as f:
        lines = f.read().strip().split('\n')
    return lines

train_sents = load_data('ngram_language_model/ptbdataset/ptb.train.txt')
valid_sents = load_data('ngram_language_model/ptbdataset/ptb.valid.txt')
test_sents  = load_data('ngram_language_model/ptbdataset/ptb.test.txt')

# ---------- Tokenization ----------
def tokenize(sentences):
    tokenized = []
    for s in sentences:
        tokens = nltk.word_tokenize(s.lower())
        tokenized.append(["<s>"] + tokens + ["</s>"])
    return tokenized

train_tokens = tokenize(train_sents)
valid_tokens = tokenize(valid_sents)
test_tokens  = tokenize(test_sents)

# ---------- Handle rare words as <unk> ----------
from collections import Counter

def replace_rare_words(corpus, min_freq=2):
    word_counts = Counter([w for sent in corpus for w in sent])
    new_corpus = []
    for sent in corpus:
        new_sent = [w if word_counts[w] >= min_freq else '<unk>' for w in sent]
        new_corpus.append(new_sent)
    return new_corpus

train_tokens = replace_rare_words(train_tokens)
valid_tokens = replace_rare_words(valid_tokens)
test_tokens  = replace_rare_words(test_tokens)

# ---------- Base NGram LM ----------
class NGramLM:
    def __init__(self, n):
        self.n = n
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocab = set()

    def train(self, corpus):
        for sent in corpus:
            for i in range(len(sent) - self.n + 1):
                ngram = tuple(sent[i:i+self.n])
                context = ngram[:-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1
                self.vocab.update(sent)

    def prob(self, ngram):
        context = ngram[:-1]
        num = self.ngram_counts[ngram]
        den = self.context_counts[context]
        if den == 0 or num == 0:
            return 0.0
        return num / den

    def perplexity(self, corpus):
        N, log_prob = 0, 0
        for sent in corpus:
            for i in range(len(sent) - self.n + 1):
                ngram = tuple(sent[i:i+self.n])
                p = self.prob(ngram)
                if p == 0:
                    return float('inf')
                log_prob += math.log(p)
                N += 1
        return math.exp(-log_prob / N)

# ---------- Laplace (Add-1) Smoothing ----------
class LaplaceLM(NGramLM):
    def prob(self, ngram):
        context = ngram[:-1]
        V = len(self.vocab)
        num = self.ngram_counts[ngram] + 1
        den = self.context_counts[context] + V
        return num / den

# ---------- Linear Interpolation ----------
class InterpolatedLM:
    def __init__(self, lambdas):
        self.l1, self.l2, self.l3 = lambdas
        self.models = {1: NGramLM(1), 2: NGramLM(2), 3: NGramLM(3)}

    def train(self, corpus):
        for m in self.models.values():
            m.train(corpus)

    def prob(self, ngram):
        w1, w2, w3 = ngram[-3:]
        p1 = self.models[1].prob((w3,))
        p2 = self.models[2].prob((w2, w3))
        p3 = self.models[3].prob((w1, w2, w3))
        return self.l1*p1 + self.l2*p2 + self.l3*p3

    def perplexity(self, corpus):
        N, logp = 0, 0
        for sent in corpus:
            for i in range(2, len(sent)):
                ngram = tuple(sent[i-2:i+1])
                p = self.prob(ngram)
                logp += math.log(p if p > 0 else 1e-12)
                N += 1
        return math.exp(-logp / N)

# ---------- Stupid Backoff ----------
class StupidBackoffLM(NGramLM):
    def __init__(self, n, alpha=0.4):
        super().__init__(n)
        self.alpha = alpha
        self.lower = None
        if n > 1:
            self.lower = StupidBackoffLM(n-1, alpha)

    def train(self, corpus):
        super().train(corpus)
        if self.lower:
            self.lower.train(corpus)

    def prob(self, ngram):
        if self.ngram_counts[ngram] > 0:
            return self.ngram_counts[ngram]/self.context_counts[ngram[:-1]]
        elif self.lower:
            return self.alpha * self.lower.prob(ngram[1:])
        else:
            return 1e-12

# ---------- Generate Sentence ----------
def generate_sentence(model, max_len=15):
    sent = ["<s>", "<s>"]
    for _ in range(max_len):
        candidates = list(model.vocab)
        probs = [model.prob(tuple(sent[-2:] + [w])) for w in candidates]
        w = random.choices(candidates, weights=probs)[0]
        if w == "</s>":
            break
        sent.append(w)
    return ' '.join(sent[2:])

# ---------- Run all models ----------
if __name__ == "__main__":
    print("Training and evaluating models...")

    # MLE Models
    for n in [1, 2, 3, 4]:
        model = NGramLM(n)
        model.train(train_tokens)
        pp = model.perplexity(test_tokens)
        print(f"{n}-gram MLE Perplexity: {pp}")

    # Laplace Smoothing
    laplace = LaplaceLM(3)
    laplace.train(train_tokens)
    print("Laplace Perplexity:", laplace.perplexity(test_tokens))

    # Interpolation
    combos = [(0.2, 0.3, 0.5), (0.1, 0.2, 0.7), (0.3, 0.3, 0.4)]
    for l in combos:
        lm = InterpolatedLM(l)
        lm.train(train_tokens)
        pp = lm.perplexity(valid_tokens)
        print(f"Interpolation λ={l} Validation PP={pp}")

    # Stupid Backoff
    sbo = StupidBackoffLM(3, alpha=0.4)
    sbo.train(train_tokens)
    print("Stupid Backoff Perplexity:", sbo.perplexity(test_tokens))

    # Text Generation
    print("\nGenerated Sentences:")
    for _ in range(5):
        print(generate_sentence(sbo))
