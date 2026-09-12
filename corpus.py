"""
Corpus builder — lumpy-geometry real text corpus.

Embedding: char-trigrams + word-unigrams → 4096-dim BOW 
→ random projection to R^64 (single fixed seed for corpus + queries).

Geometry:
  Same-topic texts share keywords/ngrams → cosine 0.4–0.9.
  Cross-topic pairs → cosine near zero.
  This is "lumpy" in the way real text embeddings are:
  clusters of similar texts with gaps between topics.

The corpus is built to be low-duplicate: each headline is fully
filled from ~10k+ entity combinations, so NN similarity is genuinely
earned by semantic overlap, not text identity.
"""

import numpy as np
import re
from itertools import product
from typing import List, Tuple

DIM       = 64
VOCAB     = 4096
PROJ_SEED = 42

# ── text → vector ──────────────────────────────────────────────────

_STOPWORDS = set("""
a an the and or but in on at to for of with is are was were be been
has have had do does did will would could should may might shall can
its it this that these those i we you he she they us them our your
his her their from by as if not no so up out about into than then
when where who which how all any some one two also new s t re
""".split())

def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1 and t not in _STOPWORDS]

def char_ngrams(tok: str, n: int = 3) -> List[str]:
    p = f"#{tok}#"
    return [p[i:i+n] for i in range(len(p)-n+1)]

def text_to_bow(text: str, vocab_size: int = VOCAB) -> np.ndarray:
    tokens = tokenize(text)
    vec = np.zeros(vocab_size, dtype=np.float32)
    for tok in tokens:
        vec[hash(tok) % vocab_size] += 2.0
        for ng in char_ngrams(tok):
            vec[hash(ng) % vocab_size] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec

_PROJ: np.ndarray = None

def _proj(vocab_size: int = VOCAB, dim: int = DIM) -> np.ndarray:
    global _PROJ
    if _PROJ is None or _PROJ.shape != (vocab_size, dim):
        rng = np.random.default_rng(PROJ_SEED)
        P = rng.standard_normal((vocab_size, dim)).astype(np.float32)
        P /= (np.linalg.norm(P, axis=0, keepdims=True) + 1e-10)
        _PROJ = P
    return _PROJ

def bow_to_embedding(bow: np.ndarray, dim: int = DIM, seed: int = PROJ_SEED) -> np.ndarray:
    emb = bow @ _proj(bow.shape[0], dim)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb /= norm
    return emb

def embed_text(text: str) -> np.ndarray:
    return bow_to_embedding(text_to_bow(text))

# ── topic templates ────────────────────────────────────────────────

_T = {
    "world": [
        "{A} security council discusses {B} crisis",
        "{B} president announces economic policy reform",
        "Protests erupt in {C} over government election decision",
        "{A} and {B} sign peace agreement ending conflict",
        "Humanitarian aid reaches {C} after natural disaster",
        "{A} foreign minister resigns amid corruption scandal",
        "NATO allies discuss {B} border military tensions",
        "Elections in {B} draw international community attention",
        "World leaders gather in {C} for climate security summit",
        "{A} imposes trade sanctions on neighboring state",
        "Civil unrest in {B} displaces tens of thousands",
        "International court rules against {A} border dispute",
        "Ceasefire agreed in {B} after months of armed conflict",
        "Refugee crisis deepens as {A} closes border crossings",
        "{C} becomes flashpoint for regional ethnic tension",
        "Bilateral peace talks collapse between {A} and {B}",
        "{A} sends peacekeeping military troops to {B} region",
        "Diplomatic relations severed between {A} and {B}",
        "Mass protests in {C} demand government resignation now",
        "{B} prime minister survives no-confidence parliamentary vote",
        "United Nations mediators arrive in {C} for peace talks",
        "{B} military conducts joint exercises with {A} allies",
        "Foreign ministers of {A} meet with {B} counterpart",
        "Human rights violations reported in {B} conflict zone",
        "Emergency UN resolution passed on {B} humanitarian crisis",
    ],
    "sports": [
        "{D} defeats {E} in championship final match",
        "{F} breaks world record at {C} international tournament",
        "{D} manager sacked following consecutive losing streak",
        "Olympic committee bans {A} athletes from international games",
        "{F} signs record transfer deal with {D} football club",
        "{G} world cup draw announced for upcoming season",
        "{D} wins league title after dramatic late comeback",
        "Injury crisis hits {D} squad ahead of crucial derby match",
        "{F} announces retirement from professional {G} career",
        "Doping scandal rocks international {G} governing body",
        "{D} advances to semifinals in {H} tournament competition",
        "{C} awarded hosting rights for upcoming international games",
        "{F} named best {G} player of the year award",
        "Controversial referee red card decision sparks {G} debate",
        "{D} sets new home ground attendance record",
        "{F} comeback victory wins {H} title after two years",
        "{D} signs veteran goalkeeper ahead of new season",
        "Football federation fines {D} for crowd disorder incident",
        "{F} scores hat-trick in crucial {D} league victory",
        "{G} association announces major rule changes next season",
        "Transfer window: {D} completes signing of {F} striker",
        "{D} academy produces two new national team players",
        "Sports arbitration court overturns {F} doping ban",
        "Broadcasting rights for {H} sold for record billion deal",
        "{C} stadium expansion approved to increase seating capacity",
    ],
    "business": [
        "{I} reports record quarterly revenue earnings growth",
        "{I} announces merger acquisition with {J} corporation",
        "Stock markets fall on {A} inflation economic data",
        "{I} lays off thousands of workers amid corporate restructuring",
        "Central bank raises benchmark interest rates to combat inflation",
        "{I} launches new product line targeting Asian markets",
        "Trade deficit widens sharply as export growth slows",
        "{I} chief executive officer steps down after board pressure",
        "Startup raises {K} million dollars in Series funding round",
        "Oil crude prices surge on supply disruption fears",
        "{I} faces antitrust monopoly probe in European Union",
        "Retail consumer sales decline for third consecutive quarter",
        "{I} expands business operations into {A} emerging market",
        "Cryptocurrency digital asset regulation proposed by finance ministry",
        "Global supply chain disruptions hurt {I} quarterly profits",
        "{I} acquires competitor rival to dominate market segment",
        "Inflation falls to lowest level recorded in two years",
        "Bond treasury yields rise as investors anticipate rate hike",
        "{I} posts quarterly loss in challenging economic environment",
        "Global trade tensions hit manufacturing production output",
        "{I} announces share buyback program worth billion dollars",
        "Consumer confidence index rises for third month consecutively",
        "{J} sues {I} over patent intellectual property violation",
        "Private equity firm acquires {I} in leveraged buyout",
        "Earnings season: {I} beats analyst estimates revenue forecast",
    ],
    "technology": [
        "{I} unveils new artificial intelligence semiconductor chip",
        "Cybersecurity data breach exposes millions of user records",
        "{I} releases major software platform update with features",
        "Researchers achieve breakthrough discovery in quantum computing",
        "{I} acquires AI startup for {K} billion dollars deal",
        "Social media platform faces congressional antitrust scrutiny",
        "Electric vehicle battery energy range hits new milestone",
        "{I} self-driving autonomous car completes cross-country trip",
        "Space agency launches internet satellite constellation broadband",
        "Open source software project gains rapid enterprise adoption",
        "{I} data center powered entirely by renewable solar energy",
        "Robotics automation firm deploys warehouse robots at scale",
        "5G network rollout reaches rural areas in pilot program",
        "Tech giant faces {K} billion fine over data privacy violations",
        "New programming language gains developer community traction",
        "Neural network AI model surpasses human performance benchmark",
        "Deepfake detection security tools lag behind generation capabilities",
        "{I} announces workforce layoffs as digital advertising revenue falls",
        "Semiconductor chip shortage eases after record production quarter",
        "Cloud computing infrastructure spending grows despite economic slowdown",
        "{I} launches new smartphone with improved camera battery life",
        "Streaming platform {I} loses subscribers amid price increase",
        "Internet of things devices exposed to security vulnerability exploit",
        "{I} partners with {J} on next-generation computing platform",
        "Venture capital investment in AI startups reaches record level",
    ],
    "science": [
        "Scientists discover new species in {L} tropical rainforest",
        "Mars planetary rover detects signs of ancient water flow",
        "Climate change study warns of accelerating Arctic ice loss",
        "New experimental vaccine shows clinical trial promise results",
        "Astronomers observe most distant galaxy ever recorded detected",
        "Gene editing CRISPR technique corrects rare hereditary genetic disorder",
        "Ancient fossil found in {L} rewrites human evolution timeline",
        "Ocean sea temperatures reach record high this year",
        "Nuclear fusion reactor experiment achieves net energy gain milestone",
        "New antibiotic drug effective against drug-resistant bacteria strain",
        "Deep sea ocean expedition discovers hydrothermal vent ecosystem",
        "Particle physics experiment confirms long-sought theoretical prediction",
        "Brain-computer neural interface restores movement in paralyzed patient",
        "Permafrost thaw is releasing carbon faster than models projected",
        "Space telescope images exoplanet distant atmosphere for first time",
        "CRISPR gene therapy trial shows safety in human clinical study",
        "Methane emissions from wetlands underestimated by climate models significantly",
        "Superconductor material operates at record high ambient temperature",
        "Study links microplastics exposure to cardiovascular heart disease risk",
        "Dark matter detector experiment finds no signal after deep search",
        "Biodiversity loss accelerating according to new global scientific survey",
        "Gravitational wave detection confirms black hole merger event",
        "Arctic permafrost fires release record carbon emissions this summer",
        "Extinct animal species successfully cloned using preserved ancient DNA",
        "Solar coronal mass ejection disrupts satellite communications worldwide",
    ],
}

# Large entity pools for low-duplicate generation
_EA  = ["Germany","Brazil","Japan","India","France","Canada","Australia","Mexico",
        "South Korea","Nigeria","Egypt","Turkey","Argentina","Indonesia","Saudi Arabia",
        "Poland","Ukraine","Israel","Colombia","South Africa","Pakistan","Bangladesh",
        "Vietnam","Thailand","Kenya","Ethiopia","Morocco","Peru","Chile","Sweden",
        "Netherlands","Belgium","Switzerland","Austria","Portugal","Greece","Romania"]
_EB  = ["China","Russia","USA","UK","Spain","Italy","Denmark","Finland","Norway",
        "Ireland","Czech Republic","Hungary","Slovakia","Croatia","Bulgaria","Serbia",
        "Belarus","Moldova","Georgia","Azerbaijan","Kazakhstan","Uzbekistan","Myanmar"]
_EC  = ["Berlin","Rio","Tokyo","Delhi","Paris","Toronto","Sydney","Cairo","Seoul",
        "Lagos","Istanbul","Geneva","Brussels","Vienna","Nairobi","Bangkok","Hanoi",
        "Manila","Jakarta","Tehran","Riyadh","Dubai","Casablanca","Accra","Addis Ababa",
        "Bogota","Lima","Santiago","Lisbon","Warsaw","Prague","Budapest","Bucharest"]
_ED  = ["Arsenal","Bayern Munich","Los Angeles Lakers","New York Yankees","Boston Celtics",
        "Ajax","Flamengo","Chennai Super Kings","Barcelona","Liverpool","Real Madrid",
        "Manchester City","Juventus","Inter Milan","AC Milan","PSG","Dortmund","Porto",
        "Benfica","Celtic","Rangers","Galatasaray","Fenerbahce","River Plate","Boca Juniors"]
_EE  = ["Chelsea","Borussia Dortmund","Golden State Warriors","Boston Red Sox","Miami Heat",
        "PSV Eindhoven","Palmeiras","Mumbai Indians","Atletico Madrid","Tottenham","Napoli",
        "Lazio","Lyon","Marseille","Sevilla","Valencia","Villarreal","Sporting CP","Braga"]
_EF  = ["Müller","Ramos","LeBron James","Novak Djokovic","Roger Federer","Serena Williams",
        "Kylian Mbappe","Cristiano Ronaldo","Lionel Messi","Usain Bolt","Lewis Hamilton",
        "Naomi Osaka","Max Verstappen","Simona Halep","Rafael Nadal","Tiger Woods",
        "Neymar Jr","Kevin Durant","Stephen Curry","Virat Kohli","MS Dhoni","Kane Williamson"]
_EG  = ["football","tennis","cricket","basketball","athletics","swimming","cycling",
        "rugby","volleyball","boxing","golf","skiing","gymnastics","rowing","sailing"]
_EH  = ["Champions League","World Series","Grand Slam","Super Bowl","Tour de France",
        "FIFA World Cup","Stanley Cup","Wimbledon","US Open","Australian Open","French Open",
        "Olympic Games","Commonwealth Games","Asian Games","Pan American Games"]
_EI  = ["Apple","Google","Microsoft","Amazon","Meta","Netflix","Tesla","Samsung",
        "Intel","AMD","Nvidia","Qualcomm","IBM","Oracle","Salesforce","Uber","Airbnb",
        "Spotify","Twitter","Snapchat","TikTok","ByteDance","Alibaba","Tencent","Baidu"]
_EJ  = ["Alphabet","OpenAI","Anthropic","DeepMind","Palantir","Snowflake","Databricks",
        "Stripe","Coinbase","Robinhood","Block","Shopify","Zoom","Slack","Dropbox"]
_EK  = ["1.2","3.5","500","2.8","10","750","4.1","1.5","6.7","2.3","8.4","12","450","900"]
_EL  = ["Amazon","Siberian","Congo","Borneo","Himalayan","Patagonian","Antarctic","Saharan",
        "Sonoran","Atacama","Mongolian","Tibetan","Andean","Carpathian","Appalachian"]

_POOL = {
    "A": _EA, "B": _EB, "C": _EC, "D": _ED, "E": _EE,
    "F": _EF, "G": _EG, "H": _EH, "I": _EI, "J": _EJ,
    "K": _EK, "L": _EL,
}


def _fill(tpl: str, rng: np.random.Generator) -> str:
    for key, pool in _POOL.items():
        placeholder = f"{{{key}}}"
        while placeholder in tpl:
            tpl = tpl.replace(placeholder, str(rng.choice(pool)), 1)
    return tpl


def build_corpus(
    n: int = 5000,
    dim: int = DIM,
    seed: int = 42,
    vocab_size: int = VOCAB,
) -> Tuple[np.ndarray, List[str]]:
    rng    = np.random.default_rng(seed)
    topics = list(_T.keys())
    texts  = []
    seen   = set()
    per    = n // len(topics)
    rem    = n % len(topics)
    for ti, topic in enumerate(topics):
        count = per + (1 if ti < rem else 0)
        tpls  = _T[topic]
        added = 0
        tries = 0
        while added < count and tries < count * 20:
            tries += 1
            tpl  = tpls[rng.integers(0, len(tpls))]
            text = _fill(tpl, rng)
            if text not in seen:
                texts.append(text)
                seen.add(text)
                added += 1
        # fallback: allow duplicates if exhausted
        while added < count:
            tpl  = tpls[rng.integers(0, len(tpls))]
            texts.append(_fill(tpl, rng))
            added += 1

    idx = np.arange(len(texts))
    rng.shuffle(idx)
    texts = [texts[i] for i in idx]

    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        vecs[i] = bow_to_embedding(text_to_bow(t, vocab_size=vocab_size), dim=dim)

    return vecs, texts


def build_query_set(
    corpus_vecs: np.ndarray,
    corpus_texts: List[str],
    n_queries: int = 500,
    k: int = 10,
    seed: int = 99,
    vocab_size: int = VOCAB,
    dim: int = DIM,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    rng    = np.random.default_rng(seed)
    topics = list(_T.keys())
    seen   = set(corpus_texts)
    qtexts = []
    tries  = 0
    while len(qtexts) < n_queries and tries < n_queries * 20:
        tries += 1
        topic = topics[rng.integers(0, len(topics))]
        tpl   = _T[topic][rng.integers(0, len(_T[topic]))]
        text  = _fill(tpl, rng)
        if text not in seen:
            qtexts.append(text)
            seen.add(text)
    # fallback
    while len(qtexts) < n_queries:
        topic = topics[rng.integers(0, len(topics))]
        tpl   = _T[topic][rng.integers(0, len(_T[topic]))]
        qtexts.append(_fill(tpl, rng))

    qvecs = np.zeros((n_queries, dim), dtype=np.float32)
    for i, t in enumerate(qtexts):
        qvecs[i] = bow_to_embedding(text_to_bow(t, vocab_size=vocab_size), dim=dim)

    print(f"Computing exact ground truth for {n_queries} queries × {len(corpus_vecs)} corpus...")
    cn = corpus_vecs / (np.linalg.norm(corpus_vecs, axis=1, keepdims=True) + 1e-10)
    qn = qvecs      / (np.linalg.norm(qvecs,       axis=1, keepdims=True) + 1e-10)
    sims = qn @ cn.T
    ground_truth = np.argsort(-sims, axis=1)[:, :k]
    print("Ground truth computed.")
    return qvecs, qtexts, ground_truth


if __name__ == "__main__":
    import collections
    vecs, texts = build_corpus(n=5000, dim=DIM)
    unique = len(set(texts))
    print(f"Unique: {unique}/5000 ({unique/5000:.1%})")
    n0 = np.linalg.norm(vecs, axis=1)
    print(f"Norms: mean={n0.mean():.3f}")
    # similarity distribution
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 5000, size=(500,2))
    sims = [float(np.dot(vecs[a], vecs[b])) for a,b in idx if a!=b]
    print(f"Random pairs: mean={np.mean(sims):.3f} std={np.std(sims):.3f} max={max(sims):.3f}")
