import string
import numpy as np
import networkx as nx
import random
import itertools
import scipy.stats
import matplotlib.pyplot as plt


def dico_alphabet() -> dict[str: int]:
    """
    Renvoie un dictionnaire permettant d'avoir l'indice associé à une lettre.

    Ici on utilisera aussi le "!" pour une fin de mot.
    """
    return {lettre: indice for indice, lettre in enumerate(string.ascii_lowercase+"!")}


def apprentissage_markov(sequences: set[str]) -> tuple[np.array]:
    """
    Prends un ensemble de séquence de mot et pour apprendre les paramètres.

    Puis renvoie le couple (PI, A).
    """
    alphabet = dico_alphabet()
    PI = np.zeros((1, 27), dtype=float)
    A = np.zeros((27, 27), dtype=float)
    A[26, 26] = 1

    for sequence in sequences:
        PI[0, alphabet[sequence[0]]] += 1

        for i in range(1, len(sequence)):
            A[alphabet[sequence[i-1]], alphabet[sequence[i]]] += 1

    for ligne in range(A.shape[0]):
        A[ligne, :] = A[ligne, :]/np.sum(A[ligne, :])

    return A, PI/np.sum(PI)


def logvraisemblance_markov(sequence: str, A: np.array, PI: np.array, epsilon : float = 1e-10) -> float:
    """"""
    alphabet = dico_alphabet()

    logvraisemblance = np.log(PI[0, alphabet[sequence[0]]])

    for i in range(1, len(sequence)):
        if A[alphabet[sequence[i-1]], alphabet[sequence[i]]] > 0:
            logvraisemblance += np.log(A[alphabet[sequence[i-1]], alphabet[sequence[i]]])
        else:
            logvraisemblance += np.log(epsilon)

    return float(logvraisemblance)


def graphe_markov(A: np.array) -> nx.Graph:
    """"""
    alphabet = dico_alphabet()
    alphabet_inv = {indice: lettre for lettre, indice in alphabet.items()}

    G = nx.DiGraph()

    for lettre in alphabet.keys():
        G.add_node(lettre)

    for ligne in range(A.shape[0]):
        for colonne in range(A.shape[1]):
            
            if A[ligne, colonne] > 0:
                G.add_edge(alphabet_inv[ligne], alphabet_inv[colonne], weight=A[ligne, colonne])

    
    return G


def generer(A: np.array, PI: np.array, limite: int = 100) -> str:
    """"""
    alphabet = dico_alphabet()
    alphabet_inv = {indice: lettre for lettre, indice in alphabet.items()}

    sequence = []

    dpi = {i: 0 for i in range(PI.shape[1])}
    da = {i: [0 for _ in range(PI.shape[1])] for i in range(PI.shape[1])}

    for i in range(PI.shape[1]):
        if i == 0:
            dpi[i] = float(PI[0, i])
        else:
            dpi[i] = float(PI[0, i] + dpi[i-1])
            if i == PI.shape[1] - 1:
                dpi[i] = 1 + 1.10e-3
        for j in range(PI.shape[1]):
            if j == 0:
                da[i][j] = float(A[i, j])
            else:
                da[i][j] = float(A[i, j] + da[i][j-1])
                if j == PI.shape[1] - 1:
                    da[i][j] = 1 + 1.10e-3

    for i in range(limite):
        proba = random.random()

        if i == 0:
            for classe, val in dpi.items():
                if proba < val:
                    sequence.append(classe)
                    break

        else:
            if alphabet_inv[sequence[-1]] == "!":
                break
            for classe, val in enumerate(da[sequence[-1]]):
                if proba < val:
                    sequence.append(classe)
                    break
    
    return "".join(list(map(lambda x: alphabet_inv[x], sequence)))


def categoriser(corpus: list[tuple[str, float]]) -> dict[int: dict[str: [list]]]:
    """"""
    taille = set()

    for mot in corpus:
        taille.add(len(mot[0])-1)

    categorie = {t: {"mot": [], "logvraisemblance": []} for t in taille}

    for data in corpus:
        mot, logv = data
        categorie[len(mot) - 1]["mot"].append(mot)
        categorie[len(mot) - 1]["logvraisemblance"].append(logv)

    return categorie


def dico_k_mers(K: int, alphabet: list[str] = list(string.ascii_lowercase)) -> dict[str, int]:
    """"""
    return {"".join(kmer): indice for indice, kmer in enumerate(itertools.product(alphabet+["!"], repeat=K))}


def apprentissage_markov_k_mers(sequences: set[str], K: int, alphabet: str = string.ascii_lowercase) -> tuple[np.array]:
    """"""
    alphabet = dico_k_mers(K, alphabet)
    PI = np.ones((1, len(alphabet)), dtype=float)
    A = np.ones((len(alphabet), len(alphabet)), dtype=float)

    # (!)* -> (!)*
    A[alphabet["!"*K], :] = 0
    A[alphabet["!"*K], alphabet["!"*K]] = 1

    for sequence in sequences:
        PI[0, alphabet[sequence[0: 0+K]]] += 1

        for i in range(1, len(sequence), K):
            mg = sequence[i-1: i-1+K]
            if i+K-1 > len(sequence):
                mg += "!"*((i+K-1) - len(sequence))

            md = sequence[i+K-1: i+2*K-1]
            if i+2*K-1 > len(sequence):
                if md == "":
                    md = "!"*K
                else:
                    md += "!"*((i+2*K-1) - len(sequence))

            A[alphabet[mg], alphabet[md]] += 1

    for ligne in range(A.shape[0]):
        A[ligne, :] = A[ligne, :]/np.sum(A[ligne, :])

    return A, PI/np.sum(PI)


def logvraisemblance_markov_k_mers(sequence: str, A: np.array, PI: np.array, K: int, epsilon: float = 1e-10, alphabet: str = string.ascii_lowercase) -> float:
    """"""
    alphabet = dico_k_mers(K, alphabet)

    if PI[0, alphabet[sequence[0: 0+K]]] == 0:
        logvraisemblance = np.log(epsilon)
    else:
        logvraisemblance = np.log(PI[0, alphabet[sequence[0: 0+K]]])

    for i in range(1, len(sequence), K):
        mg = sequence[i-1: i-1+K]
        if i+K-1 > len(sequence):
            mg += "!"*((i+K-1) - len(sequence))

        md = sequence[i+K-1: i+2*K-1]
        if i+2*K-1 > len(sequence):
            md += "!"*((i+2*K-1) - len(sequence))
        
        if A[alphabet[mg], alphabet[md]] > 0:
            logvraisemblance += np.log(A[alphabet[mg], alphabet[md]])
        else:
            logvraisemblance += np.log(epsilon)

    return float(logvraisemblance)

def generer_k_mers(A: np.array, PI: np.array, K: int, limite: int = 100, alphabet: str = string.ascii_lowercase) -> str:
    """"""
    alphabet = dico_k_mers(K, alphabet)
    alphabet_inv = {indice: lettre for lettre, indice in alphabet.items()}

    sequence = []

    dpi = {i: 0 for i in range(PI.shape[1])}
    da = {i: [0 for _ in range(PI.shape[1])] for i in range(PI.shape[1])}

    for i in range(PI.shape[1]):
        if i == 0:
            dpi[i] = float(PI[0, i])
        else:
            dpi[i] = float(PI[0, i] + dpi[i-1])
            if i == PI.shape[1] - 1:
                dpi[i] = 1 + 1.10e-3
        for j in range(PI.shape[1]):
            if j == 0:
                da[i][j] = float(A[i, j])
            else:
                da[i][j] = float(A[i, j] + da[i][j-1])
                if j == PI.shape[1] - 1:
                    da[i][j] = 1 + 1.10e-3

    for i in range(limite):
        proba = random.random()

        if i == 0:
            for classe, val in dpi.items():
                if proba < val:
                    sequence.append(classe)
                    break

        else:
            if alphabet_inv[sequence[-1]] == "!"*K:
                break
            for classe, val in enumerate(da[sequence[-1]]):
                if proba < val:
                    sequence.append(classe)
                    break
    
    return "".join(list(map(lambda x: alphabet_inv[x], sequence)))


def student(X: list, Y: list, alpha: float) -> bool:
    """"""
    n, m = len(X), len(Y)
    
    var_X = np.var(X, ddof=1)
    var_Y = np.var(Y, ddof=1)

    std = np.sqrt(((n - 1) * var_X + (m - 1) * var_Y) / (n + m - 2))
    t_stat = (np.mean(X) - np.mean(Y)) / (std * np.sqrt(1 / n + 1 / m))

    df = n + m - 2

    quantile = scipy.stats.t.ppf(1 - alpha / 2, df)

    return bool(abs(t_stat) > quantile)


def valeur_par_classe_sst3(sequences: list, classes: list) -> dict[str: list[str]]:
    """"""
    label = {"C": [], "E": [], "H": []}
    for sequence, classe in zip(sequences, classes):
        ss_seq = str(sequence[0])
        for i in range(1, len(sequence)):
            if classe[i-1] == classe[i]:
                ss_seq += sequence[i]
            else:
                label[classe[i-1]].append(ss_seq)
                ss_seq = str(sequence[i])
        label[classe[i]].append(ss_seq)
    
    return label


def classifie(sequence: str, modeles: dict[str: tuple[np.array]], K: int, alphabet: str,) -> str:
    """
    """
    resultat = {}
    for classe, modele in modeles.items():
        resultat[classe] = logvraisemblance_markov_k_mers(sequence, *modele, K, alphabet=alphabet)
    
    return max(resultat, key=resultat.get)


def confusion_structure_secondaire(X: np.array, Y: np.array, modeles: dict, K: int, alphabet: str) -> np.array:
    """"""
    label_test = valeur_par_classe_sst3(X, Y)

    confusion = np.zeros((3, 3))

    indice = {"C": 0, "E": 1, "H": 2}

    vocabulaire = {"C": "coil", "E": "sheet", "H": "helix"}

    for classe in label_test:
        erreurs, total = 0, 0
        for sequence in label_test[classe]:

            if len(sequence) > K:
                total += 1
                prediction = classifie(sequence, modeles, K, alphabet)
                
                if prediction != classe:
                    erreurs += 1

                confusion[indice[classe], indice[prediction]] += 1
                
        print(f"Pour la classe {classe} ({vocabulaire[classe]}), succès de {round((1-(erreurs/total))*100, 2)}% soit {erreurs} erreurs pour un total de {total}.")


    for ligne in range(confusion.shape[0]):
        confusion[ligne, :] /= sum(confusion[ligne, :])

    fig, ax = plt.subplots()
    im = ax.imshow(confusion, cmap="viridis")
    ax.set_xticks(np.arange(len(indice)), labels=vocabulaire.values())
    ax.set_yticks(np.arange(len(indice)), labels=vocabulaire.values())

    ax.set_title("Matrice de confusion pour la classification par MaxVraisemblance")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Proportion")

    for i in range(len(indice)):
        for j in range(len(indice)):
            ax.text(j, i, f"{confusion[i, j]:.2f}",
                        ha="center", va="center", color="red", fontsize=10)


    plt.show()
    plt.close()

    return confusion


def labelisation_ss_structure(classes: list):
    """"""
    base = []
    for classe in classes:
        gamma = 1
        ss_seq = f"S{gamma}"
        for i in range(1, len(classe)):
            if classe[i-1] == classe[i]:
                ss_seq += f"S{gamma}"
            else:
                gamma += 1
                ss_seq += f"S{gamma}"
        base.append(ss_seq)
    
    return base



def hmm_apprentissage_k_mers(etats: list[str], observations: list[str], K: int, alphabet_etats: list[str], alphabet_observations: list[str]) -> tuple[np.array]:
    """"""
    kalphabet_etats = dico_k_mers(K, alphabet_etats)
    kalphabet_observations = dico_k_mers(K, alphabet_observations)

    PI = np.ones((1, len(kalphabet_etats)), dtype=float)
    A = np.ones((len(kalphabet_etats), len(kalphabet_etats)), dtype=float)
    B = np.ones((len(kalphabet_etats), len(kalphabet_observations)), dtype=float)

    A[kalphabet_etats["!"*K], :] = 0
    A[kalphabet_etats["!"*K], kalphabet_etats["!"*K]] = 1

    for ss_structure, sequence in zip(etats, observations):
        PI[0, kalphabet_etats[ss_structure[0: 0+K]]] += 1

        for i in range(K, len(ss_structure), K):

            mgobs = sequence[i: i+K]
            mgetat = ss_structure[i: i+K]

            if i+K > len(ss_structure):
                mgobs += "!"*((i+K) - len(sequence))
                mgetat += "!"*((i+K) - len(ss_structure))

            mdetat = ss_structure[i+K: i+2*K]

            if i+2*K > len(ss_structure):
                if mdetat == "":
                    mdetat = "!"*K
                else:
                    mdetat += "!"*((i+2*K) - len(ss_structure))

            A[kalphabet_etats[mgetat], kalphabet_etats[mdetat]] += 1
            B[kalphabet_etats[mgetat], kalphabet_observations[mgobs]] += 1

    for ligne in range(A.shape[0]):
        A[ligne, :] = A[ligne, :]/np.sum(A[ligne, :])

    for ligne in range(B.shape[0]):
        B[ligne, :] = B[ligne, :]/np.sum(B[ligne, :])

    return A, PI/np.sum(PI), B


def hmm_vraisemblance_k_mers(observation: str, A: np.array, PI: np.array, B: np.array, K: int, alphabet_etats: list[str], alphabet_observations: list[str]) -> float:
    """"""
    kalphabet_etats = dico_k_mers(K, alphabet_etats)
    kalphabet_observations = dico_k_mers(K, alphabet_observations)

    alpha = {0: {kalphabet_etats[classe]: PI[0, kalphabet_etats[classe]]*B[kalphabet_etats[classe], kalphabet_observations[observation[0: K]]] for classe in kalphabet_etats.keys()}}

    for t in range(K, len(observation), K):
        alpha[t] = {}

        mgobs = observation[t: t+K]

        if t+K > len(observation):
            mgobs += "!"*((t+K) - len(observation))

        for classe in kalphabet_etats:

            alpha[t][kalphabet_etats[classe]] = 0

            for c2 in kalphabet_etats:
                alpha[t][kalphabet_etats[classe]] += alpha[t-K][kalphabet_etats[c2]]*A[kalphabet_etats[c2], kalphabet_etats[classe]]
            
            alpha[t][kalphabet_etats[classe]] *= B[kalphabet_etats[classe], kalphabet_observations[mgobs]]

    return float(sum(alpha[t].values()))


def viterbi(observation: str, A: np.array, PI: np.array, B: np.array, K: int, alphabet_etats: list[str], alphabet_observations: list[str]):
    """"""
    Ac = A.copy()

    kalphabet_etats = dico_k_mers(K, alphabet_etats)
    kalphabet_observations = dico_k_mers(K, alphabet_observations)

    kalphabet_etats_reverse = {value: key for key, value in kalphabet_etats.items()}

    delta = {0: {kalphabet_etats[classe]: PI[0, kalphabet_etats[classe]]*B[kalphabet_etats[classe], kalphabet_observations[observation[0: K]]] for classe in kalphabet_etats.keys()}}
    psi = {0: {kalphabet_etats[classe]: None for classe in kalphabet_etats.keys()}}

    for classe, indice in kalphabet_etats.items():
        if "!" in classe:
            Ac[indice, :] = 0

    for t in range(K, len(observation), K):
        delta[t] = {}
        psi[t] = {}

        mgobs = observation[t: t+K]

        if t+K > len(observation):
            mgobs += "!"*((t+K) - len(observation))

        for classe in kalphabet_etats:

            delta[t][kalphabet_etats[classe]] = 0
            MAX, st = -np.inf, 0

            for c2 in kalphabet_etats:
                if delta[t-K][kalphabet_etats[c2]]*Ac[kalphabet_etats[c2], kalphabet_etats[classe]] > MAX:
                    
                    MAX = delta[t-K][kalphabet_etats[c2]]*Ac[kalphabet_etats[c2], kalphabet_etats[classe]]
                    st = kalphabet_etats[c2]
            
            delta[t][kalphabet_etats[classe]] = MAX*B[kalphabet_etats[classe], kalphabet_observations[mgobs]]
            psi[t][kalphabet_etats[classe]] = st

    MAX, st = -np.inf, 0
    for classe, value in delta[t].items():
        if value > MAX:
            MAX = value
            st = classe
    seq = [st]

    while t >= 0:
        seq.append(psi[t][seq[-1]])
        t -= K

    return list(map(lambda x: kalphabet_etats_reverse[x].replace("!", ""), seq[::-1][1:]))


def interpolation_ss_structure(observation: str) -> str:
    """"""
    for i in range(2, len(observation)-1):
        if observation[i-1] == observation[i+1]:
            observation = observation[:i] + observation[i+1] + observation[i+1:]
    return observation


def ss_seq_contigs(etat: str, observation: str) -> tuple[list[str]]:
    """"""
    obscontigs, etacontigs = [], []
    ss_seqobs, ss_seqeta = observation[0], etat[0]
    for i in range(1, len(observation)):
        if etat[i] != etat[i-1]:
            obscontigs.append(ss_seqobs)
            etacontigs.append(ss_seqeta)
            ss_seqobs = observation[i]
            ss_seqeta = etat[i]
        else:
            ss_seqobs += observation[i]
            ss_seqeta += etat[i]
    
    obscontigs.append(ss_seqobs)
    etacontigs.append(ss_seqeta)

    return etacontigs, obscontigs