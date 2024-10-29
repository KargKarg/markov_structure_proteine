import string
import numpy as np
import networkx as nx
import random
import itertools
import scipy.stats

import scipy.stats


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


def dico_k_mers(K: int, alphabet: list = string.ascii_lowercase) -> dict[str, int]:
    """"""
    return {"".join(kmer): indice for indice, kmer in enumerate(itertools.product(alphabet+"!", repeat=K))}


def apprentissage_markov_k_mers(sequences: set[str], K: int, alphabet: str = string.ascii_lowercase) -> tuple[np.array]:
    """"""
    alphabet = dico_k_mers(K, alphabet)
    PI = np.zeros((1, 27**K), dtype=float)
    A = np.zeros((27**K, 27**K), dtype=float)

    # (!)* -> (!)*
    A[27**K - 1, 27**K - 1] = 1

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
        if np.sum(A[ligne, :]) == 0:
            # PseudoCount
            A[ligne, :] += 1
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
