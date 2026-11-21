NEGATIONS = {"tidak", "kurang", "belum", "bukan", "tanpa", "tak"}
INTENSIFIERS = {"sangat", "banget", "sekali", "terlalu", "cukup", "lumayan", "agak"}

def is_punct(w):
    return not any(c.isalnum() for c in w)

def word_shape(word):
    result = []

    for char in word:
        if char.isupper():
            result.append('X')
        elif char.islower():
            result.append('x')
        elif char.isdigit():
            result.append('d')
        else:
            result.append(char)
    return ''.join(result)


def token2features(tokens, i):
    word = tokens[i]
    features = {
        'bias': 1.0,
        'word.lower': word.lower(),
        'prefix2': word[:2].lower(),
        'prefix3': word[:3].lower(),
        'suffix2': word[-2:].lower(),
        'suffix3': word[-3:].lower(),
        'is_upper': word.isupper(),
        'is_title': word.istitle(),
        'is_digit': word.isdigit(),
        'contains_digit': any(c.isdigit() for c in word),
        'contains_hyphen': '-' in word,
        'word_shape': word_shape(word),
        'BOS': i == 0,
        'EOS': i == len(tokens) - 1
    }
    if i > 0:
        prev = tokens[i - 1]
        features.update({
        '-1:word.lower': prev.lower(),
        '-1:is_title': prev.istitle(),
        '-1:word_shape': word_shape(prev)
    })
    if i < len(tokens) - 1:
        nxt = tokens[i + 1]
        features.update({
        '+1:word.lower': nxt.lower(),
        '+1:is_title': nxt.istitle(),
        '+1:word_shape': word_shape(nxt)
    })

    return features

def sentence2features(X):
    return [[token2features(s, i) for i in range(len(s))] for s in X]