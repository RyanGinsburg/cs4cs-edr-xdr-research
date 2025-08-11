with open('sherlock.txt', 'r') as file:
    content = file.read()

contnent = content.lower()
words = content.split()

markov = {}
for i in range(len(words) - 1):
    word = words[i]
    next_word = words[i + 1]
    if word not in markov:
        markov[word] = []
    markov[word].append(next_word)