from flask import Flask, request, render_template
import re
from transformers import AutoTokenizer, AutoModelWithLMHead

app = Flask(__name__, static_url_path='/static')

def syllable_count(word):
    count = 0
    syllables = set("aeiou")
    for letter in word:
        if letter in syllables:
            count = count + 1
    if (word[-2:] == "es" or word[-2:] == "ed"):
      count = count - 1
    return count
def analyse(text):
    sw_file = open("resource/sw_file", "r", encoding="ISO-8859-1")
    stopWords = sw_file.read()
    stopWords = stopWords.replace('\n', ',').split(",")
    sw_file.close()
    punctuation = [',', '.', '!', '?', '/', ';', ':', '@', '#', '$', '%', '^', '&', '*', '’', '”', '“']
    # To get list of pronouns used in the article
    pronounRegex = re.compile(r'\b(I|we|my|ours|(?-i:us))\b', re.I)
    pronouns = pronounRegex.findall(text)
    text = str(text)
    # Count number of sentences
    line_count = 0
    for t in text:
        if t == ".":
            line_count += 1
    # Tokanize the text
    text = text.lower().split(' ')

    # Removes stopwords and punctuation from the text tokens
    text = [t for t in text if t not in stopWords and t not in punctuation]
    # Calculate positive score
    pos_file = open("resource/positive-words", "r", encoding="ISO-8859-1")
    pos_words = pos_file.read()
    pos_words = pos_words.replace('\n', ',').split(",")
    pos_file.close()
    pos_score = 0
    for t in text:
        if (t in pos_words):
            pos_score += 1

    # Calculate negative score
    neg_file = open("resource/negative-words", "r", encoding="ISO-8859-1")
    neg_words = neg_file.read()
    neg_words = neg_words.replace('\n', ',').split(",")
    neg_file.close()
    neg_score = 0
    for t in text:
        if t in neg_words:
            neg_score -= 1
            neg_score = neg_score * (-1)

    # Calculate Polarity score
    polarity_score = (pos_score - neg_score) / (pos_score + neg_score + 0.000001)
    # Calculate Subjectivity score
    sub_score = (pos_score + neg_score) / (len(text) + 0.000001)

    # Simultaneously Calculate number of complex words, sum of syllables per word and average word length
    complex_words = 0
    word = 0
    syllable = 0
    for t in text:
        word += len(t)
        syllable += syllable_count(t)
        if syllable_count(t) > 2:
            complex_words += 1
    analysis = [pos_score, neg_score, polarity_score, sub_score, len(text)/line_count, complex_words/len(text), (0.4*((len(text)/line_count)+(complex_words/len(text)))), complex_words, len(text), syllable/len(text), len(pronouns), word/len(text)]
    return analysis

def summarize(text):
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    model = AutoModelWithLMHead.from_pretrained('t5-base', return_dict=True)
    inputs = tokenizer.encode("summarize: " + text,
                              return_tensors='pt',
                              max_length=512,
                              truncation=True)
    summary_ids = model.generate(inputs, max_length=500, min_length=100, length_penalty=5., num_beams=2)
    summary = tokenizer.decode(summary_ids[0])
    print(summary)
    return summary[5:-4]

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/documentation')
def doc():
    return render_template('documentation.html')

@app.route('/main', methods=['GET','POST'])
def main():
    text = request.args.get('content')
    analysis = analyse(text)
    summary = summarize(text)
    return render_template('result.html', pos=analysis[0], neg=analysis[1], pol=analysis[2],
                               sub=analysis[3], sent=analysis[4], comp=analysis[5], fog=analysis[6], comp_word=analysis[7],
                               word=analysis[8], syl=analysis[9],
                               pro=analysis[10], word_len=analysis[11], summary=summary)


if __name__ == '__main__':
    app.run()
