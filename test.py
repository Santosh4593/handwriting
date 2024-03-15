import pytesseract
from PIL import Image
import nltk
from nltk.corpus import words
import spacy
from spellchecker import SpellChecker

# Download NLTK words if not already downloaded
nltk.download('words')

# Load English words corpus
english_words = set(words.words())

# Load SpaCy English model with word vectors
nlp = spacy.load("en_core_web_md")

# Initialize SpellChecker
spell = SpellChecker()

def extract_text(image_path):
    # Open the image file
    img = Image.open(image_path)
    
    # Use Tesseract to do OCR on the image
    return pytesseract.image_to_string(img)

def correct_spelling(word):
    # If the word is correct, return it
    if word.lower() in english_words:
        return word
    else:
        # Check if the word is misspelled
        corrected_word = spell.correction(word)
        # Return the corrected word if it's not None, otherwise return the original word
        return corrected_word if corrected_word else word

def find_similar_word(word):
    # Find the most similar word using SpaCy word vectors
    try:
        similar_word = nlp.vocab[word.lower()].text
        return similar_word
    except KeyError:
        # If the word is not found in the vocabulary, return the original word
        return word

def extract_and_find_similar_words(image_path):
    # Extract text from the image
    extracted_text = extract_text(image_path)

    # Correct spelling for each word in the extracted text
    corrected_text = ' '.join([correct_spelling(word) for word in extracted_text.split()])

    # Find similar words for each word in the corrected text
    similar_words = []
    for word in corrected_text.split():
        similar_words.append(find_similar_word(word))

    # Convert the list of similar words into a single string
    similar_sentence = ' '.join(similar_words)

    return extracted_text, similar_sentence

if __name__ == "__main__":
    # Path to the handwritten image file
    image_path = 'patil.jpg'

    # Extract text from the image and find similar words
    extracted_text, similar_sentence = extract_and_find_similar_words(image_path)

    # Print the extracted text and similar sentence
    print("Extracted Text:", extracted_text)
    print("Similar Sentence:", similar_sentence)
