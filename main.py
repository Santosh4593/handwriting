import matplotlib.pyplot as plt
import keras_ocr
import spacy
from PIL import Image
import nltk
from nltk.corpus import words
from spellchecker import SpellChecker

# Load English language model from spaCy
nlp = spacy.load("en_core_web_md")

# Initialize SpellChecker
spell = SpellChecker()

# Download NLTK words if not already downloaded
nltk.download('words')

# Load English words corpus
english_words = set(words.words())

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

def ocr_image(image_path):
    # Load the recognizer and detector models
    pipeline = keras_ocr.pipeline.Pipeline()

    # Load the image
    images = [keras_ocr.tools.read(image_path)]

    # Perform OCR
    predictions = pipeline.recognize(images)

    return predictions[0], images[0]  # Return both predictions and the image

def get_similar_sentence(text):
    # Process the text with spaCy
    doc = nlp(text)

    # Extract the lemma of each token and join them into a sentence
    similar_sentence = ' '.join(token.lemma_ for token in doc)

    return similar_sentence

if __name__ == "__main__":
    # Path to the image file
    image_path = 'hand3.jpg'

    # Perform OCR on the image
    result, image = ocr_image(image_path)

    # Extract text from OCR result and apply spelling correction and finding similar words
    corrected_text = ' '.join(correct_spelling(word[0]) for word in result)
    similar_text = ' '.join(find_similar_word(word) for word in corrected_text.split())

    # Get similar sentence using spaCy
    similar_sentence = get_similar_sentence(similar_text)

    # Display the OCR result, corrected text, and similar sentence
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    keras_ocr.tools.drawAnnotations(image=image, predictions=result, ax=ax)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    print("OCR Result:", result)
    print("Corrected Text:", corrected_text)
    print("Similar Sentence:", similar_sentence)
