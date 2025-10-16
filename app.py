import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import streamlit as st
import os
import re


# Télécharger les ressources NLTK (une seule fois)
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)


download_nltk_resources()


# -----------------------------
# Charger et parser les conversations
# -----------------------------
@st.cache_data
def load_conversations(file_path):
    """Parse le fichier pour extraire les paires User/Bot"""
    if not os.path.exists(file_path):
        st.error(f"Le fichier '{file_path}' n'existe pas!")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    conversations = []

    # Pattern pour trouver les paires User/Bot
    # Cherche **User:** ... **Bot:** ...
    pattern = r'\*\*User:\*\*\s*(.*?)\s*\*\*Bot:\*\*\s*(.*?)(?=\*\*User:|$)'
    matches = re.findall(pattern, content, re.DOTALL)

    for user_msg, bot_msg in matches:
        user_msg = user_msg.strip()
        bot_msg = bot_msg.strip()
        if user_msg and bot_msg:
            conversations.append({
                'user': user_msg,
                'bot': bot_msg
            })

    return conversations


def preprocess_text(text):
    """Prétraite un texte"""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    punctuations = string.punctuation

    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words
             if word not in stop_words and word not in punctuations and word.isalnum()]
    return words


def calculate_similarity(text1, text2):
    """Calcule la similarité entre deux textes"""
    words1 = set(preprocess_text(text1))
    words2 = set(preprocess_text(text2))

    if not words1 or not words2:
        return 0

    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))

    return intersection / union if union > 0 else 0


def find_best_response(user_input, conversations):
    """Trouve la meilleure réponse basée sur la similarité"""
    if not user_input.strip():
        return "Veuillez poser une question."

    best_match = None
    best_similarity = 0

    for conv in conversations:
        similarity = calculate_similarity(user_input, conv['user'])

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = conv

    # Seuils de confiance
    if best_similarity > 0.6:  # Excellente correspondance
        return best_match['bot']
    elif best_similarity > 0.3:  # Bonne correspondance
        return best_match['bot']
    elif best_similarity > 0.1:  # Correspondance faible
        return f"{best_match['bot']}\n\n_(Réponse avec faible confiance - reformulez si ce n'est pas ce que vous cherchiez)_"
    else:
        return "Je ne suis pas sûr de comprendre votre question. Pourriez-vous la reformuler différemment?"


def get_greeting_response(user_input):
    """Détecte et répond aux salutations courantes"""
    input_lower = user_input.lower().strip()

    greetings = {
        'hi': "Hello! How can I help you today?",
        'hello': "Hello! Nice to see you. What can I do for you?",
        'hey': "Hey there! How can I assist you?",
        'good morning': "Good morning! I hope you're having a wonderful start to your day. How can I help you?",
        'good evening': "Good evening! I hope you've had a productive day. What brings you here?",
        'good afternoon': "Good afternoon! How can I assist you today?",
        'bonjour': "Bonjour! Comment puis-je vous aider?",
        'salut': "Salut! Que puis-je faire pour vous?"
    }

    for greeting, response in greetings.items():
        if greeting in input_lower:
            return response

    # Questions sur l'état
    status_questions = ['how are you', 'how r u', 'comment vas-tu', 'ça va', 'cv']
    for question in status_questions:
        if question in input_lower:
            return "I'm doing well, thank you for asking! I'm here and ready to help. How are you doing today?"

    return None


def chatbot(user_input, conversations):
    """Fonction principale du chatbot avec logique améliorée"""
    if not user_input.strip():
        return "Veuillez poser une question."

    # D'abord vérifier les salutations communes
    greeting_response = get_greeting_response(user_input)
    if greeting_response:
        return greeting_response

    # Sinon, chercher dans la base de données
    return find_best_response(user_input, conversations)


def main():
    st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")

    st.title("🤖 Chatbot Intelligent")
    st.markdown("---")

    # Charger les conversations
    file_path = 'text.txt'
    conversations = load_conversations(file_path)

    if not conversations:
        st.error("❌ Impossible de charger les conversations. Vérifiez le fichier 'text.txt'")
        st.stop()

    st.success(f"✅ {len(conversations)} paires de conversation chargées!")

    # Initialiser l'historique
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Zone de chat
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("💬 Conversation")

        # Formulaire de saisie
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input("Votre message:", placeholder="Tapez votre message ici...", key="input")
            col_a, col_b = st.columns([1, 5])
            with col_a:
                submit = st.form_submit_button("📤 Envoyer", use_container_width=True)

        # Traiter le message
        if submit and user_input:
            with st.spinner("Réflexion en cours..."):
                response = chatbot(user_input, conversations)

            st.session_state.history.append({
                'user': user_input,
                'bot': response
            })

        # Afficher l'historique
        if st.session_state.history:
            st.markdown("---")
            for i, exchange in enumerate(reversed(st.session_state.history)):
                with st.container():
                    # Message utilisateur
                    st.markdown(f"**🧑 Vous:**")
                    st.info(exchange['user'])

                    # Réponse du bot
                    st.markdown(f"**🤖 Bot:**")
                    st.success(exchange['bot'])

                    if i < len(st.session_state.history) - 1:
                        st.markdown("---")
        else:
            st.info("👋 Commencez la conversation en envoyant un message!")

    with col2:
        st.subheader("⚙️ Contrôles")

        if st.button("🔄 Nouvelle conversation", use_container_width=True):
            st.session_state.history = []
            st.rerun()

        st.markdown("---")
        st.subheader("📊 Statistiques")
        st.metric("Messages envoyés", len(st.session_state.history))
        st.metric("Base de données", f"{len(conversations)} réponses")

        st.markdown("---")
        st.subheader("💡 Exemples")
        st.markdown("""
        - Good morning!
        - How are you?
        - Can you help me?
        - Tell me a joke
        - I'm feeling sad
        - What's the weather like?
        """)


if __name__ == "__main__":
    main()