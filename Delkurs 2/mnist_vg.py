import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import streamlit as st
from PIL import ImageOps, Image, ImageEnhance, ImageFilter, ImageDraw
from streamlit_drawable_canvas import st_canvas
import warnings
import matplotlib.pyplot as plt

# Ignorera UserWarnings relaterade till sklearn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")


def preprocess_image(image, contrast_factor):
    """
    Förbättrar en inmatad bild av en handritad siffra för MNIST-modellen.
    - Behåller gråskala istället för hård binarisering.
    - Justerar kontrast och skärpa baserat på användarens val.
    - Säkerställer 28x28 storlek.
    - Implementerar robust normalisering för att undvika division med noll.
    """
    if image is None:
        raise ValueError("Ingen bild har laddats in.")

    # 1️⃣ Rätta till bildens orientering
    image = ImageOps.exif_transpose(image)

    # 2️⃣ Konvertera till gråskala
    gray_image = image.convert("L")

    # 3️⃣ Justera ljusstyrkan något
    brightness_enhancer = ImageEnhance.Brightness(gray_image)
    brightened_image = brightness_enhancer.enhance(1.1)  # Mild ljusjustering

    # 4️⃣ Öka kontrasten baserat på användarens slider-inställning
    contrast_enhancer = ImageEnhance.Contrast(brightened_image)
    contrast_image = contrast_enhancer.enhance(contrast_factor)

    # 5️⃣ Applicera Gaussian Blur för att mjuka upp konturer
    blurred_image = contrast_image.filter(ImageFilter.GaussianBlur(0.3))  # Lätt blur

    # 6️⃣ Konvertera till NumPy-array för vidare bearbetning
    np_image = np.array(blurred_image, dtype=np.float32)

    # 7️⃣ Normalisering med skydd mot division med noll
    min_val = np.min(np_image)
    max_val = np.max(np_image)

    if max_val - min_val == 0:
        np_image = np.zeros_like(np_image, dtype=np.uint8)  # Sätt alla pixlar till 0 (svart bild)
    else:
        np_image = np.clip(((np_image - min_val) / (max_val - min_val)) * 255, 0, 255)

    # 8️⃣ Se till att inga NaN eller inf-värden finns
    np_image = np.nan_to_num(np_image)
    np_image = np_image.astype(np.uint8)

    # 9️⃣ Skapa bilden efter att den har normaliserats
    processed_image = Image.fromarray(np_image)

    # 🔟 Skärp bilden lätt
    sharpness_enhancer = ImageEnhance.Sharpness(processed_image)
    final_image = sharpness_enhancer.enhance(1.3)  # Mild skärpa

    # 🔟 Skala om till exakt 28x28 pixlar (MNIST-format)
    mnist_image = final_image.resize((28, 28))

    return mnist_image


# Steg 1: Ladda och preprocessa MNIST-data
def load_and_preprocess_mnist(subsample_percentage=100, test_size=0.2):
    if not (0 <= subsample_percentage <= 100):
        raise ValueError("subsample_percentage måste vara mellan 0 och 100.")

    mnist = fetch_openml('mnist_784', version=1)
    X = pd.DataFrame(mnist.data / 255.0)  # Konvertera till DataFrame med kolumnnamn
    y = mnist.target.astype(int)

    if subsample_percentage < 100:
        X, _, y, _ = train_test_split(X, y, test_size=(1 - subsample_percentage / 100), random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, y_train, X_test, y_test


# Steg 2: Skapa pipelines för olika modeller
def get_pipelines():
    pipelines = {
        "Logistic Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=500, verbose=0))
        ]),
        "Random Forest": Pipeline([
            ('model', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, verbose=0))
        ]),
        "SVM": Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(probability=True, kernel='rbf', C=10, gamma=0.001, verbose=False))
        ])
    }
    return pipelines


# Steg 3: Skapa en VotingClassifier
def create_voting_ensemble():
    pipelines = get_pipelines()
    voting_ensemble = VotingClassifier(estimators=[
        ('log_reg', pipelines["Logistic Regression"]),
        ('random_forest', pipelines["Random Forest"]),
        ('svm', pipelines["SVM"])
    ], voting='soft')
    return voting_ensemble


# Steg 4: Träna modell
def train_model(model_name, X_train, y_train, optimize=False):
    if model_name == "Voting Ensemble":
        model = create_voting_ensemble()
        model.fit(X_train, y_train)
        joblib.dump(model, "voting_ensemble_model.pkl")
        st.write("Voting Ensemble är tränad och sparad!")
        return

    pipelines = get_pipelines()
    pipeline = pipelines[model_name]

    if optimize:
        # 🔹 Uppdaterade parameterrum för optimering
        param_distributions = {
            "Logistic Regression": {
                'model__C': [0.01, 0.1, 1, 10],  # Minskad lista med C-värden
                'model__penalty': ['l2'],  # Endast l2-penalty
                'model__solver': ['lbfgs'],  # Snabb solver för Logistic Regression
                'model__max_iter': [500]  # Begränsat till 500 iterationer
            },
            "SVM": {
                'model__C': [0.1, 1, 10, 100],  # Fler värden för C
                'model__gamma': [0.001, 0.01, 0.1, 1],  # Fler gamma-värden
                'model__kernel': ['rbf', 'linear']  # Ytterligare kernel
            },
            "Random Forest": {
                'model__n_estimators': [50, 100, 200, 300],  # Fler träd
                'model__max_depth': [10, 20, 30, None],  # Fler maxdjup
                'model__min_samples_split': [2, 5, 10],  # Fler urval
                'model__bootstrap': [True, False],  # Bootstrap ja/nej
                'model__max_features': ['sqrt', 'log2']  # Ytterligare parameter
            }
        }

        # 🔹 Begränsa datasetet för optimering (25% av träningdatan)
        X_train_opt, _, y_train_opt, _ = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42
        )

        # 🔹 RandomizedSearchCV med förbättrade inställningar
        search = RandomizedSearchCV(
            pipeline,
            param_distributions[model_name],
            n_iter=20,  # Testa fler kombinationer
            cv=5,  # Fler korsvalideringsfolders
            n_jobs=-1,  # Använd alla CPU-kärnor
            random_state=42,
            verbose=10  # Visa förloppet
        )
        
        # 🔹 Träna med optimerad dataset
        search.fit(X_train_opt, y_train_opt)

        # 🔹 Spara bästa modellen
        best_pipeline = search.best_estimator_
        joblib.dump(best_pipeline, f"{model_name.lower().replace(' ', '_')}_optimized_model.pkl")
        st.write(f"Optimerad {model_name} är tränad och sparad!")
        st.write("Bästa parametrar:")
        st.json(search.best_params_)

    else:
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, f"{model_name.lower().replace(' ', '_')}_model.pkl")
        st.write(f"{model_name} är tränad och sparad!")


# Steg 5: Ladda modell
def load_model(model_name, optimized=False):
    try:
        if model_name == "Voting Ensemble":
            model_file = "voting_ensemble_model.pkl"
        else:
            model_file = f"{model_name.lower().replace(' ', '_')}_optimized_model.pkl" if optimized else f"{model_name.lower().replace(' ', '_')}_model.pkl"
        model = joblib.load(model_file)
        return model
    except FileNotFoundError:
        st.error(f"Modellen {model_name} kunde inte laddas. Träna modellen först.")
        return None


# Steg 6: Utvärdera modell med förvirringsmatris
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Modellens noggrannhet på testdata: {accuracy:.3f}")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    st.pyplot(plt.gcf())


# Steg 7: Streamlit-applikation
def streamlit_app():
    st.title("MNIST Sifferigenkänning med Pipelines och Optimering")

    st.sidebar.header("Inställningar")
    test_size = st.sidebar.slider("Andel testdata", min_value=0.1, max_value=0.5, step=0.1, value=0.2)
    subsample_percentage = st.sidebar.slider("Andel av datasetet att använda (%)", min_value=10, max_value=100, step=10, value=100)
    model_choice = st.sidebar.selectbox("Välj modell", ["Logistic Regression", "Random Forest", "SVM", "Voting Ensemble"])
    optimize = st.sidebar.checkbox("Optimering av hyperparametrar")
    use_optimized = st.sidebar.checkbox("Använd optimerad modell om tillgänglig")
    input_method = st.sidebar.radio("Välj inmatningsmetod", ["Rita på canvas", "Ladda upp bild"])

    # Lägg till en slider i sidopanelen för kontrastjustering
    contrast_factor = st.sidebar.slider("Justera kontrast", min_value=0.5, max_value=4.0, step=0.1, value=2.5)

    # Ladda data
    X_train, y_train, X_test, y_test = load_and_preprocess_mnist(subsample_percentage=subsample_percentage, test_size=test_size)

    # Träna modell
    if st.sidebar.button("Träna modell"):
        train_model(model_choice, X_train, y_train, optimize=optimize)

    # Ladda vald modell
    model = load_model(model_choice, optimized=use_optimized)
    if model is None:
        return

    if use_optimized and model_choice != "Voting Ensemble":
        st.write(f"Optimerad {model_choice} laddades och används för prediktion.")
    else:
        st.write(f"{model_choice} laddades och används för prediktion.")

    # 🔹 Titel för inmatningsmetod
    st.header("Inmatningsmetod")

    # 🔹 Skapa kolumner för layout
    col1, col2 = st.columns(2)

    # 🔹 Se till att `image` alltid existerar
    image = None  

    if input_method == "Rita på canvas":
        with col1:
            st.subheader("Rita en siffra")
            canvas_result = st_canvas(
                fill_color="#2E2E3E",
                stroke_width=10,
                stroke_color="Red",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas"
            )
        if canvas_result.image_data is not None:
            image = Image.fromarray((canvas_result.image_data[:, :, 3] > 0).astype(np.uint8) * 255)

    elif input_method == "Ladda upp bild":
        with col1:
            st.subheader("Ladda upp en bild")
            uploaded_file = st.file_uploader("📂 Ladda upp en bild här:", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            st.write("✅ Fil laddad:", uploaded_file.name)
            image = Image.open(uploaded_file)

    # 🔹 Om ingen bild finns, visa varning och stoppa koden
    if image is None:
        st.warning("⚠️ Ingen bild har laddats upp eller ritats! Ladda upp en fil ovan.")
        st.stop()

    # ✅ Förbättra och förbered bilden (RÄTT INDENTERAT!)
    processed_image = preprocess_image(image, contrast_factor)  # ✅ Skickar med kontrastvärdet

    with col2:
        st.image(processed_image.resize((280, 280)), caption="Förbättrad & Normaliserad Bild", use_container_width=True)

    # 🔹 Konvertera till MNIST-format
    data = (np.array(processed_image) / 255.0).reshape(1, -1)
    data = pd.DataFrame(data, columns=X_train.columns)

    # ✅ Kontrollera att modellen existerar innan prediktion
    if model is None:
        st.error("❌ Ingen modell är laddad! Träna modellen först.")
        st.stop()

    # 🔹 Gör prediktion
    prediction = model.predict(data)
    probabilities = model.predict_proba(data)
    st.write(f"Modellen förutsäger: {prediction[0]}")

    # 🔹 Visa sannolikhetsdiagram
    chart_data = pd.DataFrame(probabilities[0], columns=["Probability"], index=list(range(10)))
    st.bar_chart(chart_data, use_container_width=True)

    # 🔹 Utvärdera modellen på testdata
    if st.sidebar.button("Utvärdera modell"):
        st.header("Utvärdering av modell")
        evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    streamlit_app()
