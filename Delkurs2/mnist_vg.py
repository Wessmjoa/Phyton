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
    if image is None:
        raise ValueError("Ingen bild har laddats in.")

    # 1️⃣ Rätta till bildens orientering
    image = ImageOps.exif_transpose(image)

    # 2️⃣ Konvertera till gråskala
    gray_image = image.convert("L")

    # 3️⃣ Justera ljusstyrkan något
    brightness_enhancer = ImageEnhance.Brightness(gray_image)
    brightened_image = brightness_enhancer.enhance(1.1)

    # 4️⃣ Öka kontrasten baserat på slider-inställning
    contrast_enhancer = ImageEnhance.Contrast(brightened_image)
    contrast_image = contrast_enhancer.enhance(contrast_factor)

    # 5️⃣ Applicera Gaussian Blur
    blurred_image = contrast_image.filter(ImageFilter.GaussianBlur(0.3))

    # 6️⃣ Konvertera till NumPy-array
    np_image = np.array(blurred_image, dtype=np.float32)

    # 7️⃣ Normalisering
    min_val = np.min(np_image)
    max_val = np.max(np_image)

    if max_val - min_val == 0:
        np_image = np.zeros_like(np_image, dtype=np.uint8)
    else:
        np_image = np.clip(((np_image - min_val) / (max_val - min_val)) * 255, 0, 255)

    # 8️⃣ Säkerställ giltiga värden
    np_image = np.nan_to_num(np_image)
    np_image = np_image.astype(np.uint8)

    # 9️⃣ Skapa final image
    processed_image = Image.fromarray(np_image)

    # 🔟 Skärp bilden
    sharpness_enhancer = ImageEnhance.Sharpness(processed_image)
    final_image = sharpness_enhancer.enhance(1.3)

    # 🔟 Skala om till 28x28
    mnist_image = final_image.resize((28, 28))

    return mnist_image

def load_and_preprocess_mnist(subsample_percentage=100, test_size=0.2):
    if not (0 <= subsample_percentage <= 100):
        raise ValueError("subsample_percentage måste vara mellan 0 och 100.")

    mnist = fetch_openml('mnist_784', version=1)
    X = pd.DataFrame(mnist.data / 255.0)
    y = mnist.target.astype(int)

    if subsample_percentage < 100:
        X, _, y, _ = train_test_split(X, y, test_size=(1 - subsample_percentage / 100), random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, y_train, X_test, y_test

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

def create_voting_ensemble():
    pipelines = get_pipelines()
    voting_ensemble = VotingClassifier(estimators=[
        ('log_reg', pipelines["Logistic Regression"]),
        ('random_forest', pipelines["Random Forest"]),
        ('svm', pipelines["SVM"])
    ], voting='soft')
    return voting_ensemble

def train_model(model_name, X_train, y_train, optimize=False, validation_size=0.25):
    if model_name == "Voting Ensemble":
        model = create_voting_ensemble
        model.fit(X_train, y_train)
        joblib.dump(model, "voting_ensemble_model.pkl", compress=3)
        st.write("Voting Ensemble är tränad och sparad!")
        return

    pipelines = get_pipelines()
    pipeline = pipelines[model_name]

    if optimize:
        param_distributions = {
            "Logistic Regression": {
                'model__C': [0.01, 0.1, 1, 10],  # Minskad lista med C-värden
                'model__penalty': ['l2'],  # Endast l2-penalty
                'model__solver': ['lbfgs'],  # Snabb solver för Logistic Regression
                'model__max_iter': [500]  # Begränsat till 500 iterationer
            },
            "SVM": {
                'model__C': [0.1, 1, 10, 100],
                'model__gamma': [0.001, 0.01, 0.1, 1],
                'model__kernel': ['rbf', 'linear']
            },
            "Random Forest": {
                'model__n_estimators': [50, 100, 200, 300],
                'model__max_depth': [10, 20, 30, None],
                'model__min_samples_split': [2, 5, 10],
                'model__bootstrap': [True, False],
                'model__max_features': ['sqrt', 'log2']
            }
        }

        # Använd valideringsstorlek från slider
        X_train_opt, _, y_train_opt, _ = train_test_split(
            X_train, y_train, 
            test_size=validation_size, 
            random_state=42
        )

        search = RandomizedSearchCV(
            pipeline,
            param_distributions[model_name],
            n_iter=20,
            cv=5,
            n_jobs=-1,
            random_state=42,
            verbose=10
        )
        

        search.fit(X_train_opt, y_train_opt)

        best_pipeline = search.best_estimator_
        joblib.dump(best_pipeline, f"{model_name.lower().replace(' ', '_')}_optimized_model.pkl")
        st.write(f"Optimerad {model_name} är tränad och sparad!")
        st.write("Bästa parametrar:")
        st.json(search.best_params_)

    else:
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, f"{model_name.lower().replace(' ', '_')}_model.pkl")
        st.write(f"{model_name} är tränad och sparad!")

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

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Modellens noggrannhet på testdata: {accuracy:.3f}")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    st.pyplot(plt.gcf())

def streamlit_app():
    st.title("MNIST Sifferigenkänning med Pipelines och Optimering")

    st.sidebar.header("Inställningar")
    test_size = st.sidebar.slider("Andel testdata", 0.1, 0.5, 0.2, 0.1)
    validation_size = st.sidebar.slider("Valideringsdata för optimering", 0.1, 0.4, 0.25, 0.05)
    #st.sidebar.markdown("ℹ️ **Testdata** används för slutlig utvärdering, **valideringsdata** för hyperparameterjustering")
    
    subsample_percentage = st.sidebar.slider("Använt dataset (%)", 10, 100, 100, 10)
    model_choice = st.sidebar.selectbox("Välj modell", ["Random Forest", "Logistic Regression", "SVM", "Voting Ensemble"])
    optimize = st.sidebar.checkbox("Optimera hyperparametrar")
    use_optimized = st.sidebar.checkbox("Använd optimerad modell")
    input_method = st.sidebar.radio("Inmatningsmetod", ["Rita på canvas", "Ladda upp bild"])
    contrast_factor = st.sidebar.slider("Kontrastjustering", 0.5, 4.0, 2.5, 0.1)

    # Ladda data
    X_train, y_train, X_test, y_test = load_and_preprocess_mnist(
        subsample_percentage=subsample_percentage, 
        test_size=test_size
    )

    # Träna modell
    if st.sidebar.button("Träna modell"):
        train_model(
            model_choice, 
            X_train, 
            y_train, 
            optimize=optimize,
            validation_size=validation_size
        )

    # Ladda modell
    model = load_model(model_choice, optimized=use_optimized)
    if model is None:
        return

    # Visa modellstatus
    model_status = f"Optimerad {model_choice}" if use_optimized and model_choice != "Voting Ensemble" else model_choice
    st.write(f"{model_status} laddades och används för prediktion.")

    # Bildhantering
    st.header("Sifferinmatning")
    col1, col2 = st.columns(2)
    image = None

    if input_method == "Rita på canvas":
        with col1:
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
            uploaded_file = st.file_uploader("Ladda upp sifferbild", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)

    if image is None:
        st.warning("Vänligen rita eller ladda upp en bild först")
        st.stop()

    # Bildförbehandling
    processed_image = preprocess_image(image, contrast_factor)
    with col2:
        st.image(processed_image.resize((280, 280)), caption="Förbehandlad bild")

    # Prediktion
    data = (np.array(processed_image) / 255.0).reshape(1, -1)
    data = pd.DataFrame(data, columns=X_train.columns)
    
    prediction = model.predict(data)
    probabilities = model.predict_proba(data)
    st.write(f"**Prediktion:** {prediction[0]}")

    # Sannolikhetsvisualisering
    st.bar_chart(pd.DataFrame({
        "Siffra": range(10),
        "Sannolikhet": probabilities[0]
    }).set_index("Siffra"))

    # Utvärdering
    if st.sidebar.button("Utvärdera modell"):
        st.header("Modellutvärdering")
        evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    streamlit_app()