import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from flask import Flask, render_template, request

app = Flask(__name__)

# Funkcja sprawdzająca, czy liczba jest pierwsza
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

# Generowanie danych treningowych dla parzystości i pierwszości
def generate_data(limit=1000):
    X = np.array(range(1, limit + 2)).reshape(-1, 1)  # Zaczynamy od 1
    y_parity = np.array([1 if x % 2 == 0 else 0 for x in X.flatten()])  # 1 dla parzystych, 0 dla nieparzystych
    y_prime = np.array([1 if is_prime(x) else 0 for x in X.flatten()])  # 1 dla pierwszych, 0 dla złożonych
    y = np.column_stack((y_parity, y_prime))  # Łączymy etykiety parzystości i pierwszości w jedną macierz
    return X, y

# Tworzenie modelu sieci neuronowej
def create_model():
    model = Sequential([
        Dense(64, input_dim=1, activation='relu'),
        Dense(32, activation='relu'),
        Dense(2, activation='sigmoid')  # Dwa wyjścia: parzystość i pierwszość
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Sprawdzenie poprawności generowanych danych
X, y = generate_data(1000)
print("Przykładowe dane wejściowe i etykiety:")
for i in range(len(X)):
    print(f"Liczba: {X[i][0]}, Parzystość: {'parzysta' if y[i][0] == 1 else 'nieparzysta'}, Pierwszość: {'pierwsza' if y[i][1] == 1 else 'złożona'}")

# Trenowanie modelu
model = create_model()
model.fit(X, y, epochs=1000, batch_size=10, verbose=1)

# Flask app to load the model and make predictions
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            number = int(request.form['number'])
            if number <= 0:
                raise ValueError("Liczba musi być dodatnia i całkowita")
        except ValueError:
            result = {"error": "Podaj liczbę całkowitą większą od zera!"}
        else:
            # Przewidywanie parzystości i pierwszości
            prediction = model.predict(np.array([[number]]))

            # Surowy wynik predykcji (wydrukowanie wartości przed progiem 0.5)
            print(f"Surowy wynik predykcji dla liczby {number}: {prediction}")

            # Przekształcenie wyników na klasy
            is_even = prediction[0][0] > 0.5  # 1 dla parzystych
            is_prime = prediction[0][1] > 0.5 # 1 dla pierwszych

            # Formatuj odpowiedź
            result = {
                'number': number,
                'even': "parzysta" if is_even else "nieparzysta",
                'prime': "pierwsza" if is_prime else "złożona"
            }
    return render_template('index.html', result=result)

# Start the Flask application
if __name__ == '__main__':
    app.run(debug=True)
