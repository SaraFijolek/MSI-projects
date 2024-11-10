document.addEventListener('DOMContentLoaded', function() {
    console.log("JavaScript załadowany poprawnie!");

    document.querySelector('form').addEventListener('submit', function(event) {
        const numberInput = document.getElementById('number');
        const numberValue = parseFloat(numberInput.value);

        // Sprawdzamy, czy liczba jest dodatnia i całkowita
        if (numberValue <= 0 || !Number.isInteger(numberValue)) {
            alert("Podaj liczbę całkowitą większą od zera!");
            event.preventDefault();  // Zapobiega wysłaniu formularza
        }
    });
});
