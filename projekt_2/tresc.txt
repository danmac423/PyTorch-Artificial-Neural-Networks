Treść zadania:

Załóżmy, że chcemy kupić mieszkanie. Do dyspozycji mamy 100 000 dolarów, możemy też wziąć kredyt na kolejne 250 tysięcy, co da nam w sumie budżet w wysokości 350 000 dolarów. Stwórzmy model który pomoże nam przewidzieć, czy mieszkanie o pewnych, zadanych parametrach, możemy kupić za własne pieniądze (cheap), z kredytem (average), czy jest poza naszym zasięgiem (expensive).

W oparciu o dostępne atrybuty zbuduj model, który pomoże oszacować, czy dana nieruchomość należy do klasy cheap, average czy expensive. Do dyspozycji mają Państwo dane treningowe (train_data.csv) z oryginalnymi cenami nieruchomości (SalePrice), oraz zbiór testowy (test_data.csv).
Końcowe wyniki obliczać będę w oparciu o średnią dokładność (accuracy) dla każdej klasy. Proszę zwrócić uwagę na fakt, że klasy są mocno niezbalansowane!

UWAGA Proszę dokładnie zastosować się do poniższej instrukcji. Proszę sprawdzić dokładnie czy każdy podpunkt się zgadza!

- W ramach rozwiązania, proszę oddać poprzez Teamsy jeden plik .zip zawierający: kod (w formie notebooka, lub skryptu/skryptów .py) oraz plik .csv z predykcjami na zbiorze test_data.csv.
- Bardzo proszę nazwać plik .zip nazwiskami i imionami obu autorów z grupy ALFABETYCZNIE. Nazwę głównego archiwum z.ip proszę dodatkowo rozpocząć od przedrostka poniedzialek_ lub piatek_ lub sroda_ (NIE pon/pia/śr /inne wersje).  Przykład: sroda_KowalAndrzej_ZowalHanna.zip
- Proszę nie umieszczać plików w dodatkowych podfolderach tylko bezpośrednio.
- Proszę plik z predykcjami nazwać pred.csv
- W pliku z predykcjami powinna się znajdować dokładnie jedna kolumna, oznaczająca przewidywaną przez Państwa klasę ceny mieszkania (0 <- cheap, 1 <- average, 2 <- expensive). Koniecznie proszę sprawdzić format zwracanych przez Państwa predykcji (tyle predykcji ile elementów w zbiorze testowym, brak nagłówków, jedna kolumna, itd.)
- W MS Teams wszystkim przydzieliłam zadanie, ale bardzo proszę, żeby tylko jeden (dowolny) członek zespołu je zwrócił.

Niezastosowanie się do instrukcji może skutkować obniżeniem punktacji - ewaluacja wyników jest automatyczna, niespójne nazwy i pliki mogą spowodować złe wczytanie plików do testowania!

W razie pytań zapraszam do korespondencji lub na konsultacje.