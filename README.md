# Śledzenie obiektów z wykorzystaniem granulacji i głębokich sieci neuronowych

Projekt na przedmiot Studio Projektowe 2 na podstawie publikacji 'Granulated deep learning and Z-numbers in motion detection and object recognition'

Autorzy: Kacper Majchrzak, Mateusz Mazur

## Artykuł bazowy

[Granulated deep learning and Z-numbers in motion detection and object recognition](https://link.springer.com/article/10.1007/s00521-019-04200-1)

W artykule poruszono problematykę detekcji ruchu, rozpoznawania obiektów i opisu scen z wykorzystaniem głębokiego uczenia się w kontekście obliczeń granularnych. Ponieważ głębokie uczenie jest wymagające obliczeniowo, autorzy proponują zastosowanie obliczeń granularnych w celu zredukowania wymagań obliczeniowych.

## Podsumowanie

Nasze eksperymenty z połączeniem granulacji obrazów i detekcji ruchu przy użyciu głębokiego uczenia nie przyniosły oczekiwanych rezultatów. Przedstawione w analizowanej publikacji podejście, które zakładało zastosowanie granulacji przed lub zamiast warstwy konwolucyjnej w sieci SSD, okazało się problematyczne. Kluczowe wnioski z naszych badań to:

- **Problemy z transfer-learningiem**: Sieci SSD wytrenowane na standardowych obrazach nie potrafią efektywnie działać na obrazach granulowanych. Przetrenowanie sieci na granulowanych obrazach wymagałoby znacznych zasobów i czasu, a uzyskane wyniki prawdopodobnie byłyby niezadowalające.
- **Utrata szczegółowości**: Granulacja powoduje utratę istotnych szczegółów na obrazach, co utrudnia naukę i dokładność sieci neuronowych. Przykładem jest niewidoczne oko słonia na granulowanym obrazie.
- **Problemy z nieregularnymi granulkami**: Nieregularne granulki nie mogą być łatwo przetwarzane przez warstwy konwolucyjne, co ogranicza możliwości optymalizacji obliczeniowej.
- **Brak różniczkowalności**: Granulacja jest operacją nieliniową i nieróżniczkowalną, co uniemożliwia tworzenie trenowalnych warstw granulacyjnych w sieciach neuronowych.

### Alternatywne podejście

Podjęliśmy próbę zastosowania granulacji do wykrywania ruchu na obrazach w inny sposób. Nasze podejście polegało na:

- Obliczaniu granulacji obecnej i poprzedniej ramki,
- Porównywaniu granulacji między ramkami w celu wykrycia ruchu,
- Przekazywaniu wykrytych obszarów ruchu do sieci neuronowej do klasyfikacji obiektów.

To podejście okazało się skuteczniejsze, pozwalając na szybsze przetwarzanie obrazów i zwiększenie dokładności klasyfikacji. Jednak wraz ze wzrostem liczby obiektów na obrazie, wydajność (FPS) spadała, co wynikało z konieczności przetwarzania większej liczby obiektów przez sieć neuronową.

## Wnioski

Granulacja obrazów może być przydatna w procesie detekcji ruchu, jednak nie w formie zaprezentowanej w analizowanej publikacji. Alternatywne podejście, które łączy granulację z wykrywaniem ruchu, a następnie klasyfikacją obiektów, pokazuje pewne korzyści, ale również ujawnia ograniczenia związane z wydajnością przy dużej liczbie obiektów.