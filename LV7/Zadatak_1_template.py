import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans


def napravi_podatke(broj_uzoraka, tip_podataka):
    """
    Funkcija generira umjetne podatke za prikaz i testiranje algoritama grupiranja.
    Vraća numpy polje X, gdje prvi stupac predstavlja x1, a drugi stupac x2.
    """

    if tip_podataka == 1:
        # tri jasno odvojene grupe
        X, _ = make_blobs(
            n_samples=broj_uzoraka,
            random_state=365
        )

    elif tip_podataka == 2:
        # tri grupe koje su dodatno linearno transformirane
        X, _ = make_blobs(
            n_samples=broj_uzoraka,
            random_state=148
        )

        matrica_transformacije = np.array([
            [0.60834549, -0.63667341],
            [-0.40887718, 0.85253229]
        ])

        X = X @ matrica_transformacije

    elif tip_podataka == 3:
        # četiri grupe različite raspršenosti
        X, _ = make_blobs(
            n_samples=broj_uzoraka,
            centers=4,
            cluster_std=[1.0, 2.5, 0.5, 3.0],
            random_state=148
        )

    elif tip_podataka == 4:
        # dvije grupe u obliku koncentričnih kružnica
        X, _ = make_circles(
            n_samples=broj_uzoraka,
            factor=0.5,
            noise=0.05
        )

    elif tip_podataka == 5:
        # dvije grupe u obliku polumjeseca
        X, _ = make_moons(
            n_samples=broj_uzoraka,
            noise=0.05
        )

    else:
        raise ValueError("Tip podataka mora biti cijeli broj od 1 do 5.")

    return X


# ------------------------------------------------------------
# Odabir postavki
# ------------------------------------------------------------

broj_uzoraka = 500

# Mijenjanjem ove vrijednosti od 1 do 5 dobivaju se različiti oblici podataka
tip = 1

# Optimalan broj grupa ovisi o tipu podataka:
# tip 1 -> 3 grupe
# tip 2 -> 3 grupe
# tip 3 -> 4 grupe
# tip 4 -> 2 grupe
# tip 5 -> 2 grupe
K = 3


# ------------------------------------------------------------
# Generiranje i prikaz početnih podataka
# ------------------------------------------------------------

podaci = napravi_podatke(broj_uzoraka, tip)

plt.figure()
plt.scatter(podaci[:, 0], podaci[:, 1])
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Generirani podatkovni primjeri")
plt.show()


# ------------------------------------------------------------
# Primjena K-Means algoritma
# ------------------------------------------------------------

model = KMeans(
    n_clusters=K,
    init="random",
    n_init=5,
    random_state=0
)

oznake_grupa = model.fit_predict(podaci)


# ------------------------------------------------------------
# Prikaz rezultata grupiranja
# ------------------------------------------------------------

plt.figure()
plt.scatter(
    podaci[:, 0],
    podaci[:, 1],
    c=oznake_grupa
)

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Rezultat K-Means grupiranja")
plt.show()


# ------------------------------------------------------------
# Analiza i komentar rezultata
# ------------------------------------------------------------

# U ovom zadatku korišten je K-Means algoritam za grupiranje umjetno generiranih podataka.
# Funkcija napravi_podatke() omogućuje izradu nekoliko različitih rasporeda točaka,
# čime se može promatrati kako algoritam radi u jednostavnijim i složenijim slučajevima.

# Kada je tip_podataka = 1, dobivaju se tri jasno odvojene grupe.
# Takav raspored je vrlo pogodan za K-Means jer su grupe kompaktne, približno kružnog oblika
# i međusobno dobro razdvojene. U tom slučaju algoritam uglavnom daje vrlo dobre rezultate.

# Za tip_podataka = 2 također postoje tri prirodne grupe, ali su one izdužene i zakrenute
# zbog primjene linearne transformacije. K-Means u ovom slučaju može biti manje precizan
# jer algoritam najbolje radi kada su klasteri približno sferni.

# Kod tip_podataka = 3 generiraju se četiri grupe različitih veličina i različite raspršenosti.
# Ovdje se vidi ograničenje K-Means algoritma jer klasteri različite gustoće mogu utjecati
# na položaj centroida. Veće i raspršenije grupe mogu uzrokovati slabiju podjelu podataka.

# Kada je tip_podataka = 4, podaci imaju oblik dvije koncentrične kružnice.
# Iako se vizualno jasno mogu uočiti dvije grupe, K-Means ih ne razdvaja dobro.
# Razlog je taj što algoritam koristi udaljenost od centroida i stvara linearne granice.

# Kod tip_podataka = 5 dobivaju se dvije grupe u obliku polumjeseca.
# Kao i kod kružnica, K-Means ne daje dobar rezultat jer se radi o nelinearnom rasporedu podataka.
# Takvi oblici nisu prikladni za ovaj algoritam.

# Promjenom broja K može se primijetiti da premala vrijednost spaja više stvarnih grupa u jednu,
# dok prevelika vrijednost dijeli postojeće grupe na manje dijelove.
# Najbolji rezultat se dobiva kada je K jednak stvarnom broju grupa u podacima.

# Također se može uočiti da rezultat može ovisiti o početnom položaju centroida.
# Budući da je u programu korištena slučajna inicijalizacija, različita pokretanja mogu dati
# malo drugačije rezultate, posebno kod podataka koji nisu jasno razdvojeni.

# K-Means je zato najprikladniji za podatke koji su dobro odvojeni, slične veličine
# i približno kružnog oblika. Kod složenijih struktura, poput kružnica i polumjeseca,
# bolje bi bilo koristiti algoritme kao što su DBSCAN ili hijerarhijsko grupiranje.

# Za poboljšanje rješenja mogao bi se koristiti k-means++ umjesto slučajne inicijalizacije,
# jer on obično daje stabilnije rezultate. Osim toga, optimalan broj klastera mogao bi se
# odrediti pomoću Elbow metode ili Silhouette Score metrike.