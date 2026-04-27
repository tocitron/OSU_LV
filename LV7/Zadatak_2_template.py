import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans


# ------------------------------------------------------------
# Učitavanje i prikaz slike
# ------------------------------------------------------------

slika = Image.imread("imgs\\test_5.jpg")

plt.figure()
plt.title("Izvorna slika")
plt.imshow(slika)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Priprema podataka
# ------------------------------------------------------------

# normalizacija RGB vrijednosti na interval [0,1]
slika = slika.astype(np.float64) / 255

# pretvaranje slike u matricu (piksel = red, RGB = stupci)
visina, sirina, kanali = slika.shape
podatci = slika.reshape(visina * sirina, kanali)

# kopija za kasniju obradu
podatci_aprox = podatci.copy()


# ------------------------------------------------------------
# 1) broj različitih boja
# ------------------------------------------------------------

razlicite_boje = np.unique(podatci, axis=0)
print("Ukupan broj različitih boja:", len(razlicite_boje))


# ------------------------------------------------------------
# 2) K-Means grupiranje
# ------------------------------------------------------------

K = 5
model = KMeans(n_clusters=K, random_state=0)
oznake = model.fit_predict(podatci)
centroidi = model.cluster_centers_


# ------------------------------------------------------------
# 3) zamjena boja centroidima
# ------------------------------------------------------------

podatci_aprox = centroidi[oznake]
slika_aprox = podatci_aprox.reshape(visina, sirina, kanali)

# sigurnosno ograničenje vrijednosti
slika_aprox = np.clip(slika_aprox, 0, 1)

plt.figure()
plt.title("Kvantizirana slika")
plt.imshow(slika_aprox)
plt.show()


# ------------------------------------------------------------
# 6) Elbow metoda (J = inertia)
# ------------------------------------------------------------

vrijednosti_J = []
raspon_K = range(1, 11)

for k in raspon_K:
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(podatci)
    vrijednosti_J.append(km.inertia_)

plt.figure()
plt.plot(raspon_K, vrijednosti_J, marker='o')
plt.xlabel("Broj klastera K")
plt.ylabel("Vrijednost funkcije J")
plt.title("Elbow metoda")
plt.show()


# ------------------------------------------------------------
# 7) prikaz pojedinih klastera kao binarne slike
# ------------------------------------------------------------

K_final = 5

for i in range(K_final):
    maska = (oznake == i)
    binarna = maska.reshape(visina, sirina)

    plt.figure()
    plt.title(f"Klaster {i}")
    plt.imshow(binarna, cmap='gray')
    plt.show()


# ------------------------------------------------------------
# Analiza rezultata
# ------------------------------------------------------------

# U ovom primjeru korišten je K-Means za smanjenje broja boja u slici.
# Ideja je svesti velik broj nijansi na manji broj reprezentativnih boja.

# Svaki piksel promatran je kao trodimenzionalna točka (R, G, B).
# Time se slika pretvara u skup podataka gdje svaki red predstavlja jedan piksel.

# Funkcija np.unique pokazuje da originalna slika sadrži vrlo velik broj boja,
# što je tipično za fotografije zbog sitnih varijacija u osvjetljenju i teksturi.

# Nakon primjene K-Means algoritma određuje se K dominantnih boja.
# Svaki piksel dobiva boju najbližeg centroida, čime se smanjuje ukupna raznolikost boja.

# Veći K znači bolju sličnost s originalom jer se koristi više nijansi.
# Manji K pojednostavljuje sliku, ali uz gubitak detalja i pojavu oštrih prijelaza.

# Graf funkcije J (inertia) pokazuje kako pogreška opada s povećanjem K.
# Točka gdje pad postaje sporiji (tzv. "lakat") može sugerirati dobar izbor K.

# Binarne slike klastera otkrivaju koje dijelove slike pokriva pojedina boja.
# Često se vidi da neki klaster predstavlja pozadinu, a drugi objekte ili sjene.


# ------------------------------------------------------------
# Kritički osvrt
# ------------------------------------------------------------

# K-Means je jednostavan i brz način za kvantizaciju boje.
# Dobro funkcionira kada je cilj smanjiti kompleksnost slike.

# Nedostatak je što koristi euklidsku udaljenost u RGB prostoru,
# koji ne odgovara savršeno ljudskoj percepciji boja.

# Također, algoritam ne uzima u obzir položaj piksela u slici,
# pa može doći do neprirodnih prijelaza između područja.

# Rezultat jako ovisi o odabiru K.
# Premali K uklanja detalje, dok preveliki smanjuje efekt kompresije.

# Inicijalizacija centroida također može utjecati na rezultat,
# iako random_state djelomično stabilizira ponašanje.


# ------------------------------------------------------------
# Moguća poboljšanja
# ------------------------------------------------------------

# Umjesto ručnog odabira K može se koristiti Elbow metoda ili Silhouette analiza.

# Bolji rezultati često se postižu korištenjem prostora boja kao što su LAB ili HSV,
# jer su bliži ljudskom doživljaju boje.

# Dodavanje prostornih koordinata (x, y) u podatke moglo bi poboljšati kontinuitet slike.

# Alternativno, mogu se isprobati i druge metode poput GMM ili DBSCAN.

# Također je korisno testirati algoritam na različitim tipovima slika
# kako bi se vidjelo gdje daje najbolje rezultate.