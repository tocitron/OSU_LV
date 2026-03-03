rjecnik={}

with open("song.txt", "r", encoding="utf-8") as datoteka:
    for red in datoteka:
        rijeci = red.lower().split()
        for rijec in rijeci:
            rjecnik[rijec] = rjecnik.get(rijec, 0)+1

jednom = []

for rijec in rjecnik:
    if rjecnik[rijec]==1:
        jednom.append(rijec)

print("Broj riječi koje se pojavljuju samo jednom:", len(jednom))
print("Riječi:")
for rijec in jednom:
    print(rijec)